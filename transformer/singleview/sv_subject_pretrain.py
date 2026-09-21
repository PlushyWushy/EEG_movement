"""
Subject-conditioned front encoder and its contrastive, non-causal pretraining.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))

from train_multiview_transformer import make_pad_mask


class SubjectFrontEncoder(nn.Module):
    """Per-subject nonlinear feature extraction on the raw sensor signal,
    learned self-supervised and then frozen. Sits in front of ViewEncoder,
    which it leaves untouched.

    A pointwise spatial bottleneck: mix the 64 electrodes down to `hidden`,
    modulate that with a per-subject scale/shift, rectify, mix back up. Being
    pointwise across channels is what lets it learn a per-subject
    re-referencing -- the part of subject variability that is genuinely
    spatial (electrode placement, impedance, head geometry) and that nothing
    downstream can recover, since ViewEncoder's spatial conv collapses the
    channel axis to 1 immediately after.

    The ReLU is the point: it makes this feature extraction rather than the
    affine whitening an earlier version of this module did. It also becomes
    the network's FIRST spiking nonlinearity once converted, which is what
    costs ViewEncoder's cached prologue -- see SpikingSubjectFront.

    `up` is zero-initialised and the transform is residual, so an untrained
    front encoder is EXACTLY the identity: turning the flag on without
    pretraining cannot change a single downstream number.
    """

    def __init__(self, n_channels, n_subjects, hidden=16):
        super().__init__()
        self.down = nn.Conv1d(n_channels, hidden, 1, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Conv1d(hidden, n_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.subj_scale = nn.Embedding(n_subjects, hidden)
        self.subj_shift = nn.Embedding(n_subjects, hidden)
        nn.init.ones_(self.subj_scale.weight)
        nn.init.zeros_(self.subj_shift.weight)

    def condition(self, x, sid):
        """Everything up to (not including) the ReLU. Linear in x given sid,
        so the spiking path can evaluate it once on the constant input."""
        gain = self.subj_scale(sid).unsqueeze(-1)
        shift = self.subj_shift(sid).unsqueeze(-1)
        return gain * self.down(x) + shift

    def forward(self, x, sid):                 # x: (N, C, L), sid: (N,)
        return x + self.up(self.act(self.condition(x, sid)))

    def freeze(self):
        """Point 3: after pretraining nobody's parameters move again, so a
        held-out subject's embedding and a training subject's embedding were
        fit by the same objective. Letting the shared weights keep training on
        the supervised task would drift them toward the training subjects and
        undo the invariance the adversary just bought."""
        self.requires_grad_(False)
        self.eval()


class SpikingSubjectFront:
    """SubjectFrontEncoder run as a spiking stage, for phase 2.

    Its ReLU becomes a subtractive-reset IF neuron with a calibrated
    threshold, exactly as ViewEncoder's act1/act2 do -- so the nonlinearity
    is performed with spikes rather than left as an analog shortcut, and the
    network is spiking all the way from the first nonlinearity onward.

    `condition` is linear in the input and the input is constant across
    timesteps, so it is evaluated ONCE here. What is NOT free is downstream:
    this stage emits a different signal every timestep, so ViewEncoder's own
    linear prologue can no longer be cached and must be recomputed per step.
    That is the real price of putting a spiking nonlinearity at the very
    front, and it is why SpikingViewEncoder.start() takes a dynamic source.
    """

    def __init__(self, front, threshold, x, sid):
        self.front, self.thr, self.x = front, threshold, x
        self.cur = front.condition(x, sid)
        self.mem = torch.zeros_like(self.cur)

    @torch.no_grad()
    def step(self):
        self.mem = (self.mem + self.cur).clamp(min=0.0)
        spk = (self.mem >= self.thr).float()
        self.mem = self.mem - spk * self.thr
        return self.x + self.front.up(spk * self.thr)


@torch.no_grad()
def calibrate_subject_threshold(front, loader, device, n_batches, percentile):
    """One threshold for the front encoder's ReLU, taken the same way
    calibrate_thresholds takes ViewEncoder's: a high percentile of the
    pre-activation over unaugmented training chunks."""
    front.eval()
    peak = 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        xb, lb, sid = batch[0].to(device), batch[2], batch[3].to(device)
        B, T, C, L = xb.shape
        valid = (~make_pad_mask(lb, T, device)).reshape(-1)
        cur = front.condition(xb.reshape(B * T, C, L)[valid],
                              sid.repeat_interleave(T)[valid])
        peak = max(peak, float(torch.quantile(cur.flatten().float(),
                                              percentile / 100.0)))
    if peak <= 0.0:
        print("  WARNING: subject front never activated during calibration; "
              "flooring its threshold")
        peak = 1e-4
    return peak


class _GradReverse(torch.autograd.Function):
    """Identity forwards, sign-flipped backwards: the discriminator below
    descends its own loss while the front encoder ascends it."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class _SubjectDiscriminator(nn.Module):
    """Reads only per-channel mean and standard deviation, which is exactly
    the per-subject amplitude/offset structure the front encoder is supposed
    to be normalising away -- handing it richer features would let it win on
    task content instead, which is not what we want penalised."""

    def __init__(self, n_channels, n_subjects, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * n_channels, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_subjects))

    def forward(self, x):                      # (N, C, L) -> (N, n_subjects)
        return self.net(torch.cat([x.mean(-1), x.std(-1)], dim=-1))


class _SSLEncoder(nn.Module):
    """Throwaway chunk encoder: gives the contrastive task the nonlinear
    capacity the linear front encoder deliberately lacks, so the pretext
    problem is not trivially linear. Discarded when pretraining ends -- only
    SubjectFrontEncoder survives."""

    def __init__(self, n_channels, d_model, n_samples, f1=8, depth=2):
        super().__init__()
        f2 = f1 * depth
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, (1, 65), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2), nn.ReLU(), nn.AvgPool2d((1, 8)),
            nn.Conv2d(f2, f2, (1, 15), padding="same", groups=f2, bias=False),
            nn.BatchNorm2d(f2), nn.ReLU(), nn.AvgPool2d((1, 8)),
        )
        self.proj = nn.Linear(f2 * (n_samples // 64), d_model)

    def forward(self, x):                      # (N, C, L) -> (N, d_model)
        return self.proj(self.net(x.unsqueeze(1)).flatten(1))


def pretrain_subject_front(front, loader, device, epochs, lr, mask_frac,
                           adv_weight, n_subjects, n_channels, n_samples,
                           d_model=128, heads=4, layers=2, max_len=64,
                           temperature=0.1):
    """Contrastive, NON-CAUSAL masked pretraining of the front encoder.

    Two deliberate departures from the existing `pretrain()` in
    train_multiview_transformer.py, both of which were the suspected reasons
    that one underperformed:

    * The context transformer runs with NO causal mask. Masked positions see
      the whole run in both directions, so early positions are not asked to
      reconstruct themselves from a history that does not exist yet.
    * The target is a LATENT embedding, scored by InfoNCE against every other
      valid chunk in the batch, not the raw 640-sample waveform under MSE.
      Picking the right embedding out of a few hundred candidates cannot be
      won by predicting sample-level noise, so capacity is not spent there.

    The adversarial term is the second half of the subject story: the front
    encoder is conditioned ON subject identity at the input and penalised FOR
    subject identity at the output, so it learns to use who someone is in
    order to remove how they differ.
    """
    ssl_enc = _SSLEncoder(n_channels, d_model, n_samples).to(device)
    disc = _SubjectDiscriminator(n_channels, n_subjects).to(device)
    mask_token = nn.Parameter(torch.zeros(d_model, device=device))
    nn.init.normal_(mask_token, std=0.02)
    pos = nn.Embedding(max_len, d_model).to(device)
    nn.init.normal_(pos.weight, std=0.02)
    layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                       batch_first=True, norm_first=True)
    context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False).to(device)
    predictor = nn.Linear(d_model, d_model).to(device)

    params = (list(front.parameters()) + list(ssl_enc.parameters())
              + list(disc.parameters()) + list(context.parameters())
              + list(predictor.parameters()) + list(pos.parameters()) + [mask_token])
    opt = torch.optim.AdamW(params, lr=lr)

    front.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tot_nce, tot_adv, tot_acc, n_batches = 0.0, 0.0, 0.0, 0
        for batch in loader:
            xb, lb, sid = batch[0].to(device), batch[2].to(device), batch[3].to(device)
            B, T, C, L = xb.shape
            pad_mask = make_pad_mask(lb, T, device)
            valid = ~pad_mask
            if not valid.any():
                continue

            sid_flat = sid.repeat_interleave(T)
            h = front(xb.reshape(B * T, C, L), sid_flat)

            keep = valid.reshape(-1)
            adv_logits = disc(_GradReverse.apply(h[keep], adv_weight))
            adv_loss = F.cross_entropy(adv_logits, sid_flat[keep])

            z = ssl_enc(h).reshape(B, T, d_model)
            target = z.detach()                       # stop-grad target

            sel = (torch.rand(B, T, device=device) < mask_frac) & valid
            if not sel.any():
                continue
            inp = torch.where(sel.unsqueeze(-1), mask_token.expand(B, T, -1), z)
            inp = inp + pos(torch.arange(T, device=device)).unsqueeze(0)
            # No attn mask: this is the non-causal half of the design.
            ctx = context(inp, src_key_padding_mask=pad_mask)

            pred = F.normalize(predictor(ctx)[sel], dim=-1)
            cand = F.normalize(target[valid], dim=-1)  # every valid chunk is a candidate
            # Row of each masked chunk inside `cand`, which is `valid` flattened.
            slot = (torch.cumsum(keep.long(), 0) - 1).reshape(B, T)
            nce = F.cross_entropy(pred @ cand.t() / temperature, slot[sel])

            loss = nce + adv_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            tot_nce += nce.item()
            tot_adv += adv_loss.item()
            tot_acc += (adv_logits.argmax(1) == sid_flat[keep]).float().mean().item()
            n_batches += 1

        d = max(n_batches, 1)
        # Discriminator accuracy is the number to watch: 1/n_subjects means the
        # front encoder has made subjects indistinguishable, which is the point.
        print(f"  subj-pretrain epoch {ep:3d}  infonce={tot_nce / d:.4f}  "
              f"adv_ce={tot_adv / d:.4f}  disc_acc={tot_acc / d:.4f} "
              f"(chance {1 / n_subjects:.4f})  ({time.time() - t0:.0f}s)", flush=True)

    front.freeze()
    return front
