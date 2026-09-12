
"""
SNN multiview, spiking attention
"""
import argparse
import sys
import time
from pathlib import Path

import mne
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import surrogate
from snntorch import utils as snn_utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mlp.train_mlp import (
    CLASS_TO_IDX,
    CLASSES,
    DATA_ROOT,
    EPOCH_TMAX,
    EPOCH_TMIN,
    N_SAMPLES,
    RUN_LABELS,
    subject_dirs,
)

CACHE_DIR = ROOT / ".cache"

# The 64 channels as they appear in the EDFs (trailing dots stripped), in a
# fixed canonical order so view indices below are stable constants.
CANONICAL_CHANNELS = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1", "Oz", "O2", "Iz",
]

# Five views. Sensorimotor pair is mirrored (same channel geometry, opposite
# hemisphere) which is what makes weight-tying them meaningful.
VIEWS = {
    # ERD contralateral to RIGHT-hand movement
    "sensorimotor_left": ["C5", "C3", "C1", "Fc5", "Fc3", "Fc1", "Cp5", "Cp3", "Cp1"],
    # ERD contralateral to LEFT-hand movement
    "sensorimotor_right": ["C6", "C4", "C2", "Fc6", "Fc4", "Fc2", "Cp6", "Cp4", "Cp2"],
    # foot/leg representation is medial -- separates both_feet from both_fists
    "central_midline": ["Fz", "Fcz", "Cz", "Cpz", "Pz"],
    # premotor/SMA planning + prefrontal (also carries blink, left in on purpose)
    "frontal": ["Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz", "Af4", "Af8",
                "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8", "Ft7", "Ft8"],
    # proprioceptive feedback + visual; weakest view, kept for completeness
    "posterior": ["T7", "T8", "T9", "T10", "Tp7", "Tp8",
                  "P7", "P5", "P3", "P1", "P2", "P4", "P6", "P8",
                  "Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz"],
}
VIEW_NAMES = list(VIEWS)
TIED_VIEWS = ("sensorimotor_left", "sensorimotor_right")

# Runs group by which label triple they contain; the split keeps both groups
# represented in val and test so all 5 classes appear in every split.
RUN_GROUPS = {
    "lr": [4, 8, 12],   # baseline / left_fist / right_fist
    "ff": [6, 10, 14],  # baseline / both_fists / both_feet
}

_view_idx = {v: np.array([CANONICAL_CHANNELS.index(c) for c in chs])
             for v, chs in VIEWS.items()}
assert sorted(c for chs in VIEWS.values() for c in chs) == sorted(CANONICAL_CHANNELS), \
    "views must partition the 64 channels exactly once"

# Left/right electrode counterparts. The montage is symmetric, so reflecting
# the scalp across the midline is a physically meaningful relabelling of the
# data: a right-hand trial mirrored IS a left-hand trial. See MIRROR_INDEX /
# LABEL_MIRROR use in RunSequenceDataset.
MIRROR_PAIRS = [
    ("Fc5", "Fc6"), ("Fc3", "Fc4"), ("Fc1", "Fc2"),
    ("C5", "C6"), ("C3", "C4"), ("C1", "C2"),
    ("Cp5", "Cp6"), ("Cp3", "Cp4"), ("Cp1", "Cp2"),
    ("Fp1", "Fp2"), ("Af7", "Af8"), ("Af3", "Af4"),
    ("F7", "F8"), ("F5", "F6"), ("F3", "F4"), ("F1", "F2"),
    ("Ft7", "Ft8"), ("T7", "T8"), ("T9", "T10"), ("Tp7", "Tp8"),
    ("P7", "P8"), ("P5", "P6"), ("P3", "P4"), ("P1", "P2"),
    ("Po7", "Po8"), ("Po3", "Po4"), ("O1", "O2"),
]

MIRROR_INDEX = np.arange(len(CANONICAL_CHANNELS))
for _a, _b in MIRROR_PAIRS:
    _ia, _ib = CANONICAL_CHANNELS.index(_a), CANONICAL_CHANNELS.index(_b)
    MIRROR_INDEX[_ia], MIRROR_INDEX[_ib] = _ib, _ia
assert sorted(MIRROR_INDEX.tolist()) == list(range(len(CANONICAL_CHANNELS))), \
    "mirror map must be a permutation of the 64 channels"

# Mirroring the scalp swaps the lateralized classes and leaves the
# symmetric ones (both_fists, both_feet, baseline) alone.
LABEL_MIRROR = np.arange(len(CLASSES))
LABEL_MIRROR[CLASS_TO_IDX["left_fist"]] = CLASS_TO_IDX["right_fist"]
LABEL_MIRROR[CLASS_TO_IDX["right_fist"]] = CLASS_TO_IDX["left_fist"]


# --------------------------------------------------------------------------
# Data: one sequence per (subject, run), chunks kept in recording order
# --------------------------------------------------------------------------

def load_run(sid, subj_dir, run, label_map):
    """Epoch one run into (X, y) with chunks in TIME ORDER -- the order is
    the whole point here, since the context transformer attends backwards
    along it. Returns None if the run is missing or unusable."""
    edf_path = subj_dir / f"S{sid:03d}R{run:02d}.edf"
    if not edf_path.exists():
        return None

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted = {k: v for k, v in event_id.items() if k in label_map}
    if not wanted:
        return None

    epochs = mne.Epochs(raw, events, event_id=wanted,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=None, preload=True, verbose="ERROR")
    if len(epochs) == 0:
        return None

    missing = [c for c in CANONICAL_CHANNELS if c not in epochs.ch_names]
    if missing:
        return None
    order = [epochs.ch_names.index(c) for c in CANONICAL_CHANNELS]

    X = epochs.get_data(copy=True)[:, order, :N_SAMPLES].astype(np.float32)
    codes = epochs.events[:, 2]
    code_to_desc = {v: k for k, v in wanted.items()}
    y = np.array([CLASS_TO_IDX[label_map[code_to_desc[c]]] for c in codes],
                 dtype=np.int64)
    return X, y


def build_sequences(cache_path, max_subjects=None, use_cache=True):
    """Build one variable-length sequence per (subject, run).

    Returns X (n_seq, T_max, 64, 640) float16, y (n_seq, T_max) int64 with
    -1 in padded positions, lengths, sids, runs. Stored as float16 to keep
    the full 100-subject tensor near ~1.5 GB instead of ~3 GB; it is cast
    back to float32 per batch. After per-subject z-scoring values are O(1),
    so float16 has plenty of headroom.
    """
    if use_cache and cache_path.exists():
        print(f"Loading cached sequences from {cache_path}")
        d = np.load(cache_path)
        return d["X"], d["y"], d["lengths"], d["sids"], d["runs"]

    subs = subject_dirs(DATA_ROOT, max_subjects)
    print(f"Processing {len(subs)} subjects from {DATA_ROOT} (no cleaning pipeline)")

    seqs, labels, sids, runs = [], [], [], []
    t0 = time.time()
    for i, (sid, subj_dir) in enumerate(subs, 1):
        subj_seqs, subj_labels, subj_runs = [], [], []
        for run, label_map in RUN_LABELS.items():
            out = load_run(sid, subj_dir, run, label_map)
            if out is None:
                continue
            subj_seqs.append(out[0])
            subj_labels.append(out[1])
            subj_runs.append(run)
        if not subj_seqs:
            print(f"  [{i}/{len(subs)}] S{sid:03d}: no usable runs, skipped")
            continue

        # Per-subject, per-channel z-score across all of that subject's runs
        # (same rationale as the baselines: removes subject-specific scale,
        # uses only that subject's own signal, so no train/test leakage).
        stacked = np.concatenate(subj_seqs, axis=0)
        mean = stacked.mean(axis=(0, 2), keepdims=True)
        std = stacked.std(axis=(0, 2), keepdims=True) + 1e-8
        for s, lab, run in zip(subj_seqs, subj_labels, subj_runs):
            seqs.append(((s - mean) / std).astype(np.float16))
            labels.append(lab)
            sids.append(sid)
            runs.append(run)
        print(f"  [{i}/{len(subs)}] S{sid:03d}: {len(subj_seqs)} runs, "
              f"{sum(len(l) for l in subj_labels)} chunks ({time.time()-t0:.0f}s)")

    T_max = max(len(l) for l in labels)
    n_seq = len(labels)
    X = np.zeros((n_seq, T_max, len(CANONICAL_CHANNELS), N_SAMPLES), dtype=np.float16)
    y = np.full((n_seq, T_max), -1, dtype=np.int64)
    lengths = np.zeros(n_seq, dtype=np.int64)
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        X[i, :len(lab)] = s
        y[i, :len(lab)] = lab
        lengths[i] = len(lab)

    sids = np.array(sids, dtype=np.int64)
    runs = np.array(runs, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y=y, lengths=lengths, sids=sids, runs=runs)
    print(f"Cached {n_seq} run-sequences (T_max={T_max}) to {cache_path}")
    return X, y, lengths, sids, runs


# --------------------------------------------------------------------------
# Splits: whole runs, never individual chunks
# --------------------------------------------------------------------------

def run_holdout_split(sids, runs, seed=42):
    """Subject-dependent but leak-free for a context model: whole RUNS go to
    a split, never individual chunks.

    Splitting chunks (as the per-epoch baselines do) would be unsound here:
    chunk k's causal history contains chunks < k from the same run, so a
    test chunk's input would contain signal that was a training target.
    Holding out entire runs means a test chunk's whole history lives inside
    the test run.

    Per subject: of the 3 runs in each task group, 2 go to train and 1 is
    held out (4 train / 1 val / 1 test runs, ~67/17/17). Which group's
    held-out run goes to val vs test alternates across subjects, so val and
    test each end up containing all 5 classes.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    subjects = sorted(set(sids.tolist()))

    for k, sid in enumerate(subjects):
        heldout = []
        for group_runs in RUN_GROUPS.values():
            idx = [i for i in range(len(sids))
                   if sids[i] == sid and runs[i] in group_runs]
            if not idx:
                heldout.append(None)
                continue
            idx = list(rng.permutation(idx))
            heldout.append(idx[0])
            train.extend(idx[1:])
        # alternate so both val and test see lr-runs and ff-runs
        first_to_val = (k % 2 == 0)
        for gi, h in enumerate(heldout):
            if h is None:
                continue
            to_val = (gi == 0) == first_to_val
            (val if to_val else test).append(h)

    print(f"Run-holdout split: {len(train)} train / {len(val)} val / "
          f"{len(test)} test runs across {len(subjects)} subjects "
          f"(whole runs, so no chunk's history crosses a split)")
    return np.array(train), np.array(val), np.array(test)


def subject_holdout_split(sids, val_frac=0.15, test_frac=0.15, seed=42):
    """Strictest protocol: whole subjects held out, testing generalization
    to unseen users."""
    rng = np.random.default_rng(seed)
    subjects = sorted(set(sids.tolist()))
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test:n_test + n_val])

    train, val, test = [], [], []
    for i, sid in enumerate(sids.tolist()):
        (test if sid in test_s else val if sid in val_s else train).append(i)
    print(f"Subject-independent split: {n - n_test - n_val} train / {n_val} val / "
          f"{n_test} test subjects (disjoint)")
    return np.array(train), np.array(val), np.array(test)


# --------------------------------------------------------------------------
# SMOTE, adapted from per-epoch classification to run sequences
# --------------------------------------------------------------------------

def build_smote_neighbors(X, y, lengths, indices, sids, k=5):
    """Precompute, for every training chunk, its k nearest SAME-CLASS,
    SAME-SUBJECT neighbours (flattened 64x640 vectors, Euclidean) -- the
    same neighbour rule train_mlp.smote_augment uses.

    Two deliberate departures from the per-epoch version:

    * Neighbours are restricted to the same subject. The per-epoch scripts
      let SMOTE interpolate across subjects, which is already borderline;
      here every sequence is per-subject z-scored, so a cross-subject
      partner would be on a different scale.
    * Synthetic chunks are substituted IN PLACE rather than appended as new
      samples. Appending is what balances classes, but the unit of training
      here is an ordered run, and an appended chunk has no coherent place
      in a run's timeline -- its causal history would be fabricated.
      In-place substitution leaves run length, chunk order and the label
      sequence exactly as recorded.

    Restricting to same-subject also keeps this cheap: each subject has
    only ~23 chunks per task class, so the kNN is tiny despite the 40,960
    dimensions.
    """
    groups = {}
    for i in indices:
        for t in range(int(lengths[i])):
            groups.setdefault((int(sids[i]), int(y[i, t])), []).append((int(i), t))

    neighbors, skipped = {}, 0
    for (sid, cls), locs in groups.items():
        if len(locs) < 2:
            skipped += len(locs)
            continue
        flat = np.stack([X[i, t].astype(np.float32).ravel() for i, t in locs])
        n_neighbors = min(k + 1, len(locs))  # +1 to drop self at distance 0
        nn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(flat)
        _, nbr_idx = nn_model.kneighbors(flat)
        for row, (i, t) in enumerate(locs):
            cand = [locs[j] for j in nbr_idx[row][1:]]
            if cand:
                neighbors[(i, t)] = np.array(cand, dtype=np.int64)

    print(f"SMOTE: indexed {len(neighbors)} training chunks "
          f"({skipped} skipped -- no same-subject same-class partner)")
    return neighbors


class RunSequenceDataset(torch.utils.data.Dataset):
    """Holds the float16 tensor and casts per item, keeping RAM at ~1.5 GB.

    When `neighbors` is supplied, chunks are SMOTE-interpolated on the fly:
    each chunk is independently replaced, with probability `smote_prob`, by
    x_i + lam*(x_j - x_i) for a random one of its k same-class neighbours,
    lam ~ U(0, lam_max) -- the interpolation rule from the CNN-GRU paper.
    Only ever passed for the training split.
    """

    def __init__(self, X, y, lengths, indices, neighbors=None, smote_prob=0.0,
                 lam_max=1.0, mirror_prob=0.0, channel_drop=0.0, scale_jitter=0.0,
                 time_mask_frac=0.0, noise_std=0.0, seed=42):
        self.X, self.y, self.lengths, self.indices = X, y, lengths, indices
        self.neighbors = neighbors
        self.smote_prob = smote_prob
        self.lam_max = lam_max
        self.mirror_prob = mirror_prob
        self.channel_drop = channel_drop
        self.scale_jitter = scale_jitter
        self.time_mask_frac = time_mask_frac
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.augment = any([smote_prob > 0 and neighbors is not None, mirror_prob > 0,
                            channel_drop > 0, scale_jitter > 0, time_mask_frac > 0,
                            noise_std > 0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        x = self.X[j].astype(np.float32)
        yv = self.y[j]
        n_valid = int(self.lengths[j])

        if not self.augment:
            return torch.from_numpy(x), torch.from_numpy(yv), n_valid

        x, yv = x.copy(), yv.copy()
        rng = self.rng

        # --- SMOTE: per-chunk interpolation toward a same-class neighbour ---
        if self.neighbors is not None and self.smote_prob > 0:
            for t in range(n_valid):
                if rng.random() >= self.smote_prob:
                    continue
                cand = self.neighbors.get((j, t))
                if cand is None:
                    continue
                nj, nt = cand[rng.integers(len(cand))]
                lam = rng.uniform(0.0, self.lam_max)
                x[t] += lam * (self.X[nj, nt].astype(np.float32) - x[t])

        # --- Hemisphere mirror: reflect the scalp, swap left/right labels ---
        # Applied to the WHOLE run so the sequence the context transformer
        # reads stays internally consistent. This is the one augmentation
        # here that manufactures genuinely new lateralized examples rather
        # than perturbing existing ones.
        if self.mirror_prob > 0 and rng.random() < self.mirror_prob:
            x = x[:, MIRROR_INDEX, :]
            lab = yv >= 0
            yv[lab] = LABEL_MIRROR[yv[lab]]

        # --- Channel dropout: simulate dead electrodes for the whole run ---
        if self.channel_drop > 0:
            dead = rng.random(x.shape[1]) < self.channel_drop
            if dead.any():
                x[:, dead, :] = 0.0

        # --- Per-run amplitude jitter (gain/impedance drift between runs) ---
        if self.scale_jitter > 0:
            x *= rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)

        # --- Per-chunk contiguous time mask (SpecAugment-style) ---
        if self.time_mask_frac > 0:
            width = int(round(x.shape[2] * self.time_mask_frac))
            if width > 0:
                for t in range(n_valid):
                    s = int(rng.integers(0, x.shape[2] - width + 1))
                    x[t, :, s:s + width] = 0.0

        # --- Additive Gaussian noise (data is z-scored, so std is relative) ---
        if self.noise_std > 0:
            x[:n_valid] += rng.normal(
                0.0, self.noise_std, x[:n_valid].shape).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(yv), n_valid


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class ViewEncoder(nn.Module):
    """Compact EEGNet-style CNN for ONE view: temporal conv -> depthwise
    spatial conv collapsing that view's channels -> separable conv -> a
    d_model embedding for the chunk. Kept small on purpose; the unique-trial
    count here is modest and this is the module that gets instantiated 5x.

    Two departures from the parent script, both there to make this module
    convertible (see the module docstring): the nn.Sequential is unrolled
    into named submodules so SpikingViewEncoder can address each conv,
    BatchNorm and activation, and the two ELUs are ReLUs so their output
    maps onto a firing rate. Everything else -- layer order, kernel sizes,
    groups, bias=False, pooling, dropout -- is unchanged."""

    def __init__(self, n_channels, d_model, f1=8, depth=2, dropout=0.4):
        super().__init__()
        f2 = f1 * depth
        # ~0.4 s temporal kernel at 160 Hz -- spans mu/beta cycles
        self.temporal = nn.Conv2d(1, f1, (1, 65), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_depth = nn.Conv2d(f2, f2, (1, 15), padding="same", groups=f2,
                                   bias=False)
        self.sep_point = nn.Conv2d(f2, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.proj = nn.Linear(f2 * (N_SAMPLES // 32), d_model)

    def forward(self, x):                         # x: (N, C_view, 640)
        h = self.bn2(self.spatial(self.bn1(self.temporal(x.unsqueeze(1)))))
        h = self.drop1(self.pool1(self.act1(h)))
        h = self.bn3(self.sep_point(self.sep_depth(h)))
        h = self.drop2(self.pool2(self.act2(h)))  # (N, f2, 1, 20)
        return self.proj(h.flatten(1))


class MultiViewChunkEncoder(nn.Module):
    """Encodes one 4 s chunk into a single vector: per-view CNN -> 5 view
    tokens (+ learned view embeddings) -> 1 attention layer fusing views ->
    mean pool. The fusion layer is where laterality gets read out."""

    def __init__(self, d_model, tie_sm=True, n_heads=4, dropout=0.3, f1=8, depth=2):
        super().__init__()
        self.tie_sm = tie_sm
        encoders = {}
        for v in VIEW_NAMES:
            if tie_sm and v == TIED_VIEWS[1]:
                continue  # shares the left encoder
            encoders[v] = ViewEncoder(len(VIEWS[v]), d_model, f1=f1, depth=depth,
                                      dropout=dropout)
        self.encoders = nn.ModuleDict(encoders)
        self.view_emb = nn.Parameter(torch.zeros(len(VIEW_NAMES), d_model))
        nn.init.normal_(self.view_emb, std=0.02)
        self.fuse = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )

    def forward(self, x):  # x: (N, 64, 640)
        toks = []
        for vi, v in enumerate(VIEW_NAMES):
            enc = self.encoders[TIED_VIEWS[0]] if (self.tie_sm and v == TIED_VIEWS[1]) \
                else self.encoders[v]
            idx = torch.as_tensor(_view_idx[v], device=x.device)
            toks.append(enc(x.index_select(1, idx)) + self.view_emb[vi])
        return self.fuse(torch.stack(toks, dim=1)).mean(dim=1)  # (N, d_model)


class MultiViewRunModel(nn.Module):
    """Per-chunk multi-view encoding + causal context over the run, combined
    as LayerNorm(local + gate * context)."""

    def __init__(self, d_model=128, n_classes=len(CLASSES), tie_sm=True,
                 ctx_layers=2, ctx_heads=4, dropout=0.3, max_len=64,
                 context_weight=0.1, freeze_context=False, context_len=0,
                 f1=8, depth=2):
        super().__init__()
        self.chunk_encoder = MultiViewChunkEncoder(d_model, tie_sm=tie_sm,
                                                   dropout=dropout, f1=f1, depth=depth)
        # positional info lives only on the context path, so the local path
        # (and therefore the context_weight=0 ablation) stays position-free
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, ctx_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, ctx_layers,
                                             enable_nested_tensor=False)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        gate = torch.tensor(float(context_weight))
        self.context_gate = nn.Parameter(gate, requires_grad=not freeze_context)
        self.context_len = context_len
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def attn_mask(self, T, device):
        """True = disallowed. Causal, optionally limited to a sliding window
        of `context_len` chunks (1 == self only, 0 == unlimited history)."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        mask = j > i
        if self.context_len > 0:
            mask = mask | ((i - j) >= self.context_len)
        return mask

    def encode_chunks(self, x):  # (B, T, 64, 640) -> (B, T, d)
        B, T = x.shape[:2]
        h = self.chunk_encoder(x.reshape(B * T, x.shape[2], x.shape[3]))
        return h.reshape(B, T, -1)

    def apply_context(self, local, pad_mask, mask_positions=None):
        B, T, _ = local.shape
        h = local
        if mask_positions is not None:
            h = torch.where(mask_positions.unsqueeze(-1),
                            self.mask_token.expand(B, T, -1), h)
        h = h + self.pos(torch.arange(T, device=local.device)).unsqueeze(0)
        return self.context(h, mask=self.attn_mask(T, local.device),
                            src_key_padding_mask=pad_mask)

    def forward(self, x, pad_mask):
        local = self.encode_chunks(x)
        ctx = self.apply_context(local, pad_mask)
        return self.head(self.norm(local + self.context_gate * ctx))


class ReconHeads(nn.Module):
    """Per-view decoders used only during pre-training: project the causal
    representation back up to that view's raw (C_view, 640) signal."""

    def __init__(self, d_model):
        super().__init__()
        self.heads = nn.ModuleDict()
        for v in VIEW_NAMES:
            c = len(VIEWS[v])
            self.heads[v] = nn.ModuleDict({
                "lin": nn.Linear(d_model, c * 4 * 40),
                "up": nn.Sequential(
                    nn.ConvTranspose1d(c * 4, c * 2, 4, stride=4), nn.ELU(),
                    nn.ConvTranspose1d(c * 2, c, 4, stride=4),
                ),
            })

    def forward(self, h, view):  # h: (N, d) -> (N, C_view, 640)
        c = len(VIEWS[view])
        z = self.heads[view]["lin"](h).reshape(-1, c * 4, 40)
        return self.heads[view]["up"](z)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def make_pad_mask(lengths, T, device):
    """True where padded (what nn.Transformer's src_key_padding_mask wants)."""
    ar = torch.arange(T, device=device).unsqueeze(0)
    return ar >= lengths.unsqueeze(1).to(device)


def save_checkpoint(path, **payload):
    """torch.save wrapper shared by every --save handler here: creates the
    parent dir and confirms what was written, so a bare state_dict is never
    saved alone -- callers also bundle the constructor config, the calibrated
    thresholds/token scales, and (for the spiking model) its own config,
    since the frozen front end isn't an nn.Module and would otherwise be
    invisible to state_dict() entirely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}  (keys: {', '.join(sorted(payload))})")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb, lb in loader:
        xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
        pad_mask = make_pad_mask(lb, xb.shape[1], device)
        with torch.set_grad_enabled(train):
            logits = model(xb, pad_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        valid = yb.reshape(-1) != -1
        n_valid = int(valid.sum())
        total_loss += loss.item() * n_valid
        correct += int((logits.reshape(-1, logits.shape[-1]).argmax(1)[valid]
                        == yb.reshape(-1)[valid]).sum())
        n += n_valid
    return total_loss / max(n, 1), correct / max(n, 1)


def pretrain(model, loader, device, epochs, mask_frac, lr, d_model):
    """Causal masked-chunk pre-training: replace a fraction of chunk
    embeddings with the mask token and reconstruct their raw per-view signal
    from PRECEDING chunks only. Encoder still gets gradient through the
    unmasked chunks that form the context."""
    recon = ReconHeads(d_model).to(device)
    params = list(model.chunk_encoder.parameters()) + list(model.context.parameters()) \
        + [model.mask_token] + list(model.pos.parameters()) + list(recon.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb, lb in loader:
            xb, lb = xb.to(device), lb.to(device)
            B, T = xb.shape[:2]
            pad_mask = make_pad_mask(lb, T, device)
            valid = ~pad_mask
            # never mask position 0: it has no history to predict from
            maskable = valid.clone()
            maskable[:, 0] = False
            sel = (torch.rand(B, T, device=device) < mask_frac) & maskable
            if not sel.any():
                continue

            local = model.encode_chunks(xb)
            h = model.apply_context(local, pad_mask, mask_positions=sel)

            loss = 0.0
            hm = h[sel]                       # (M, d)
            for v in VIEW_NAMES:
                idx = torch.as_tensor(_view_idx[v], device=device)
                target = xb.index_select(2, idx)[sel]   # (M, C_view, 640)
                loss = loss + F.mse_loss(recon(hm, v), target)
            loss = loss / len(VIEW_NAMES)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * int(sel.sum())
            n += int(sel.sum())
        print(f"  pretrain epoch {ep:3d}  recon_mse={total / max(n, 1):.4f}")
    return model



# --------------------------------------------------------------------------
# ANN -> SNN conversion of the per-view CNNs
# --------------------------------------------------------------------------

ACT_LAYERS = ("act1", "act2")


def encoder_key(view, tie_sm):
    """Which entry of MultiViewChunkEncoder.encoders actually runs `view`
    (the two sensorimotor views share one encoder when tied)."""
    return TIED_VIEWS[0] if (tie_sm and view == TIED_VIEWS[1]) else view


def fold_conv_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single equivalent Conv2d. Every conv
    in ViewEncoder is bias=False, so the fused bias comes entirely from
    BatchNorm's learned shift. The BN must be in eval mode -- running
    statistics are what gets folded."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=conv.padding,
                      dilation=conv.dilation, groups=conv.groups,
                      bias=True).to(conv.weight.device)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv_bias = conv.bias if conv.bias is not None \
        else torch.zeros(conv.out_channels, device=conv.weight.device)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_((conv_bias - bn.running_mean) * scale + bn.bias)
    fused.eval()
    for param in fused.parameters():
        param.requires_grad_(False)
    return fused


def robust_max(act, percentile, max_elems=1 << 22):
    """A high percentile rather than the raw max, so one outlier activation
    can't inflate a threshold and starve that layer of spikes (Rueckauer et
    al. 2017). Sampled down first: torch.quantile refuses very large inputs,
    and one calibration batch of act1 is tens of millions of values."""
    act = act.detach().flatten().float().cpu()
    if act.numel() > max_elems:
        act = act[torch.randint(act.numel(), (max_elems,))]
    return torch.quantile(act, percentile / 100.0).item()


@torch.no_grad()
def calibrate_thresholds(model, loader, device, n_batches, percentile):
    """One threshold per (view encoder, activation), taken over unaugmented
    training chunks.

    Padded positions are dropped before encoding: they are all-zero, and
    letting a third of the calibration set be zeros would drag every
    percentile down. The tied sensorimotor encoder is hooked once and
    called twice per chunk, so it is calibrated on both views' activations,
    which is right -- it is the same weights seeing both.
    """
    model.eval()
    chunk_enc = model.chunk_encoder
    thresholds = {k: {a: 0.0 for a in ACT_LAYERS} for k in chunk_enc.encoders}

    def make_hook(key, act):
        def hook(module, inp, out):
            thresholds[key][act] = max(thresholds[key][act],
                                       robust_max(out, percentile))
        return hook

    handles = [getattr(enc, act).register_forward_hook(make_hook(key, act))
               for key, enc in chunk_enc.encoders.items() for act in ACT_LAYERS]
    try:
        for i, (xb, _, lb) in enumerate(loader):
            if i >= n_batches:
                break
            xb = xb.to(device)
            valid = ~make_pad_mask(lb, xb.shape[1], device)
            chunk_enc(xb[valid])
    finally:
        for h in handles:
            h.remove()

    # A layer that never fired on the calibration set would otherwise give a
    # zero threshold and divide-by-zero spiking; floor it and say so.
    for key, th in thresholds.items():
        for act, v in th.items():
            if v <= 0.0:
                print(f"  WARNING: {key}.{act} never activated during "
                      f"calibration; flooring its threshold")
                th[act] = 1e-4
    return thresholds


class SpikingViewEncoder:
    """One trained ViewEncoder run as a spiking convolutional network.

    Weights are copied unchanged (BatchNorm folded into its preceding conv)
    and each ReLU becomes a subtractive-reset integrate-and-fire neuron with
    the calibrated threshold. Spikes are transmitted scaled by that
    threshold: this is threshold balancing with the per-layer gain folded
    into the downstream weights, where it would live on hardware anyway. The
    scaling is what makes a firing rate r = a/theta arrive at the next conv
    as the activation `a` the ANN would have sent, so biases need no
    rescaling and the time-averaged output converges on the ANN's own
    embedding rather than a shrunken copy of it.

    temporal->bn1->spatial->bn2 is a purely linear run -- no activation sits
    between them in the ANN -- so it composes into one operator, and since
    the analog input is constant across timesteps it is evaluated once
    outside the loop rather than redundantly inside it. Spikes appear only
    where the ANN itself had a nonlinearity. `proj` stays a non-spiking
    accumulator: it is the handoff to the ANN transformer, which wants a
    real-valued token, not a spike train.
    """

    def __init__(self, enc, thresholds):
        self.pre = nn.Sequential(fold_conv_bn(enc.temporal, enc.bn1),
                                 fold_conv_bn(enc.spatial, enc.bn2))
        self.mid = nn.Sequential(enc.sep_depth,
                                 fold_conv_bn(enc.sep_point, enc.bn3))
        self.pool1, self.pool2, self.proj = enc.pool1, enc.pool2, enc.proj
        self.thr1, self.thr2 = thresholds["act1"], thresholds["act2"]

    @staticmethod
    def _if_step(cur, mem, threshold):
        mem = mem + cur
        # Zero floor: these neurons have no leak, so a unit whose ANN
        # activation is 0 (net negative input current) would integrate an
        # unboundedly negative membrane and stay unresponsive for the rest
        # of the simulation. Clamping at 0 keeps "silent" and "deeply
        # inhibited" the same state, which is what ReLU does.
        mem = mem.clamp(min=0.0)
        spk = (mem >= threshold).float()
        return spk, mem - spk * threshold

    @torch.no_grad()
    def start(self, x):
        """Membrane state for one view's chunks. The temporal->spatial run is
        linear and the analog input is constant, so its output current is
        computed once here rather than at every timestep."""
        cur1 = self.pre(x.unsqueeze(1))
        return {"cur1": cur1, "mem1": torch.zeros_like(cur1), "mem2": None}

    @torch.no_grad()
    def step(self, state):
        """Advance one timestep. Returns the projection's input current --
        the raw ANN-scale token, before any rate decoding or token
        normalisation -- and both layers' spike tensors.

        Stepwise rather than a self-contained loop because the spiking
        transformer above has to be advanced in lockstep with this."""
        spk1, state["mem1"] = self._if_step(state["cur1"], state["mem1"], self.thr1)
        cur2 = self.mid(self.pool1(spk1 * self.thr1))
        if state["mem2"] is None:
            state["mem2"] = torch.zeros_like(cur2)
        spk2, state["mem2"] = self._if_step(cur2, state["mem2"], self.thr2)
        return self.proj(self.pool2(spk2 * self.thr2).flatten(1)), spk1, spk2

    @torch.no_grad()
    def run(self, x, timesteps, checkpoints):
        """x: (N, C_view, 640) -> ({t: (N, d_model)}, mean spike rates)."""
        state, acc = self.start(x), None
        spike_sums = {a: 0.0 for a in ACT_LAYERS}
        emb_by_t = {}

        for t in range(1, timesteps + 1):
            emb, spk1, spk2 = self.step(state)
            acc = emb if acc is None else acc + emb
            spike_sums["act1"] += spk1.mean().item()
            spike_sums["act2"] += spk2.mean().item()
            if t in checkpoints:
                emb_by_t[t] = acc / t

        return emb_by_t, {a: s / timesteps for a, s in spike_sums.items()}


class SpikingChunkEncoder:
    """MultiViewChunkEncoder with every per-view CNN swapped for its spiking
    counterpart. The learned view embeddings and the fusion attention layer
    are the trained model's own ANN modules, called unchanged -- they just
    receive rate-decoded chunk tokens instead of ANN ones."""

    def __init__(self, chunk_encoder, thresholds):
        self.tie_sm = chunk_encoder.tie_sm
        self.encoders = {key: SpikingViewEncoder(enc, thresholds[key])
                         for key, enc in chunk_encoder.encoders.items()}
        self.view_emb = chunk_encoder.view_emb
        self.fuse = chunk_encoder.fuse

    @torch.no_grad()
    def forward(self, x, timesteps, checkpoints):
        """x: (N, 64, 640) -> ({t: (N, d_model)}, per-view spike rates).

        Rates are keyed by view rather than by encoder, so a tied
        sensorimotor encoder reports separately for the left and right
        views it was run on."""
        toks = {t: [] for t in checkpoints}
        rates = {}
        for vi, v in enumerate(VIEW_NAMES):
            idx = torch.as_tensor(_view_idx[v], device=x.device)
            emb_by_t, view_rates = self.encoders[encoder_key(v, self.tie_sm)].run(
                x.index_select(1, idx), timesteps, checkpoints)
            for t, emb in emb_by_t.items():
                toks[t].append(emb + self.view_emb[vi])
            for act, r in view_rates.items():
                rates[f"{v}.{act}"] = r
        return ({t: self.fuse(torch.stack(v, dim=1)).mean(dim=1)
                 for t, v in toks.items()}, rates)


class HybridSNNModel:
    """MultiViewRunModel with a spiking front end: spiking per-view CNNs ->
    ANN view fusion -> ANN causal context transformer -> ANN head. Only the
    convolutions changed; everything from the view tokens upward is the
    trained model's own modules, called as they are. Inference only.

    `chunk_batch` caps how many chunks are simulated at once: the outer
    dataloader batches whole runs, so one batch can be several hundred
    chunks x 5 views x T timesteps of live activation.
    """

    def __init__(self, model, thresholds, chunk_batch=64):
        self.model = model
        self.chunk_encoder = SpikingChunkEncoder(model.chunk_encoder, thresholds)
        self.chunk_batch = chunk_batch

    @torch.no_grad()
    def forward(self, x, pad_mask, timesteps, checkpoints):
        """x: (B, T, 64, 640) -> ({t: logits}, mean spike rates)."""
        B, T = x.shape[:2]
        d = self.model.pos.embedding_dim
        keep = (~pad_mask).reshape(-1)
        # Padded positions are dropped rather than simulated: they are
        # zero-padding, the context attention masks them out as keys and
        # they carry no label, so simulating them buys nothing but runtime.
        # Their tokens stay at zero.
        chunks = x.reshape(B * T, x.shape[2], x.shape[3])[keep]

        parts = {t: [] for t in checkpoints}
        rate_sums, n_parts = {}, 0
        for i in range(0, len(chunks), self.chunk_batch):
            emb_by_t, rates = self.chunk_encoder.forward(
                chunks[i:i + self.chunk_batch], timesteps, checkpoints)
            for t, emb in emb_by_t.items():
                parts[t].append(emb)
            for k, r in rates.items():
                rate_sums[k] = rate_sums.get(k, 0.0) + r
            n_parts += 1

        logits = {}
        for t in checkpoints:
            local = torch.zeros(B * T, d, device=x.device)
            local[keep] = torch.cat(parts[t])
            local = local.reshape(B, T, d)
            ctx = self.model.apply_context(local, pad_mask)
            logits[t] = self.model.head(
                self.model.norm(local + self.model.context_gate * ctx))
        return logits, {k: r / max(n_parts, 1) for k, r in rate_sums.items()}


@torch.no_grad()
def evaluate_hybrid(hybrid, loader, device, timesteps, checkpoints,
                    limit_batches=0):
    """Run the hybrid at every checkpoint T, and the unconverted ANN on
    exactly the same chunks. Evaluating both here (rather than reusing the
    full-test ANN number) keeps the comparison honest when
    --snn-limit-batches shortens the SNN pass."""
    hybrid.model.eval()
    preds = {t: [] for t in checkpoints}
    ann_preds, trues = [], []
    rate_sums, n_batches = {}, 0

    for i, (xb, yb, lb) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        xb, lb = xb.to(device), lb.to(device)
        pad_mask = make_pad_mask(lb, xb.shape[1], device)
        t0 = time.time()
        logits_by_t, rates = hybrid.forward(xb, pad_mask, timesteps, checkpoints)
        valid = yb.reshape(-1) != -1

        for t, logits in logits_by_t.items():
            preds[t].append(
                logits.reshape(-1, logits.shape[-1]).argmax(1).cpu()[valid].numpy())
        ann_logits = hybrid.model(xb, pad_mask)
        ann_preds.append(
            ann_logits.reshape(-1, ann_logits.shape[-1]).argmax(1).cpu()[valid].numpy())
        trues.append(yb.reshape(-1)[valid].numpy())

        for k, r in rates.items():
            rate_sums[k] = rate_sums.get(k, 0.0) + r
        n_batches += 1
        print(f"  [SNN] batch {i + 1}: {int(valid.sum())} chunks x 5 views "
              f"simulated for T={timesteps} ({time.time() - t0:.1f}s)", flush=True)

    trues = np.concatenate(trues)
    acc_by_t = {t: float((np.concatenate(preds[t]) == trues).mean())
                for t in checkpoints}
    spike_rates = {k: r / max(n_batches, 1) for k, r in rate_sums.items()}
    return (acc_by_t, trues, np.concatenate(preds[max(checkpoints)]),
            np.concatenate(ann_preds), spike_rates)


def hybrid_report(model, test_loader, device, args, thresholds, checkpoints):
    """The intermediate baseline: converted CNNs still feeding ANN attention.
    Reported at the same T the spiking model runs at, so the three numbers in
    the summary differ only in what replaced what."""
    if args.snn_limit_batches:
        print(f"NOTE: SNN evaluated on the first {args.snn_limit_batches} test "
              f"batches only (--snn-limit-batches)")
    print(f"\nSimulating the hybrid (spiking view CNNs -> ANN attention) "
          f"up to T={args.timesteps}:")
    hybrid = HybridSNNModel(model, thresholds, chunk_batch=args.snn_chunk_batch)
    acc_by_t, snn_trues, snn_preds, ann_preds, spike_rates = evaluate_hybrid(
        hybrid, test_loader, device, args.timesteps, checkpoints,
        limit_batches=args.snn_limit_batches)

    t_max = max(checkpoints)
    print(f"\n[SNN, T={t_max}] Classification report (test set):")
    print(classification_report(snn_trues, snn_preds,
                                labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=3, zero_division=0))
    snn_task = snn_trues != CLASS_TO_IDX["baseline"]
    if snn_task.any():
        print(f"[SNN, T={t_max}] Task-only accuracy (baseline chunks excluded): "
              f"{(snn_preds[snn_task] == snn_trues[snn_task]).mean():.4f}")
    print(f"[SNN, T={t_max}] Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(snn_trues, snn_preds,
                           labels=list(range(len(CLASSES)))))

    print("\n=== ANN vs hybrid SNN (same test chunks) ===")
    print(f"ANN view CNNs:            test_acc={(ann_preds == snn_trues).mean():.4f}")
    for t in checkpoints:
        print(f"Spiking view CNNs (T={t:>4}): test_acc={acc_by_t[t]:.4f}")
    agree = (snn_preds == ann_preds).mean()
    print(f"Prediction agreement with the ANN at T={t_max}: {agree:.4f}  "
          f"(1.0 would be a lossless conversion)")
    print(f"\n[SNN] Mean spike rate per view CNN layer at T={args.timesteps} "
          f"(fraction of neuron-timesteps that fired; lower is sparser and "
          f"cheaper on spiking hardware):")
    for key in sorted(spike_rates):
        print(f"  {key:>26}: {spike_rates[key]:.4f}")


# --------------------------------------------------------------------------
# Spikformer-style spiking attention on top of the frozen spiking CNNs
# --------------------------------------------------------------------------

@torch.no_grad()
def calibrate_token_scales(model, loader, device, n_batches, percentile):
    """A per-encoder gain for the frozen projection's output.

    ViewEncoder.proj has no activation after it, so its output is signed and
    O(1)-ish -- feed it straight to a threshold-1 neuron and the firing rate
    depends entirely on how large that particular encoder happens to have
    grown. Dividing by a high percentile of its own output puts every view
    in the same, usable firing regime at the start of phase 2. It is a
    constant gain, so like the spike scaling it folds into the downstream
    Q/K/V weights on hardware, and phase 2 is free to absorb it anyway.

    Calibrated, not trained: no gradient ever reaches the CNN.
    """
    model.eval()
    chunk_enc = model.chunk_encoder
    scales = {k: 0.0 for k in chunk_enc.encoders}

    def make_hook(key):
        def hook(module, inp, out):
            scales[key] = max(scales[key], robust_max(out, percentile))
        return hook

    handles = [enc.proj.register_forward_hook(make_hook(key))
               for key, enc in chunk_enc.encoders.items()]
    try:
        for i, (xb, _, lb) in enumerate(loader):
            if i >= n_batches:
                break
            xb = xb.to(device)
            chunk_enc(xb[~make_pad_mask(lb, xb.shape[1], device)])
    finally:
        for h in handles:
            h.remove()
    return {k: max(v, 1e-4) for k, v in scales.items()}


class SpikingViewFrontEnd:
    """The five converted view CNNs, advanced one timestep at a time and
    stopping at the projection -- the frozen base the spiking transformer
    sits on.

    Deliberately not an nn.Module: kept as a plain object, its convs and
    projections never appear in the spiking model's .parameters(), so the
    phase-2 optimizer cannot touch the CNN even by accident. It runs under
    no_grad throughout, which is also why the hand-written IF neurons here
    need no surrogate gradient -- only the layers above them do.
    """

    def __init__(self, chunk_encoder, thresholds, token_scales):
        self.tie_sm = chunk_encoder.tie_sm
        self.encoders = {key: SpikingViewEncoder(enc, thresholds[key])
                         for key, enc in chunk_encoder.encoders.items()}
        self.keys = [encoder_key(v, self.tie_sm) for v in VIEW_NAMES]
        self.scales = [token_scales[k] for k in self.keys]

    @torch.no_grad()
    def start(self, x):
        """x: (N, 64, 640) -> one IF state per view."""
        return [self.encoders[k].start(
                    x.index_select(1, torch.as_tensor(_view_idx[v], device=x.device)))
                for v, k in zip(VIEW_NAMES, self.keys)]

    @torch.no_grad()
    def step(self, states, record=False):
        """-> (N, n_views, d_model) token currents, and the per-view spike
        rates only when asked: this runs at every timestep of every training
        batch, and ten spare reductions there are not free."""
        toks, rates = [], {}
        for v, k, scale, state in zip(VIEW_NAMES, self.keys, self.scales, states):
            cur, spk1, spk2 = self.encoders[k].step(state)
            toks.append(cur / scale)
            if record:
                rates[f"{v}.act1"] = spk1.mean()
                rates[f"{v}.act2"] = spk2.mean()
        return torch.stack(toks, dim=1), rates


def lif(beta, spike_grad, threshold=1.0):
    """Leaky integrate-and-fire with a surrogate gradient. init_hidden keeps
    the membrane inside the module: the stack below is deep enough that
    threading every membrane through the call chain, as cnn-gru/train_snn.py
    does, would drown the code. snn_utils.reset() clears them per batch."""
    return snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                     init_hidden=True)


class SpikingSelfAttention(nn.Module):
    """Spikformer's SSA (Zhou et al., ICLR 2023).

    Q, K and V are spike matrices -- Linear -> BatchNorm -> LIF -- so Q K^T
    and (Q K^T) V are products of {0,1} matrices: accumulate-only, no
    multiplications, which is the point of the design. There is no softmax
    (a spike matrix has no negatives to normalise away, and exp() is exactly
    what you cannot afford on spiking hardware); a scalar scale stands in for
    it, and the result passes through its own LIF before the output
    projection.

    Masking is simpler here than in the ANN: with no softmax, a disallowed
    position is zeroed rather than set to -inf.
    """

    def __init__(self, dim, heads, beta, spike_grad, scale=0.125,
                 attn_threshold=0.5):
        super().__init__()
        assert dim % heads == 0, "d_model must divide evenly among SSA heads"
        self.heads, self.head_dim = heads, dim // heads
        # Learnable, unlike the paper's constant: the value that puts the
        # attention neuron in range depends on how many tokens the stage
        # attends over, and the two stages here differ by 6x. See
        # calibrate_ssa_scales for where its starting value comes from.
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.calibrating, self.calib_sum, self.calib_nonzero = False, 0.0, 0.0
        self.q_lin, self.k_lin, self.v_lin = (
            nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.q_bn, self.k_bn, self.v_bn = (nn.BatchNorm1d(dim) for _ in range(3))
        self.q_lif, self.k_lif, self.v_lif = (
            lif(beta, spike_grad) for _ in range(3))
        # Paper's threshold for the attention neuron: Q K^T V sums over both
        # the head dimension and the tokens, so it arrives much larger than a
        # single synaptic current.
        self.attn_lif = lif(beta, spike_grad, threshold=attn_threshold)
        self.out_lin = nn.Linear(dim, dim, bias=False)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_lif = lif(beta, spike_grad)

    @staticmethod
    def _spike_proj(x, linear, bn, neuron):
        # BatchNorm1d normalises per feature over (batch x tokens), so the
        # token axis has to be last.
        return neuron(bn(linear(x).transpose(1, 2)).transpose(1, 2))

    def forward(self, x, mask=None):       # x: (N, L, D) spikes
        N, L, D = x.shape
        heads = lambda t: t.reshape(N, L, self.heads, self.head_dim).transpose(1, 2)
        q = heads(self._spike_proj(x, self.q_lin, self.q_bn, self.q_lif))
        k = heads(self._spike_proj(x, self.k_lin, self.k_bn, self.k_lif))
        v = heads(self._spike_proj(x, self.v_lin, self.v_bn, self.v_lif))

        attn = q @ k.transpose(-2, -1)     # (N, heads, L, L) spike coincidences
        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)
        raw = attn @ v
        if self.calibrating:
            self.calib_sum += float(raw.sum())
            self.calib_nonzero += float((raw != 0).sum())
        out = self.attn_lif((raw * self.scale).transpose(1, 2).reshape(N, L, D))
        return self._spike_proj(out, self.out_lin, self.out_bn, self.out_lif)


class SpikingMLP(nn.Module):
    """The block's feedforward half, spiking: Linear -> BN -> LIF, twice.
    Same 2x expansion the ANN TransformerEncoderLayer used."""

    def __init__(self, dim, hidden, beta, spike_grad):
        super().__init__()
        self.fc1, self.bn1, self.lif1 = (nn.Linear(dim, hidden),
                                         nn.BatchNorm1d(hidden),
                                         lif(beta, spike_grad))
        self.fc2, self.bn2, self.lif2 = (nn.Linear(hidden, dim),
                                         nn.BatchNorm1d(dim),
                                         lif(beta, spike_grad))

    def forward(self, x):
        h = self.lif1(self.bn1(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        return self.lif2(self.bn2(self.fc2(h).transpose(1, 2)).transpose(1, 2))


class SpikformerBlock(nn.Module):
    """SSA + MLP with residuals, replacing one nn.TransformerEncoderLayer.

    The residuals add spike trains, so activations are small non-negative
    integers rather than binary -- that is what the paper does, and the
    BatchNorm at the head of each sub-block is what keeps it in range."""

    def __init__(self, dim, heads, beta, spike_grad, scale, attn_threshold):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, heads, beta, spike_grad,
                                         scale=scale, attn_threshold=attn_threshold)
        self.mlp = SpikingMLP(dim, 2 * dim, beta, spike_grad)

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask)
        return x + self.mlp(x)


class SpikformerRunModel(nn.Module):
    """MultiViewRunModel with both attention stages replaced by Spikformer
    blocks, sitting on the frozen converted view CNNs.

    The structure of the ANN is kept line for line -- per-view tokens plus
    view embeddings, one fusion attention over the 5 views, a causal context
    attention over the run, and the LayerNorm(local + gate * context)
    combination -- so the only variable being changed is what kind of
    attention does the work.

    Three joins deserve comment:

    * The frozen projection's output is a current, not a spike train, so a
      LIF turns each view token into spikes before the fusion block. Its
      input is (calibrated projection current + view embedding), which lets
      the trainable view embedding act as the per-view bias that decides
      where that neuron sits in its firing range.
    * The local and context paths get separate input neurons off the same
      fused current, because the ANN kept position information on the
      context path only -- the local path, and therefore the
      context_weight=0 ablation, stays position-free.
    * The readout accumulates local + gate * context over the simulation and
      normalises once at the end. That is Spikformer's mean-over-time
      readout, and it keeps the trained ANN's head usable as a warm start.

    Everything trainable here is initialised from the ANN where the shapes
    line up (view embeddings, positions, context gate, norm, head); only the
    spiking attention blocks start fresh.
    """

    def __init__(self, ann_model, thresholds, token_scales, timesteps,
                 n_classes=len(CLASSES), heads=4, ctx_layers=2, dropout=0.3,
                 max_len=64, beta=0.9, scale=0.125, attn_threshold=0.5,
                 spike_grad=None):
        super().__init__()
        spike_grad = spike_grad or surrogate.fast_sigmoid()
        self.front = SpikingViewFrontEnd(ann_model.chunk_encoder, thresholds,
                                         token_scales)
        self.timesteps = timesteps
        self.context_len = ann_model.context_len
        d = ann_model.pos.embedding_dim

        self.view_emb = nn.Parameter(
            ann_model.chunk_encoder.view_emb.detach().clone())
        self.token_lif = lif(beta, spike_grad)
        self.fuse = SpikformerBlock(d, heads, beta, spike_grad, scale, attn_threshold)

        self.pos = nn.Embedding(max_len, d)
        self.pos.weight.data.copy_(ann_model.pos.weight.detach())
        self.local_lif = lif(beta, spike_grad)
        self.ctx_lif = lif(beta, spike_grad)
        self.context = nn.ModuleList([
            SpikformerBlock(d, heads, beta, spike_grad, scale, attn_threshold)
            for _ in range(ctx_layers)])

        self.context_gate = nn.Parameter(ann_model.context_gate.detach().clone())
        self.norm = nn.LayerNorm(d)
        self.norm.load_state_dict(ann_model.norm.state_dict())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))
        self.head.load_state_dict(ann_model.head.state_dict())

        self.record_spikes = False
        self.spike_rates = {}

    def attn_mask(self, T, device, pad_mask):
        """True = disallowed, as in MultiViewRunModel.attn_mask, broadcast
        over heads and combined with the padding mask on the key axis."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        causal = j > i
        if self.context_len > 0:
            causal = causal | ((i - j) >= self.context_len)
        return causal[None, None] | pad_mask[:, None, None, :]

    def forward(self, x, pad_mask):
        B, T = x.shape[:2]
        d = self.view_emb.shape[1]
        snn_utils.reset(self)

        # Padded positions are never simulated -- they are zero-padding with
        # no label, masked out of the context attention as keys. Their slots
        # in the grid stay at zero, and each block re-zeroes them so nothing
        # a BatchNorm sees drifts away from that.
        keep = (~pad_mask).reshape(-1)
        valid = (~pad_mask).unsqueeze(-1).float()
        chunks = x.reshape(B * T, x.shape[2], x.shape[3])[keep]
        states = self.front.start(chunks)
        mask = self.attn_mask(T, x.device, pad_mask)
        pos = self.pos(torch.arange(T, device=x.device)).unsqueeze(0)

        feat, rate_sums = 0.0, {}
        for _ in range(self.timesteps):
            tok_cur, rates = self.front.step(states, self.record_spikes)
            tok = self.token_lif(tok_cur + self.view_emb.unsqueeze(0))
            fused = self.fuse(tok).mean(dim=1)                # (N, d) current

            grid = torch.zeros(B * T, d, device=x.device, dtype=fused.dtype)
            grid[keep] = fused
            grid = grid.reshape(B, T, d)

            local = self.local_lif(grid)                      # position-free
            h = self.ctx_lif(grid + pos)                      # context path only
            for block in self.context:
                h = block(h, mask) * valid
            feat = feat + local + self.context_gate * h

            if self.record_spikes:
                for k, v in rates.items():
                    rate_sums[k] = rate_sums.get(k, 0.0) + v

        if self.record_spikes:
            self.spike_rates = {k: float(v / self.timesteps)
                                for k, v in rate_sums.items()}
        return self.head(self.norm(feat / self.timesteps))


def load_model(path, device="cpu"):
    """Reconstruct the trained spiking model saved via --save. Rebuilds the
    ANN first (its weights are what the frozen, folded, IF-converted view
    CNNs are deterministically derived from -- SpikingViewFrontEnd is not an
    nn.Module, so its weights are never in spiking_state at all), then
    restores the trained spiking attention weights on top. Returns
    (spiking_model, classes)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ann_model = MultiViewRunModel(**ckpt["ann_config"]).to(device)
    ann_model.load_state_dict(ckpt["ann_state"])
    spiking = SpikformerRunModel(
        ann_model, ckpt["thresholds"], ckpt["token_scales"],
        **ckpt["spiking_config"]).to(device)
    spiking.load_state_dict(ckpt["spiking_state"])
    spiking.eval()
    return spiking, ckpt["classes"]


@torch.no_grad()
def calibrate_ssa_scales(spiking, loader, device, target=1.0):
    """Put each attention neuron in range before phase 2 starts.

    Spikformer's fixed 0.125 is sized for ImageNet-sized token counts: Q K^T V
    sums over the head dimension and over L tokens, and at L=196 patches it
    lands comfortably on a 0.5 threshold. The fusion stage here attends over
    5 view tokens and the context stage over ~30 chunks, an order of
    magnitude apart from each other and two from the paper -- left at 0.125
    the attention neuron fires ~0.2% of the time and the block degenerates
    into its residual branch, which is the one failure mode that would make
    this whole comparison meaningless.

    So one forward pass measures Q K^T V per stage and sets the scale so a
    TYPICAL NONZERO entry charges the attention neuron to `target` times its
    threshold in one step. The nonzero mean is the statistic that survives
    here: Q K^T V is a matrix of spike coincidences, mostly exactly zero, so
    its high percentiles are zero and its overall mean is small enough that
    matching it would hand back a scale in the hundreds -- which would then
    saturate the neuron on any coincidence at all. Anchoring on the events
    that actually happen sizes every stage sensibly regardless of how many
    tokens it attends over. From here the scale is a learned parameter.
    """
    attns = [(n, m) for n, m in spiking.named_modules()
             if isinstance(m, SpikingSelfAttention)]
    for _, m in attns:
        m.calibrating, m.calib_sum, m.calib_nonzero = True, 0.0, 0.0

    was_training = spiking.training
    spiking.eval()
    xb, _, lb = next(iter(loader))
    xb, lb = xb.to(device), lb.to(device)
    spiking(xb, make_pad_mask(lb, xb.shape[1], device))
    spiking.train(was_training)

    scales = {}
    for name, m in attns:
        m.calibrating = False
        if m.calib_nonzero > 0.0:
            m.scale.data.fill_(target * float(m.attn_lif.threshold)
                               / (m.calib_sum / m.calib_nonzero))
        else:
            print(f"  WARNING: {name} saw no Q K^T coincidences at all during "
                  f"calibration -- too few timesteps for the stack to warm up? "
                  f"Leaving its scale at {float(m.scale):.4f}")
        scales[name] = float(m.scale)

    # Measure rather than assume: a second pass reports what the attention
    # neurons actually do at these scales, before a single epoch is spent.
    with SpikeMonitor(spiking) as monitor:
        spiking(xb, make_pad_mask(lb, xb.shape[1], device))
        rates = {n: r for n, r in monitor.rates().items() if n.endswith("attn_lif")}
    return scales, rates


class SpikeMonitor:
    """Records the mean output of every LIF in the model -- i.e. the fraction
    of neuron-timesteps that fired -- for the sparsity report."""

    def __init__(self, model):
        self.model = model
        self.sums, self.counts, self.handles = {}, {}, []

    def __enter__(self):
        def make_hook(name):
            def hook(module, inp, out):
                self.sums[name] = self.sums.get(name, 0.0) + out.mean().item()
                self.counts[name] = self.counts.get(name, 0) + 1
            return hook
        self.handles = [m.register_forward_hook(make_hook(n))
                        for n, m in self.model.named_modules()
                        if isinstance(m, snn.Leaky)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def rates(self):
        return {n: self.sums[n] / self.counts[n] for n in sorted(self.sums)}


def main(task_spec=None, split_spec=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--split-mode", choices=["run-holdout", "subject-independent"],
                    default="run-holdout",
                    help="'run-holdout' (default): whole runs held out per subject. "
                         "'subject-independent': whole subjects held out.")
    # model
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ctx-heads", type=int, default=4)
    ap.add_argument("--f1", type=int, default=8, help="ViewEncoder temporal filters")
    ap.add_argument("--depth", type=int, default=2, help="ViewEncoder depth multiplier")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-tie-sm", action="store_true",
                    help="Give the two sensorimotor views independent encoders")
    # context knobs
    ap.add_argument("--context-weight", type=float, default=0.1,
                    help="Initial value of the learned context gate (0 = context off)")
    ap.add_argument("--freeze-context-weight", action="store_true",
                    help="Keep the gate fixed at --context-weight instead of learning it")
    ap.add_argument("--context-len", type=int, default=0,
                    help="Max chunks of history attention may see (0 = whole run so far, "
                         "1 = self only)")
    # SMOTE augmentation (in-place chunk interpolation; see build_smote_neighbors)
    ap.add_argument("--no-smote", action="store_true",
                    help="Disable SMOTE chunk interpolation on the training split")
    ap.add_argument("--smote-prob", type=float, default=0.3,
                    help="Per-chunk probability of being SMOTE-interpolated")
    ap.add_argument("--smote-k", type=int, default=5,
                    help="Nearest neighbours considered (CNN-GRU paper default: 5)")
    ap.add_argument("--smote-lambda-max", type=float, default=1.0,
                    help="Upper bound on the interpolation coefficient; lower values "
                         "keep synthetic chunks anchored nearer the real one")
    # signal-level augmentation (training split only)
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable ALL signal augmentation (mirror/channel-drop/"
                         "scale/time-mask/noise); SMOTE is controlled separately")
    ap.add_argument("--mirror-prob", type=float, default=0.5,
                    help="Probability of reflecting a run across the midline, swapping "
                         "left_fist<->right_fist. The strongest augmentation here: it "
                         "manufactures new lateralized trials rather than perturbing old ones")
    ap.add_argument("--channel-drop", type=float, default=0.1,
                    help="Per-channel probability of being zeroed for a whole run "
                         "(simulates a dead electrode)")
    ap.add_argument("--scale-jitter", type=float, default=0.1,
                    help="Per-run amplitude scaling ~ U(1-j, 1+j)")
    ap.add_argument("--time-mask", type=float, default=0.1,
                    help="Fraction of each chunk's 640 samples zeroed as one "
                         "contiguous span")
    ap.add_argument("--noise-std", type=float, default=0.1,
                    help="Std of additive Gaussian noise (signal is z-scored, so this "
                         "is relative to signal SD)")
    # pretraining
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--pretrain-mask-frac", type=float, default=0.3)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    # optimization
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8, help="runs per batch")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Baseline is ~50%% of chunks; weights are on by default")
    ap.add_argument("--seed", type=int, default=42)
    # ANN -> SNN conversion of the per-view CNNs (post-training, no retraining)
    ap.add_argument("--no-snn", action="store_true",
                    help="Stop after the ANN evaluation, skipping the conversion")
    ap.add_argument("--timesteps", type=int, default=32,
                    help="SNN simulation length, shared by the converted CNNs and "
                         "the spiking transformer. Lower than a pure conversion "
                         "would want (rate coding converges as ~1/T) because "
                         "phase 2 backpropagates through every timestep; "
                         "Spikformer itself trains at T=4")
    ap.add_argument("--timestep-checkpoints", type=str, default="4,8,16,32",
                    help="Comma-separated T values to report the converted CNNs' "
                         "accuracy at (capped at --timesteps)")
    ap.add_argument("--calib-batches", type=int, default=10,
                    help="Unaugmented training batches used for threshold calibration")
    ap.add_argument("--calib-percentile", type=float, default=99.9,
                    help="Percentile of calibration activations used as each IF "
                         "neuron's threshold (robust max, Rueckauer et al. 2017)")
    ap.add_argument("--snn-chunk-batch", type=int, default=64,
                    help="Chunks simulated at once; lower it if the SNN pass runs "
                         "out of memory (a batch of runs is hundreds of chunks)")
    ap.add_argument("--snn-limit-batches", type=int, default=0,
                    help="Evaluate the SNN on only the first N test batches "
                         "(0 = all). The ANN reference is recomputed on the same "
                         "chunks, so the comparison stays valid")
    ap.add_argument("--no-hybrid-eval", action="store_true",
                    help="Skip the converted-CNN + ANN-attention baseline and go "
                         "straight to the spiking-attention model")
    # Phase 2: Spikformer attention trained with surrogate gradients
    ap.add_argument("--snn-epochs", type=int, default=None,
                    help="Epochs of surrogate-gradient training for the spiking "
                         "attention (default: same budget as --epochs)")
    ap.add_argument("--snn-lr", type=float, default=1e-3)
    ap.add_argument("--ssa-heads", type=int, default=4,
                    help="Heads in both spiking attention stages")
    ap.add_argument("--ssa-scale", type=float, default=0.125,
                    help="Spikformer's scalar in place of softmax normalisation; "
                         "only a fallback, calibrate_ssa_scales overwrites it")
    ap.add_argument("--ssa-target", type=float, default=1.0,
                    help="Calibrate each SSA scale so a typical nonzero Q K^T V "
                         "charges its attention neuron to this multiple of the "
                         "firing threshold in one timestep")
    ap.add_argument("--attn-threshold", type=float, default=0.5,
                    help="Firing threshold of the attention neuron (paper: 0.5)")
    ap.add_argument("--lif-beta", type=float, default=0.9,
                    help="Membrane decay of every trained LIF")
    ap.add_argument("--token-percentile", type=float, default=99.0,
                    help="Percentile of the frozen projection's output used as "
                         "each view's token gain (see calibrate_token_scales)")
    ap.add_argument("--save", nargs="?", const="", default=None, metavar="PATH",
                    help="Save the trained spiking model to PATH at the end (bare "
                         "--save picks checkpoints/<script>_<timestamp>.pt). Bundles "
                         "the ANN's weights/config, the calibrated thresholds and "
                         "token scales, and the spiking attention's weights/config "
                         "together -- see load_model().")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    suffix = "all" if args.max_subjects is None else str(args.max_subjects)
    cache_path = CACHE_DIR / f"mv_seq_raw_{suffix}.npz"
    X, y, lengths, sids, runs = build_sequences(cache_path, args.max_subjects,
                                                use_cache=not args.no_cache)
    # Task hook. A variant script passes a spec that narrows the label problem
    # (see transformer/leftright.py); None keeps the full 5-class task, so this
    # script's own behaviour is unchanged.
    classes = CLASSES
    if task_spec is not None:
        X, y, lengths, sids, runs = task_spec.filter(X, y, lengths, sids, runs)
        classes = task_spec.classes
        print(f"\nTask: {task_spec.name}  ->  classes={classes}")
    print(f"Sequences: X={X.shape} (n_runs, T_max, channels, times), "
          f"subjects={len(set(sids.tolist()))}")
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    print("Chunk counts:", counts)

    # Split hook, same idea as the task hook: a variant or the master runner
    # can hand over its own splitter (see transformer/splits.py). None keeps
    # this script's own --split-mode behaviour.
    if split_spec is not None:
        tr_idx, va_idx, te_idx = split_spec(sids, runs, args.seed)
    elif args.split_mode == "run-holdout":
        tr_idx, va_idx, te_idx = run_holdout_split(sids, runs, seed=args.seed)
    else:
        tr_idx, va_idx, te_idx = subject_holdout_split(sids, seed=args.seed)

    # Neighbours are built from training runs only -- val/test are never
    # augmented and never act as interpolation partners.
    neighbors = None
    if not args.no_smote and args.smote_prob > 0:
        t0 = time.time()
        neighbors = build_smote_neighbors(X, y, lengths, tr_idx, sids, k=args.smote_k)
        print(f"SMOTE: neighbour index built in {time.time()-t0:.0f}s "
              f"(prob={args.smote_prob}, lambda~U(0,{args.smote_lambda_max}))")

    aug = not args.no_augment
    if aug:
        print(f"Augmentation (train only): mirror={args.mirror_prob} "
              f"channel_drop={args.channel_drop} scale_jitter={args.scale_jitter} "
              f"time_mask={args.time_mask} noise_std={args.noise_std}")

    def loader(idx, shuffle, augment=False):
        ds = RunSequenceDataset(
            X, y, lengths, idx,
            neighbors=neighbors if augment else None,
            smote_prob=args.smote_prob if augment else 0.0,
            lam_max=args.smote_lambda_max,
            mirror_prob=args.mirror_prob if (augment and aug) else 0.0,
            channel_drop=args.channel_drop if (augment and aug) else 0.0,
            scale_jitter=args.scale_jitter if (augment and aug) else 0.0,
            time_mask_frac=args.time_mask if (augment and aug) else 0.0,
            noise_std=args.noise_std if (augment and aug) else 0.0,
            seed=args.seed)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                           shuffle=shuffle)

    train_loader, val_loader, test_loader = (loader(tr_idx, True, augment=True),
                                             loader(va_idx, False),
                                             loader(te_idx, False))
    print(f"Chunks: train={int(lengths[tr_idx].sum())} val={int(lengths[va_idx].sum())} "
          f"test={int(lengths[te_idx].sum())}")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    ann_config = dict(
        n_classes=len(classes),
        d_model=args.d_model, tie_sm=not args.no_tie_sm,
        ctx_layers=args.ctx_layers, ctx_heads=args.ctx_heads,
        dropout=args.dropout, max_len=X.shape[1],
        context_weight=args.context_weight,
        freeze_context=args.freeze_context_weight,
        context_len=args.context_len, f1=args.f1, depth=args.depth,
    )
    model = MultiViewRunModel(**ann_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(sensorimotor encoders {'tied' if not args.no_tie_sm else 'independent'})")

    save_path = None
    if args.save is not None:
        save_path = args.save or str(ROOT / "checkpoints" /
            f"{Path(__file__).stem}_{time.strftime('%Y%m%d-%H%M%S')}.pt")

    if args.pretrain_epochs > 0:
        print(f"\nCausal masked-chunk pre-training on {len(tr_idx)} training runs "
              f"(mask_frac={args.pretrain_mask_frac}):")
        pretrain(model, train_loader, device, args.pretrain_epochs,
                 args.pretrain_mask_frac, args.pretrain_lr, args.d_model)

    if args.no_class_weights:
        weight = None
    else:
        train_y = y[tr_idx]
        freq = np.array([max((train_y == i).sum(), 1) for i in range(len(classes))],
                        dtype=np.float64)
        weight = torch.tensor((freq.sum() / (len(classes) * freq)),
                              dtype=torch.float32, device=device)
        print("Class weights:", {c: round(float(w), 3) for c, w in zip(classes, weight)})

    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"gate={model.context_gate.item():.3f}")
        if va_loss < best_val_loss - 1e-4:
            best_val_loss, stale = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\n[ANN] Test loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")
    print(f"[ANN] Learned context gate: {model.context_gate.item():.4f}  "
          f"(initialised at {args.context_weight}; how much history the model wanted)")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, lb in test_loader:
            xb, lb = xb.to(device), lb.to(device)
            logits = model(xb, make_pad_mask(lb, xb.shape[1], device))
            valid = yb.reshape(-1) != -1
            preds.append(logits.reshape(-1, logits.shape[-1]).argmax(1).cpu()[valid].numpy())
            trues.append(yb.reshape(-1)[valid].numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    print("\n[ANN] Classification report (test set):")
    print(classification_report(trues, preds, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    task = (trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if task.any():
        print(f"[ANN] Task-only accuracy (baseline chunks excluded): "
              f"{(preds[task] == trues[task]).mean():.4f}")
    print("[ANN] Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(trues, preds, labels=list(range(len(classes)))))

    if args.no_snn:
        return

    # ---------------------------------------------------------------------
    # Convert the per-view CNNs to spiking networks and re-run the test set.
    # The trained weights are used exactly as they are -- the only thing
    # calibration measures is how large each ReLU's activations get, which
    # is what sets its IF neuron's threshold.
    # ---------------------------------------------------------------------
    n_enc = len(model.chunk_encoder.encoders)
    print(f"\nConverting {n_enc} view CNN{'s' if n_enc != 1 else ''} "
          f"({len(VIEW_NAMES)} views, sensorimotor "
          f"{'tied' if not args.no_tie_sm else 'independent'}) to spiking networks.")
    print(f"Calibrating IF thresholds on {args.calib_batches} unaugmented "
          f"training batches (percentile={args.calib_percentile})...")
    # Calibration wants clean, representative activations, so it gets its own
    # unaugmented loader over the training runs.
    thresholds = calibrate_thresholds(model, loader(tr_idx, True), device,
                                      args.calib_batches, args.calib_percentile)
    for key, th in thresholds.items():
        print(f"  {key:>19}: act1={th['act1']:.4f}  act2={th['act2']:.4f}")

    checkpoints = sorted({min(int(t), args.timesteps)
                          for t in args.timestep_checkpoints.split(",")}
                         | {args.timesteps})
    if not args.no_hybrid_eval:
        hybrid_report(model, test_loader, device, args, thresholds, checkpoints)

    # ---------------------------------------------------------------------
    # Phase 2: replace both attention stages with Spikformer blocks and
    # train them with surrogate gradients, the CNNs frozen.
    # ---------------------------------------------------------------------
    token_scales = calibrate_token_scales(model, loader(tr_idx, True), device,
                                          args.calib_batches, args.token_percentile)
    print("\nToken gains (percentile of each frozen projection's output):",
          {k: round(v, 3) for k, v in token_scales.items()})

    spiking_config = dict(
        timesteps=args.timesteps, n_classes=len(classes), heads=args.ssa_heads,
        ctx_layers=args.ctx_layers, dropout=args.dropout, max_len=X.shape[1],
        beta=args.lif_beta, scale=args.ssa_scale, attn_threshold=args.attn_threshold)
    spiking = SpikformerRunModel(
        model, thresholds, token_scales, **spiking_config).to(device)

    frozen = sum(p.numel() for p in model.chunk_encoder.encoders.parameters())
    trainable = sum(p.numel() for p in spiking.parameters() if p.requires_grad)
    print(f"Spiking model: {trainable:,} trainable attention parameters, "
          f"{frozen:,} frozen CNN parameters "
          f"(the front end is not an nn.Module, so it cannot reach the optimizer)")

    ssa_scales, ssa_rates = calibrate_ssa_scales(spiking, train_loader, device,
                                                 target=args.ssa_target)
    print("SSA scales (a typical Q K^T V coincidence charges the attention "
          f"neuron to {args.ssa_target:.2f}x threshold), and the firing rate "
          "each one produces at initialisation:")
    for name, s in ssa_scales.items():
        rate = ssa_rates.get(f"{name}.attn_lif", float("nan"))
        print(f"  {name:>22}: scale={s:<10.4f} attn_lif fires {rate:.4f}")

    # Snapshot the CNN so the freeze is something the run demonstrates rather
    # than something this script claims.
    cnn_snapshot = {n: q.detach().clone()
                    for n, q in model.chunk_encoder.encoders.named_parameters()}

    snn_epochs = args.epochs if args.snn_epochs is None else args.snn_epochs
    print(f"\nTraining Spikformer attention with surrogate gradients "
          f"(fast sigmoid, beta={args.lif_beta}, T={args.timesteps}) for "
          f"{snn_epochs} epochs:")
    snn_opt = torch.optim.AdamW(spiking.parameters(), lr=args.snn_lr,
                                weight_decay=args.weight_decay)
    best_val, best_snn_state, stale = float("inf"), None, 0
    for epoch in range(1, snn_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(spiking, train_loader, criterion, snn_opt,
                                    device, True)
        va_loss, va_acc = run_epoch(spiking, val_loader, criterion, snn_opt,
                                    device, False)
        print(f"snn epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"gate={spiking.context_gate.item():.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if va_loss < best_val - 1e-4:
            best_val, stale = va_loss, 0
            best_snn_state = {k: v.clone() for k, v in spiking.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break
    if best_snn_state is not None:
        spiking.load_state_dict(best_snn_state)

    drift = max((cnn_snapshot[n] - q).abs().max().item()
                for n, q in model.chunk_encoder.encoders.named_parameters())
    print(f"Frozen-CNN check: largest weight change across phase 2 = {drift:.2e}")

    snn_loss, snn_acc = run_epoch(spiking, test_loader, criterion, snn_opt,
                                  device, False)
    print(f"\n[Spiking] Test loss={snn_loss:.4f}  Test accuracy={snn_acc:.4f}")
    print(f"[Spiking] Learned context gate: {spiking.context_gate.item():.4f}")

    spiking.eval()
    spiking.record_spikes = True
    sp_preds, sp_trues = [], []
    with SpikeMonitor(spiking) as monitor, torch.no_grad():
        for xb, yb, lb in test_loader:
            xb, lb = xb.to(device), lb.to(device)
            logits = spiking(xb, make_pad_mask(lb, xb.shape[1], device))
            valid = yb.reshape(-1) != -1
            sp_preds.append(
                logits.reshape(-1, logits.shape[-1]).argmax(1).cpu()[valid].numpy())
            sp_trues.append(yb.reshape(-1)[valid].numpy())
        lif_rates = monitor.rates()
    sp_preds, sp_trues = np.concatenate(sp_preds), np.concatenate(sp_trues)

    print("\n[Spiking] Classification report (test set):")
    print(classification_report(sp_trues, sp_preds, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    sp_task = (sp_trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if sp_task.any():
        print(f"[Spiking] Task-only accuracy (baseline chunks excluded): "
              f"{(sp_preds[sp_task] == sp_trues[sp_task]).mean():.4f}")
    print("[Spiking] Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(sp_trues, sp_preds, labels=list(range(len(classes)))))

    print("\n=== ANN attention vs spiking attention (full test set, "
          f"T={args.timesteps}) ===")
    print(f"ANN throughout:                        test_acc={test_acc:.4f}  "
          f"({args.epochs} epoch budget)")
    print(f"Spiking CNNs + Spikformer attention:   test_acc={snn_acc:.4f}  "
          f"(+{snn_epochs} more on frozen CNNs)")
    print("These budgets are not equal -- phase 2 trains on top of phase 1, so "
          "the spiking model has seen the training set for longer, and freezing "
          "the CNN to retrain the head is itself a regulariser. Read the gap as "
          "'spiking attention can carry this task', not as a like-for-like win.")
    print(f"\n[Spiking] Mean spike rate per layer (fraction of neuron-timesteps "
          f"that fired; lower is sparser and cheaper on spiking hardware):")
    for key, rate in {**spiking.spike_rates, **lif_rates}.items():
        print(f"  {key:>28}: {rate:.4f}")

    if save_path:
        save_checkpoint(save_path, ann_state=model.state_dict(), ann_config=ann_config,
                        thresholds=thresholds, token_scales=token_scales,
                        spiking_state=spiking.state_dict(), spiking_config=spiking_config,
                        classes=classes)


if __name__ == "__main__":
    main()
