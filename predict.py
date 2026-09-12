"""
Predict left vs right imagined fist from a raw EDF with a saved checkpoint
"""
import argparse
import re
import sys
from pathlib import Path

import mne
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "transformer", ROOT / "transformer" / "multiview",
          ROOT / "transformer" / "singleview"):
    sys.path.insert(0, str(p))

from sv_spikformer import load_model
from train_multiview_transformer import CANONICAL_CHANNELS, make_pad_mask
from mlp.train_mlp import EPOCH_TMIN, N_SAMPLES, RUN_LABELS

SFREQ = 160.0  # what the model was trained at; anything else gets resampled


def read_chunks(edf_path):
    """One EDF -> (T, 64, 640) chunks in recording order + their markers.

    Three things here exist only to survive a file that is not from the
    training set, since we are told the test EDFs may not be:

    * Resampling. EEGMMIDB itself is mixed -- S088/S092/S100 are 128 Hz, and
      a 4 s epoch there is 513 samples, not 640.
    * Name-matched channels, zero-filled when absent. A file with a different
      montage still lines up on whatever it shares. Zero is the right filler
      because training used channel_drop=0.1, so a dead channel is a
      perturbation the encoder has already seen.
    * Slicing on annotation onsets rather than mne.Epochs, so markers named
      anything at all still cut the same windows.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.rename_channels({c: c.rstrip(".") for c in raw.ch_names})
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose="ERROR")

    have = {c.lower(): i for i, c in enumerate(raw.ch_names)}
    data = raw.get_data()
    rows, missing = [], []
    for c in CANONICAL_CHANNELS:
        i = have.get(c.lower())
        rows.append(data[i] if i is not None else np.zeros(data.shape[1]))
        if i is None:
            missing.append(c)
    data = np.stack(rows)
    if missing:
        print(f"  warning: {len(missing)}/64 channels absent, zero-filled "
              f"({', '.join(missing[:6])}{', ...' if len(missing) > 6 else ''})")

    onsets = raw.annotations.onset
    marks = list(raw.annotations.description)
    if len(onsets) == 0:
        # No markers at all: tile the recording into back-to-back 4 s windows
        # so the file is at least decodable, and say so.
        step = N_SAMPLES / SFREQ
        onsets = np.arange(0, raw.n_times / SFREQ - step, step)
        marks = ["-"] * len(onsets)
        print(f"  warning: no annotations, tiling {len(onsets)} fixed 4 s windows")

    chunks, kept_onsets, kept_marks = [], [], []
    for onset, mark in zip(onsets, marks):
        s = int(round((onset + EPOCH_TMIN) * SFREQ))
        if s < 0 or s + N_SAMPLES > data.shape[1]:
            continue  # window runs off the end of the recording
        chunks.append(data[:, s:s + N_SAMPLES])
        kept_onsets.append(float(onset))
        kept_marks.append(str(mark))
    if not chunks:
        raise SystemExit(f"{edf_path}: no 4 s window fits inside this recording")

    x = np.stack(chunks).astype(np.float32)
    # Per-recording, per-channel z-score -- the deployment-time stand-in for
    # training's per-SUBJECT z-score. Uses only this file's own signal.
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    return (x - mean) / std, kept_onsets, kept_marks


def truth_for(edf_path, marks, classes):
    """Ground truth per chunk, or None where there is none.

    The annotations ARE the answer key, but only once you know the run: T1 is
    left_fist in runs 4/8/12 and both_fists in 6/10/14, so the run number in
    the EEGMMIDB filename is what makes scoring possible. A file named
    anything else scores nothing rather than guessing, and a run whose classes
    this checkpoint was never trained on (feet, for a left/right model) is
    left unscored too -- that mismatch is the thing worth being told about.
    """
    m = re.search(r"R(\d{2})\.edf$", str(edf_path), re.IGNORECASE)
    label_map = RUN_LABELS.get(int(m.group(1))) if m else None
    if label_map is None:
        return [None] * len(marks), None
    truth = [label_map.get(k) for k in marks]
    known = {t for t in truth if t is not None}
    if known and not (known & set(classes)):
        print(f"  warning: run {int(m.group(1))} is a {'/'.join(sorted(known - {'baseline'}))} "
              f"run -- this checkpoint predicts {'/'.join(classes)}, so its output "
              f"here is meaningless")
    return [t if t in classes else None for t in truth], int(m.group(1))


def predict_edf(edf_path, checkpoint, device="cpu"):
    """Returns ([(onset_s, marker, label, confidence)], class_names)."""
    model, classes = load_model(checkpoint, device)
    x, onsets, marks = read_chunks(edf_path)

    # The context transformer reads a whole run causally, so chunks go in
    # together and in order -- not one at a time. Its learned positions run
    # out at max_len, so a longer recording is processed in consecutive
    # blocks of that size.
    max_len = model.pos.num_embeddings
    xt = torch.from_numpy(x).unsqueeze(0)
    logits = []
    with torch.no_grad():
        for s in range(0, xt.shape[1], max_len):
            xb = xt[:, s:s + max_len].to(device)
            lengths = torch.tensor([xb.shape[1]])
            logits.append(model(xb, make_pad_mask(lengths, xb.shape[1], device))[0].cpu())
    probs = torch.cat(logits).softmax(-1)
    conf, pred = probs.max(-1)
    rows = [(o, m, classes[p], float(c))
            for o, m, p, c in zip(onsets, marks, pred.tolist(), conf.tolist())]
    return rows, classes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("edf", nargs="+", help="raw .edf file(s)")
    ap.add_argument("--checkpoint", default=None,
                    help="default: newest .pt in checkpoints/")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--all-chunks", action="store_true",
                    help="also print rest (T0) chunks, which a left/right model "
                         "only ever saw as unlabelled context")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        found = sorted((ROOT / "checkpoints").glob("*.pt"),
                       key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit("No checkpoint found: pass --checkpoint, or train "
                             "with --save to populate checkpoints/")
        ckpt = found[-1]
    print(f"Checkpoint: {ckpt}\n")

    grand_hit = grand_n = 0
    for path in args.edf:
        print(f"{path}")
        rows, classes = predict_edf(path, ckpt, args.device)
        truth, _ = truth_for(path, [r[1] for r in rows], classes)

        # Rest chunks carry no answer unless the checkpoint can actually
        # predict baseline, so by default only the cued trials are shown.
        keep = [i for i, r in enumerate(rows)
                if args.all_chunks or "baseline" in classes or r[1].upper() != "T0"]
        keep = keep or list(range(len(rows)))

        scored = [(rows[i][2], truth[i]) for i in keep if truth[i] is not None]
        head = f"  {'#':>3} {'onset_s':>8} {'marker':>7} {'prediction':>11} {'conf':>6}"
        print(head + (f" {'truth':>11}" if scored else ""))
        for n, i in enumerate(keep, 1):
            onset, mark, label, conf = rows[i]
            line = f"  {n:>3} {onset:>8.1f} {mark:>7} {label:>11} {conf:>6.2f}"
            if scored:
                line += (f" {truth[i]:>11} {'ok' if label == truth[i] else 'MISS'}"
                         if truth[i] is not None else f" {'-':>11}")
            print(line)

        counts = {}
        for i in keep:
            counts[rows[i][2]] = counts.get(rows[i][2], 0) + 1
        print(f"  {len(keep)} trials: " +
              ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if scored:
            hit = sum(p == t for p, t in scored)
            grand_hit += hit
            grand_n += len(scored)
            print(f"  accuracy = {hit}/{len(scored)} = {hit / len(scored):.3f}")
        else:
            print("  accuracy = n/a (no ground truth recoverable from this filename)")
        print()

    if grand_n and len(args.edf) > 1:
        print(f"TOTAL: {grand_hit}/{grand_n} = {grand_hit / grand_n:.4f} "
              f"over {len(args.edf)} files")


if __name__ == "__main__":
    main()
