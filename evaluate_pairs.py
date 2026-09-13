"""
Evaluate the pair-constraint and error structure of the singleview Spikformer.

Usage:
    python evaluate_pairs.py
    python evaluate_pairs.py --split test10
    python evaluate_pairs.py --split 35
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "transformer", ROOT / "transformer" / "multiview",
          ROOT / "transformer" / "singleview"):
    sys.path.insert(0, str(p))

from sv_spikformer import load_model
from predict import read_chunks, truth_for
from train_multiview_transformer import make_pad_mask
from splits import subject_count_split


def get_test_subjects(n_val=0, seed=42):
    cache_path = ROOT / ".cache" / "mv_seq_raw_all.npz"
    d = np.load(cache_path)
    sids, runs = d["sids"], d["runs"]
    _, _, _, test_s = subject_count_split(sids, runs, n_train=90, n_val_subjects=n_val, seed=seed)
    return test_s


def evaluate(subject_list, checkpoint_path, device="cpu"):
    data_dir = ROOT / "physionet.org" / "files" / "eegmmidb" / "1.0.0"
    model, classes = load_model(checkpoint_path, device=device)
    max_len = model.pos.num_embeddings
    runs = [4, 8, 12]

    total_pairs = 0
    total_paired_trials = 0
    paired_hits = 0

    all_trials_total = 0
    all_trials_hits = 0

    both_wrong_count = 0
    one_wrong_count = 0
    both_right_count = 0

    t1_hits = 0
    t1_total = 0
    t2_hits = 0
    t2_total = 0

    sub_rows = []
    sub_diffs = []

    print(f"Evaluating {len(subject_list)} subjects: {subject_list}")
    print(f"Checkpoint: {checkpoint_path.name}\n")

    for sid in subject_list:
        sname = f"S{sid:03d}"
        s_all_hits, s_all_n = 0, 0
        s_t1_hits, s_t1_n = 0, 0
        s_t2_hits, s_t2_n = 0, 0

        for r in runs:
            edf_path = data_dir / sname / f"{sname}R{r:02d}.edf"
            if not edf_path.exists():
                continue

            x, onsets, marks = read_chunks(str(edf_path))
            truth, _ = truth_for(str(edf_path), marks, classes)

            xt = torch.from_numpy(x).unsqueeze(0)
            logits = []
            with torch.no_grad():
                for s in range(0, xt.shape[1], max_len):
                    xb = xt[:, s:s + max_len].to(device)
                    lengths = torch.tensor([xb.shape[1]])
                    logits.append(model(xb, make_pad_mask(lengths, xb.shape[1], device))[0].cpu())
            probs = torch.cat(logits).softmax(-1)
            pred = probs.argmax(-1).tolist()
            pred_labels = [classes[p] for p in pred]

            # Filter cued task trials (exclude T0 baseline)
            task_trials = []
            for mark, pred_l, t in zip(marks, pred_labels, truth):
                if mark != "T0" and t is not None:
                    task_trials.append((pred_l, t))
                    s_all_hits += int(pred_l == t)
                    s_all_n += 1

            # In a 15-trial run, pairs are (0,1), (2,3), ..., (12,13)
            n_pairs = len(task_trials) // 2
            for p in range(n_pairs):
                pred1, true1 = task_trials[2 * p]
                pred2, true2 = task_trials[2 * p + 1]

                c1 = (pred1 == true1)
                c2 = (pred2 == true2)

                total_pairs += 1
                total_paired_trials += 2
                paired_hits += int(c1) + int(c2)

                t1_hits += int(c1)
                t1_total += 1
                t2_hits += int(c2)
                t2_total += 1

                s_t1_hits += int(c1)
                s_t1_n += 1
                s_t2_hits += int(c2)
                s_t2_n += 1

                if not c1 and not c2:
                    both_wrong_count += 1
                elif c1 and c2:
                    both_right_count += 1
                else:
                    one_wrong_count += 1

        all_trials_hits += s_all_hits
        all_trials_total += s_all_n

        if s_all_n > 0:
            overall_acc = s_all_hits / s_all_n
            t1_acc = s_t1_hits / s_t1_n if s_t1_n else 0
            t2_acc = s_t2_hits / s_t2_n if s_t2_n else 0
            diff = (t2_acc - t1_acc) * 100
            sub_diffs.append(t2_acc - t1_acc)
            sub_rows.append((sname, s_all_hits, s_all_n, overall_acc, t1_acc, t2_acc, diff))

    # Print per-subject table
    print(f"{'Subject':<8} {'Overall (all 15)':<18} {'T1 Acc':<12} {'T2 Acc':<12} {'Asymmetry (T2 - T1)':<20}")
    print("-" * 72)
    for sname, h, n, o_acc, t1, t2, diff in sub_rows:
        print(f"{sname:<8} {h:>3}/{n:<3} ({o_acc*100:>5.2f}%)     {t1*100:>5.2f}%      {t2*100:>5.2f}%       {diff:>+6.2f} pts")
    print("-" * 72)

    # Summary calculations
    p_overall = all_trials_hits / all_trials_total
    p_paired = paired_hits / total_paired_trials

    t1_acc_total = t1_hits / t1_total
    t2_acc_total = t2_hits / t2_total
    asymmetry_total = (t2_acc_total - t1_acc_total) * 100

    # Probabilities using OVERALL ACCURACY (p_overall)
    exp_both_wrong = (1 - p_overall) ** 2
    obs_both_wrong = both_wrong_count / total_pairs

    exp_one_wrong = 2 * p_overall * (1 - p_overall)
    obs_one_wrong = one_wrong_count / total_pairs

    t_stat, p_val = stats.ttest_1samp(sub_diffs, 0.0)

    # Pairing effect decomposition on the 3.94% attention gain
    pairing_boost = 0.467 * asymmetry_total
    remaining_attention = 3.94 - pairing_boost

    print(f"\n{'=' * 72}")
    print("SUMMARY RESULTS (USING OVERALL ACCURACY):")
    print(f"{'=' * 72}")
    print(f"Overall Accuracy (all {all_trials_total} trials):         {all_trials_hits}/{all_trials_total} = {p_overall*100:.2f}% ({p_overall:.4f})")
    print(f"Paired Accuracy (14 trials/run, n={total_paired_trials}):   {paired_hits}/{total_paired_trials} = {p_paired*100:.2f}% ({p_paired:.4f})")
    print()
    print(f"First-of-pair  (T1, unconstrained):       {t1_hits}/{t1_total} = {t1_acc_total*100:.2f}%")
    print(f"Second-of-pair (T2, constrained):         {t2_hits}/{t2_total} = {t2_acc_total*100:.2f}%")
    print(f"Asymmetry gain (T2 - T1):                 {asymmetry_total:+.2f} percentage points")
    print(f"Statistical test (paired t-test, n={len(sub_diffs)}):  t = {t_stat:+.2f}, p-value = {p_val:.4f}")
    print()
    print("Error Structure (Baseline = Overall Accuracy):")
    print(f"  P(both members wrong), observed:        {obs_both_wrong:.4f} ({obs_both_wrong*100:.2f}%)")
    print(f"  P(both members wrong), if independent:  {exp_both_wrong:.4f} ({exp_both_wrong*100:.2f}%)  [ratio: {obs_both_wrong/exp_both_wrong:.2f}x higher]")
    print()
    print(f"  P(exactly one wrong),  observed:        {obs_one_wrong:.4f} ({obs_one_wrong*100:.2f}%)")
    print(f"  P(exactly one wrong),  if independent:  {exp_one_wrong:.4f} ({exp_one_wrong*100:.2f}%)  [ratio: {(1 - obs_one_wrong/exp_one_wrong)*100:.1f}% lower]")
    print()
    print("Decomposition of 3.94% Attention Gain:")
    print(f"  Protocol structure (0.467 * {asymmetry_total:.2f}%):    {pairing_boost:.2f} percentage points (~{pairing_boost/3.94*100:.1f}%)")
    print(f"  Genuine cross-trial context:            {remaining_attention:.2f} percentage points (~{remaining_attention/3.94*100:.1f}%)")
    print(f"{'=' * 72}")

    return {
        "overall_acc": p_overall,
        "paired_acc": p_paired,
        "t1_acc": t1_acc_total,
        "t2_acc": t2_acc_total,
        "asymmetry": asymmetry_total,
        "both_wrong_obs": obs_both_wrong,
        "both_wrong_exp": exp_both_wrong,
        "one_wrong_obs": obs_one_wrong,
        "one_wrong_exp": exp_one_wrong,
    }


def main():
    import re
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test13", "test10", "35"], default="test13",
                        help="Subject split: test13 (default: 13 held-out test subjects), "
                             "test10 (10 held-out test subjects), or 35 (S001-S035)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pt file (default: newest in checkpoints/)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed used for subject split (default: auto-detected from checkpoint filename, else 42)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Evaluate all three checkpoints (seed42, seed43, seed44) on their respective test sets")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    if args.all_seeds:
        seeds = [42, 43, 44]
        results = {}
        for s in seeds:
            ckpt_path = ROOT / "checkpoints" / f"seed{s}rhlh.pt"
            if not ckpt_path.exists():
                candidates = list((ROOT / "checkpoints").glob(f"*seed{s}*.pt"))
                ckpt_path = candidates[0] if candidates else None
            if ckpt_path is None or not ckpt_path.exists():
                print(f"Warning: Checkpoint for seed {s} not found, skipping.")
                continue
            subs = get_test_subjects(n_val=0, seed=s)
            print(f"\n>>> RUNNING SEED {s} on its held-out test subjects (seed={s}) <<<")
            results[s] = evaluate(subs, ckpt_path, device=args.device)

        print("\n" + "#" * 72)
        print("SUMMARY ACROSS ALL 3 SEEDS (ON THEIR RESPECTIVE HELD-OUT TEST SUBJECTS):")
        print("#" * 72)
        print(f"{'Seed':<8} {'Overall Acc':<16} {'T1 Acc (Unconstrained)':<24} {'T2 Acc (Constrained)':<22} {'Asymmetry'}")
        print("-" * 72)
        for s, res in results.items():
            print(f"Seed {s:<3} {res['overall_acc']*100:>6.2f}%          {res['t1_acc']*100:>6.2f}%                  {res['t2_acc']*100:>6.2f}%                 {res['asymmetry']:>+5.2f} pts")
        print("-" * 72)
        avg_overall = np.mean([r["overall_acc"] for r in results.values()]) * 100
        avg_t1 = np.mean([r["t1_acc"] for r in results.values()]) * 100
        avg_t2 = np.mean([r["t2_acc"] for r in results.values()]) * 100
        avg_asym = np.mean([r["asymmetry"] for r in results.values()])
        print(f"{'Mean':<8} {avg_overall:>6.2f}%          {avg_t1:>6.2f}%                  {avg_t2:>6.2f}%                 {avg_asym:>+5.2f} pts")
        print("#" * 72)
        return

    ckpt = args.checkpoint
    if ckpt is None:
        found = sorted((ROOT / "checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit("No checkpoint found in checkpoints/")
        ckpt = found[-1]
    else:
        ckpt = Path(ckpt)

    # Auto-detect seed from filename if not specified
    if args.seed is not None:
        seed = args.seed
    else:
        m = re.search(r"seed(\d+)", ckpt.name, re.IGNORECASE)
        seed = int(m.group(1)) if m else 42

    print(f"Using split seed = {seed} for checkpoint {ckpt.name}")

    if args.split == "test13":
        subs = get_test_subjects(n_val=0, seed=seed)
    elif args.split == "test10":
        subs = get_test_subjects(n_val=3, seed=seed)
    elif args.split == "35":
        subs = list(range(1, 36))

    evaluate(subs, ckpt, device=args.device)


if __name__ == "__main__":
    main()
