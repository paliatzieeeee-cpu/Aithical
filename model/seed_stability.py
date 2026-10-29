"""
Seed stability of the ethics-dimension classifier, stage by stage.

Usage (from the project root):

    python -m model.seed_stability                 # 10 seeds, 2000 bootstrap resamples
    python -m model.seed_stability --seeds 5 --boot 1000

The five training-set stages of the thesis (240, 261, 345, 408, 444 examples)
are rebuilt from model/training_data.py: the original examples, the first N
LLM-augmented examples in order of approval, and the Greek batch. Each stage
is retrained with several seeds (same head, split and early stopping as
model/train_classifier.py) and measured on the fixed 62-example test set of
model/evaluate_classifier.py.

Reports mean ± sd over seeds, 95% bootstrap intervals over the test set
(pooled over seeds), a paired bootstrap of the 240 -> 444 difference and the
micro precision/recall of the final stage at lower thresholds.

Nothing is saved to saved_model/: the production model is not touched.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import argparse
import ast
import json
import os

import numpy as np
from sklearn.metrics import hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from model.ethics_classifier import build_classifier_head, embed_texts
from model.evaluate_classifier import HELD_OUT_EXAMPLES, THRESHOLD, EPOCHS, BATCH_SIZE
from model.train_classifier import VALIDATION_SPLIT
from model.training_data import CATEGORIES, TRAINING_EXAMPLES

# Πλήθος παραδειγμάτων LLM που είχαν εγκριθεί σε κάθε στάδιο
STAGES = {240: 0, 261: 21, 345: 105, 408: 168, 444: 168}
LOW_THRESHOLDS = [0.45, 0.35, 0.25]


# -----------------------------------------------------------------------
# Ανασύνθεση των σταδίων: η θέση κάθε παραδείγματος στο training_data.py
# δείχνει σε ποιο μπλοκ ανήκει (αρχικά, ελληνικό batch, επέκταση LLM)
# -----------------------------------------------------------------------
def split_blocks():
    path = os.path.join(os.path.dirname(__file__), "training_data.py")
    src = open(path, encoding="utf-8").read()
    lines = src.split("\n")
    greek_start = next(i for i, l in enumerate(lines, 1) if "ΕΛΛΗΝΙΚΟ BATCH" in l)
    llm_start = next(i for i, l in enumerate(lines, 1) if "ΣΥΣΤΗΜΑΤΙΚΗ ΕΠΕΚΤΑΣΗ ΜΕΣΩ LLM" in l)
    llm_end = next(i for i, l in enumerate(lines, 1) if l.strip() == "# --- end of LLM-augmented examples ---")
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and n.targets[0].id == "TRAINING_EXAMPLES")
    assert len(node.value.elts) == len(TRAINING_EXAMPLES)
    original, greek, llm = [], [], []
    for elt, example in zip(node.value.elts, TRAINING_EXAMPLES):
        if greek_start <= elt.lineno < llm_start:
            greek.append(example)
        elif llm_start <= elt.lineno < llm_end:
            llm.append(example)
        else:
            original.append(example)
    return original, greek, llm


def stage_examples(size, original, greek, llm):
    examples = original + llm[:STAGES[size]] + (greek if size == 444 else [])
    assert len(examples) == size, (size, len(examples))
    return examples


def to_matrix(label_lists):
    return np.array([[1 if c in labels else 0 for c in CATEGORIES] for labels in label_lists])


def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    mp, mr, _, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return {
        "macro_f1": float(np.mean(f1)),
        "micro_p": float(mp),
        "micro_r": float(mr),
        "subset_acc": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming": float(hamming_loss(y_true, y_pred)),
        "f1": {c: float(v) for c, v in zip(CATEGORIES, f1)},
    }


def macro_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return float(np.mean(f1))


# -----------------------------------------------------------------------
# Εκπαίδευση μίας κεφαλής με δεδομένο seed (ίδια διαδικασία με το
# model/train_classifier.py, χωρίς αποθήκευση)
# -----------------------------------------------------------------------
def train_head(x, y, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=VALIDATION_SPLIT, random_state=seed, shuffle=True)
    model = build_classifier_head(embedding_dim=x.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),
              callbacks=[stop], verbose=0)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--out", default=None, help="optional JSON file for the raw results")
    args = parser.parse_args()

    original, greek, llm = split_blocks()
    print(f"original {len(original)} | LLM-augmented {len(llm)} | Greek batch {len(greek)}")

    all_texts = [t for t, _ in TRAINING_EXAMPLES]
    cache = dict(zip(all_texts, embed_texts(all_texts)))
    test_texts = [t for t, _ in HELD_OUT_EXAMPLES]
    x_test = embed_texts(test_texts)
    y_test = to_matrix([labels for _, labels in HELD_OUT_EXAMPLES])
    n_test = len(test_texts)

    rng = np.random.default_rng(12345)
    boot_idx = rng.integers(0, n_test, size=(args.boot, n_test))
    seeds = list(range(args.seeds))
    results, probs_by_stage = {}, {}

    for size in STAGES:
        examples = stage_examples(size, original, greek, llm)
        x = np.array([cache[t] for t, _ in examples])
        y = to_matrix([labels for _, labels in examples]).astype("float32")
        runs, probs = [], []
        for seed in seeds:
            prob = train_head(x, y, seed).predict(x_test, verbose=0)
            probs.append(prob)
            runs.append(metrics(y_test, (prob >= THRESHOLD).astype(int)))
        probs_by_stage[size] = probs
        results[size] = runs
        mf = [r["macro_f1"] for r in runs]
        print(f"stage {size}: macro F1 {np.mean(mf):.3f} ± {np.std(mf, ddof=1):.3f} "
              f"(min {min(mf):.2f}, max {max(mf):.2f})")

    # ---------------------------------------------------------------
    # Σύνοψη: μέσος όρος ± τυπική απόκλιση και διαστήματα bootstrap
    # ---------------------------------------------------------------
    summary = {}
    for size, runs in results.items():
        preds = [(p >= THRESHOLD).astype(int) for p in probs_by_stage[size]]
        boot = [macro_f1(y_test[idx], pr[idx]) for pr in preds for idx in boot_idx]
        row = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1)))
               for k in ("macro_f1", "micro_p", "micro_r", "subset_acc", "hamming")}
        row["f1"] = {c: (float(np.mean([r["f1"][c] for r in runs])), float(np.std([r["f1"][c] for r in runs], ddof=1)))
                     for c in CATEGORIES}
        row["macro_f1_ci95"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        summary[size] = row

    print("\nstage | macro F1 (mean ± sd) [95% bootstrap CI] | micro P | micro R | subset acc | Hamming")
    for size, row in summary.items():
        lo, hi = row["macro_f1_ci95"]
        print(f"{size:5} | {row['macro_f1'][0]:.3f} ± {row['macro_f1'][1]:.3f} [{lo:.2f}, {hi:.2f}] | "
              f"{row['micro_p'][0]:.2f} ± {row['micro_p'][1]:.2f} | {row['micro_r'][0]:.2f} ± {row['micro_r'][1]:.2f} | "
              f"{row['subset_acc'][0]:.2f} ± {row['subset_acc'][1]:.2f} | {row['hamming'][0]:.3f} ± {row['hamming'][1]:.3f}")
    print("\nF1 per dimension (mean ± sd):")
    for size, row in summary.items():
        print(f"{size:5} | " + " | ".join(f"{c} {m:.2f}±{s:.2f}" for c, (m, s) in row["f1"].items()))

    # ---------------------------------------------------------------
    # Ζευγαρωτό bootstrap για τη διαφορά βάσης αναφοράς -> τελικού σταδίου
    # (ίδιο seed και ίδιο δείγμα του συνόλου δοκιμής στις δύο πλευρές)
    # ---------------------------------------------------------------
    paired = {}
    for a, b in [(240, 444), (408, 444), (240, 408)]:
        diffs = []
        for pa, pb in zip(probs_by_stage[a], probs_by_stage[b]):
            ya, yb = (pa >= THRESHOLD).astype(int), (pb >= THRESHOLD).astype(int)
            diffs += [macro_f1(y_test[idx], yb[idx]) - macro_f1(y_test[idx], ya[idx]) for idx in boot_idx]
        diffs = np.array(diffs)
        paired[f"{a}->{b}"] = {"mean": float(diffs.mean()), "ci95": (float(np.percentile(diffs, 2.5)),
                               float(np.percentile(diffs, 97.5))), "p_le_0": float(np.mean(diffs <= 0))}
        print(f"\nmacro F1 {a} -> {b}: {diffs.mean():+.3f} [{np.percentile(diffs, 2.5):+.3f}, "
              f"{np.percentile(diffs, 97.5):+.3f}], share of resamples <= 0: {np.mean(diffs <= 0):.3f}")

    # ---------------------------------------------------------------
    # Σημείο λειτουργίας: precision/recall του τελικού σταδίου σε
    # χαμηλότερα κατώφλια
    # ---------------------------------------------------------------
    operating = {}
    print("\nFinal stage (444) at lower thresholds (mean over seeds):")
    for t in LOW_THRESHOLDS:
        runs = [metrics(y_test, (p >= t).astype(int)) for p in probs_by_stage[444]]
        operating[t] = {k: float(np.mean([r[k] for r in runs])) for k in ("micro_p", "micro_r", "macro_f1", "subset_acc")}
        o = operating[t]
        print(f"  threshold {t:.2f}: micro P {o['micro_p']:.2f} | micro R {o['micro_r']:.2f} | "
              f"macro F1 {o['macro_f1']:.2f} | subset acc {o['subset_acc']:.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"seeds": seeds, "boot": args.boot, "runs": results, "summary": summary,
                       "paired": paired, "operating": operating}, f, indent=1)
        print(f"\nRaw results written to {args.out}")


if __name__ == "__main__":
    main()
