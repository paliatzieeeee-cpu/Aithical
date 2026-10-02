"""
Evaluate engine/risk_tier.py on the labelled scenarios of model/risk_tier_cases.py.

Usage:
    python -m model.evaluate_risk_tier            # both splits
    python -m model.evaluate_risk_tier --split dev
    python -m model.evaluate_risk_tier --split test --errors

Reports accuracy, per-tier precision / recall / F1, macro F1, the confusion
matrix and the two error types that matter most for an advisory tool:

    over-prohibition   a non-prohibited system shown as "unacceptable"
    under-estimation   a high-risk or prohibited system shown as limited/minimal
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import argparse
import json

import engine.risk_tier as risk_tier
from engine.risk_tier import assess_risk_tier
from model.risk_tier_cases import DEV_CASES, TEST_CASES, TEST2_CASES, TIERS

SPLITS = {"dev": DEV_CASES, "test": TEST_CASES, "test2": TEST2_CASES}

# The nearest-neighbour memory contains the labelled scenarios themselves,
# so each scenario is scored with itself left out (leave-one-out); otherwise
# it would simply find its own label.
risk_tier.KNN_EXCLUDE_IDENTICAL = True


# -----------------------------------------------------------------------
# Μετρικές για ένα σύνολο σεναρίων
# -----------------------------------------------------------------------
def evaluate(cases):
    preds = [assess_risk_tier(text, sector=sector, in_eu=True, lang="en")["tier"] for text, sector, _, _ in cases]
    golds = [gold for _, _, gold, _ in cases]

    confusion = {g: {p: 0 for p in TIERS} for g in TIERS}
    for g, p in zip(golds, preds):
        confusion[g][p] += 1

    per_tier = {}
    for t in TIERS:
        tp = confusion[t][t]
        predicted = sum(confusion[g][t] for g in TIERS)
        actual = sum(confusion[t].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_tier[t] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                       "support": actual}

    over_prohibition = sum(1 for g, p in zip(golds, preds) if p == "unacceptable" and g != "unacceptable")
    under_estimation = sum(1 for g, p in zip(golds, preds)
                           if g in ("unacceptable", "high") and p in ("limited", "minimal"))
    high_or_above = sum(1 for g in golds if g in ("unacceptable", "high"))
    errors = [(text, gold, pred) for (text, _, gold, _), pred in zip(cases, preds) if gold != pred]

    return {
        "n": len(cases),
        "accuracy": round(sum(g == p for g, p in zip(golds, preds)) / len(cases), 3),
        "macro_f1": round(sum(m["f1"] for m in per_tier.values()) / len(TIERS), 3),
        "per_tier": per_tier,
        "confusion": confusion,
        "over_prohibition": over_prohibition,
        "under_estimation": under_estimation,
        "high_or_above": high_or_above,
        "errors": errors,
    }


def _print(name, r, show_errors):
    print(f"\n=== {name} (n={r['n']}) ===")
    print(f"accuracy {r['accuracy']:.3f} | macro F1 {r['macro_f1']:.3f} | "
          f"over-prohibition {r['over_prohibition']} | "
          f"under-estimation {r['under_estimation']}/{r['high_or_above']}")
    print(f"{'tier':<13}{'P':>7}{'R':>7}{'F1':>7}{'n':>5}")
    for t, m in r["per_tier"].items():
        print(f"{t:<13}{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}{m['support']:>5}")
    print("confusion (rows = gold, cols = predicted):")
    print(" " * 13 + "".join(f"{t[:5]:>7}" for t in TIERS))
    for g in TIERS:
        print(f"{g:<13}" + "".join(f"{r['confusion'][g][p]:>7}" for p in TIERS))
    if show_errors:
        for text, gold, pred in r["errors"]:
            print(f"  [{gold} -> {pred}] {text[:110]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "test2", "all"], default="all")
    parser.add_argument("--errors", action="store_true", help="list misclassified scenarios")
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args()

    names = list(SPLITS) if args.split == "all" else [args.split]
    results = {n: evaluate(SPLITS[n]) for n in names}
    for n in names:
        _print(n, results[n], args.errors)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
