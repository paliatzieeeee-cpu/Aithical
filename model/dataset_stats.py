"""
Αναφορά ισορροπίας του συνόλου εκπαίδευσης.

Δείχνει πόσα παραδείγματα έχει κάθε ηθική διάσταση και πόσα ουδέτερα
υπάρχουν, τόσο στο model/training_data.py όσο και (ως πρόβλεψη) αν
εγκριθούν όλα τα εκκρεμή υποψήφια του data/augmentation_candidates.jsonl
με τις προτεινόμενες ετικέτες τους. Προτείνει την ακριβή εντολή
συμπλήρωσης για τις κατηγορίες που υστερούν.

Γιατί μετράει: ο ταξινομητής μαθαίνει και τη συχνότητα κάθε ετικέτας. Αν
μία κατηγορία είναι πολύ συχνότερη (π.χ. privacy), γίνεται η «εύκολη
απάντηση» και προβλέπεται και εκεί που δεν υπάρχει· αν μία είναι
σπάνια (π.χ. fairness), χάνεται. Λίγα ουδέτερα παραδείγματα κάνουν το
μοντέλο να βλέπει ηθικό πρόβλημα σε κάθε ερώτηση που αναφέρει ΑΙ.

Εκτέλεση:
    python -m model.dataset_stats
"""

import json
import os
import re
from collections import Counter

from model.training_data import CATEGORIES, TRAINING_EXAMPLES

ROOT = os.path.dirname(os.path.dirname(__file__))
CANDIDATES_PATH = os.path.join(ROOT, "data", "augmentation_candidates.jsonl")

TARGET_RATIO = 1.2         # μέγιστη/ελάχιστη κατηγορία: έως 1,2 θεωρείται ισορροπημένο
TARGET_NEUTRAL_SHARE = 0.10
EXAMPLES_PER_TOPUP_RUN = 36  # ~12 θέματα × 3 παραδείγματα ανά κελί, πριν τις απορρίψεις


def _counts(pairs):
    labels = Counter(l for _, ls in pairs for l in ls)
    neutral = sum(1 for _, ls in pairs if not ls)
    return labels, neutral


def _greek_share(pairs):
    if not pairs:
        return 0.0
    return sum(1 for t, _ in pairs if re.search(r"[α-ωΑ-Ωά-ώ]", t)) / len(pairs)


def main():
    train = [(t, ls) for t, ls in TRAINING_EXAMPLES]
    pending = []
    if os.path.isfile(CANDIDATES_PATH):
        with open(CANDIDATES_PATH, encoding="utf-8") as f:
            pending = [(r["text"], r["labels"]) for r in (json.loads(l) for l in f if l.strip())]

    t_lab, t_neu = _counts(train)
    p_lab, p_neu = _counts(pending)
    total = len(train) + len(pending)

    print(f"Εκπαίδευση: {len(train)} παραδείγματα | Εκκρεμή υποψήφια: {len(pending)}\n")
    print(f"{'':18}{'τώρα':>7}{'+εκκρεμή':>10}{'σύνολο':>9}")
    projected = {}
    for c in CATEGORIES:
        projected[c] = t_lab[c] + p_lab[c]
        print(f"{c:18}{t_lab[c]:>7}{p_lab[c]:>10}{projected[c]:>9}")
    neutral_total = t_neu + p_neu
    print(f"{'ουδέτερα':18}{t_neu:>7}{p_neu:>10}{neutral_total:>9}   ({neutral_total / max(total, 1):.0%} του συνόλου)")
    print(f"\nΕλληνικά στο σύνολο (αν εγκριθούν όλα): {_greek_share(train + pending):.0%}")

    hi = max(projected, key=projected.get)
    lo = min(projected, key=projected.get)
    ratio = projected[hi] / max(projected[lo], 1)
    print(f"\nΑναλογία συχνότερης/σπανιότερης κατηγορίας: {ratio:.2f} ({hi} / {lo})")
    if pending:
        print("(Η πρόβλεψη υποθέτει ότι όλα τα εκκρεμή εγκρίνονται με τις προτεινόμενες ετικέτες.)")

    top_up = [c for c in CATEGORIES if projected[hi] - projected[c] >= EXAMPLES_PER_TOPUP_RUN // 2]
    need_neutral = neutral_total < TARGET_NEUTRAL_SHARE * total

    if ratio <= TARGET_RATIO and not need_neutral:
        print("\n✔ Καλή ισορροπία. Επόμενο βήμα: έγκριση (αν υπάρχουν εκκρεμή) και εκπαίδευση.")
        return

    print("\nΠροτεινόμενη συμπλήρωση:")
    if top_up:
        print(f"  Υστερούν: {', '.join(top_up)} (κάθε τρέξιμο προσθέτει ~{EXAMPLES_PER_TOPUP_RUN} ανά κατηγορία πριν τις απορρίψεις)")
    if need_neutral:
        missing = int(TARGET_NEUTRAL_SHARE * total) - neutral_total
        print(f"  Ουδέτερα: λείπουν ~{missing} για να φτάσουν το {TARGET_NEUTRAL_SHARE:.0%}")
    wanted = top_up + (["neutral"] if need_neutral else [])
    if wanted:
        print("\n  python -m model.augment_with_llm --langs en --seed 7 --cells-for " + ",".join(wanted))
        print("  python -m model.review_augmentation")
        print("  python -m model.dataset_stats      # ξανά, μέχρι να ισορροπήσει")
    print(f"\n  Όσο υπάρχει ανισορροπία, ΜΗΝ εγκρίνεις κι άλλα παραδείγματα της «{hi}» αν διστάζεις γι' αυτά.")


if __name__ == "__main__":
    main()
