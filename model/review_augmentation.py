"""
Ανθρώπινη έγκριση των υποψηφίων που παρήγαγε το model/augment_with_llm.py.

Κάθε υποψήφιο εμφανίζεται με το θέμα, τη γλώσσα και τις ετικέτες για τις
οποίες ζητήθηκε. Ο ερευνητής αποφασίζει:

  Enter            -> αποδοχή με τις προτεινόμενες ετικέτες
  ετικέτες,με,κόμμα -> αποδοχή με ΔΙΚΕΣ ΣΟΥ ετικέτες (αν το κείμενο
                      αγγίζει κι άλλη διάσταση ή όχι αυτή που ζητήθηκε)
  n                -> αποδοχή ως ουδέτερο (καμία ηθική διάσταση)
  d                -> απόρριψη (κακό, αφύσικο, λάθος ελληνικά, αμφίσημο)
  s                -> παράλειψη (μένει για την επόμενη φορά)
  q                -> έξοδος

Τα εγκεκριμένα μπαίνουν στο model/training_data.py ακριβώς πάνω από τη
γραμμή-δείκτη "end of LLM-augmented examples", δηλαδή σε δικό τους,
ξεχωριστό block — ποτέ ανάμεσα στα prompts του review_log. Κάθε απόφαση
καταγράφεται στο data/augmentation_reviewed.jsonl ως ίχνος ελέγχου.

Εκτέλεση:
    python -m model.review_augmentation
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import json
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
CANDIDATES_PATH = os.path.join(ROOT, "data", "augmentation_candidates.jsonl")
REVIEWED_PATH = os.path.join(ROOT, "data", "augmentation_reviewed.jsonl")
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.py")
AUG_MARKER = "    # --- end of LLM-augmented examples ---"

VALID = ["transparency", "fairness", "non_maleficence", "accountability", "privacy"]


def _normalize(text):
    return " ".join(text.lower().split())


def _load(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_candidates(entries):
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _append_reviewed(entry, decision, final_labels):
    record = dict(entry)
    record["decision"] = decision
    record["final_labels"] = final_labels
    with open(REVIEWED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------
# Εισαγωγή ενός εγκεκριμένου παραδείγματος στο training_data.py, πάνω από
# τον δείκτη του block επέκτασης
# -----------------------------------------------------------------------
def _append_to_training_data(text, labels):
    with open(TRAINING_DATA_PATH, encoding="utf-8") as f:
        content = f.read()
    if content.count(AUG_MARKER) != 1:
        raise SystemExit(
            "Δεν βρέθηκε (ή βρέθηκε πολλές φορές) η γραμμή-δείκτης\n"
            f"{AUG_MARKER.strip()}\nστο model/training_data.py. Πρόσθεσέ την ξανά πριν συνεχίσεις."
        )
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    labels_str = ", ".join(f'"{l}"' for l in labels)
    line = f'    ("{escaped}", [{labels_str}]),\n'
    content = content.replace(AUG_MARKER, line + AUG_MARKER, 1)
    with open(TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    candidates = _load(CANDIDATES_PATH)
    if not candidates:
        print("Δεν υπάρχουν υποψήφια. Τρέξε πρώτα: python -m model.augment_with_llm")
        return

    # Φρέσκια φόρτωση του training_data.py για απόρριψη όσων υπάρχουν ήδη
    import importlib
    import model.training_data as td
    importlib.reload(td)
    in_training = {_normalize(t) for t, _ in td.TRAINING_EXAMPLES}

    from model.evaluate_classifier import HELD_OUT_EXAMPLES
    in_heldout = {_normalize(t) for t, _ in HELD_OUT_EXAMPLES}

    pending, dropped = [], 0
    seen = set()
    for c in candidates:
        norm = _normalize(c["text"])
        if norm in in_training or norm in in_heldout or norm in seen:
            dropped += 1
            continue
        seen.add(norm)
        pending.append(c)
    if dropped:
        print(f"Αφαιρέθηκαν αυτόματα {dropped} υποψήφια (ήδη στην εκπαίδευση, στο held-out, ή διπλότυπα).")
        _write_candidates(pending)
    if not pending:
        print("Δεν απομένουν υποψήφια προς έλεγχο.")
        return

    # Σειρά εμφάνισης «ένα από κάθε κελί ανά γύρο»: τα υποψήφια παράγονται
    # θέμα-θέμα, οπότε μια μερική έγκριση με τη σειρά παραγωγής θα έβαζε
    # στην εκπαίδευση μόνο τα πρώτα θέματα, ξαναδημιουργώντας ακριβώς την
    # ανισορροπία θέματος που το πλέγμα φτιάχτηκε να σπάσει. Με εναλλαγή
    # κελιών, όπου κι αν σταματήσει ο έλεγχος, τα εγκεκριμένα καλύπτουν
    # ομοιόμορφα θέματα, διαστάσεις και ουδέτερα.
    import random
    from collections import OrderedDict
    cells = OrderedDict()
    for c in pending:
        cells.setdefault((c.get("topic"), tuple(c.get("labels", []))), []).append(c)
    keys = list(cells)
    random.Random(0).shuffle(keys)
    interleaved = []
    while any(cells[k] for k in keys):
        for k in keys:
            if cells[k]:
                interleaved.append(cells[k].pop(0))
    pending = interleaved

    print(f"\n{len(pending)} υποψήφια προς έλεγχο.")
    print("Enter=αποδοχή | ετικέτες,με,κόμμα=δικές σου | n=ουδέτερο | d=απόρριψη | s=παράλειψη | q=έξοδος")
    print(f"Έγκυρες ετικέτες: {', '.join(VALID)}\n")

    remaining = list(pending)
    accepted = discarded = 0

    for idx, entry in enumerate(pending, 1):
        print("-" * 70)
        print(f"[{idx}/{len(pending)}] θέμα: {entry.get('topic', '?')} | γλώσσα: {entry.get('lang', '?')}")
        print(f"Κείμενο: {entry['text']}")
        print(f"Προτεινόμενες ετικέτες: {entry['labels']}")
        answer = input("> ").strip()

        if answer.lower() == "q":
            break
        if answer.lower() == "s":
            continue
        if answer.lower() == "d":
            _append_reviewed(entry, "discarded", None)
            remaining.remove(entry)
            discarded += 1
            continue

        if answer == "":
            labels, decision = entry["labels"], "accepted"
        elif answer.lower() == "n":
            labels, decision = [], "accepted_neutral"
        else:
            labels = [l.strip() for l in answer.split(",") if l.strip()]
            bad = [l for l in labels if l not in VALID]
            if bad:
                print(f"  [!] Άγνωστες ετικέτες {bad} — το υποψήφιο παραμένει για την επόμενη φορά.")
                continue
            decision = "accepted_relabelled"

        _append_to_training_data(entry["text"], labels)
        _append_reviewed(entry, decision, labels)
        remaining.remove(entry)
        accepted += 1

    _write_candidates(remaining)
    print(f"\nΕγκρίθηκαν: {accepted} | Απορρίφθηκαν: {discarded} | Εκκρεμούν: {len(remaining)}")
    if accepted:
        print("Τρέξε: python -m model.train_classifier  και μετά  python -m model.evaluate_classifier")


if __name__ == "__main__":
    main()
