"""
Διαδραστική επιθεώρηση καταγεγραμμένων χρήσεων (data/usage_log.jsonl),
ώστε ο χρήστης της εφαρμογής (ο ερευνητής, όχι ο τελικός χρήστης) να
αποφασίσει ποιες πραγματικές υποβολές αξίζει να γίνουν νέα παραδείγματα
εκπαίδευσης — ο άνθρωπος βλέπει πάντα το κείμενο πριν αποφασίσει.

Για κάθε καταγεγραμμένη υποβολή, εμφανίζει το κείμενο και τις
βαθμολογίες του μοντέλου. Το Enter (χωρίς πληκτρολόγηση ετικετών)
αποδέχεται μια αυτόματη πρόταση με τρία επίπεδα εμπιστοσύνης βάσει της
κορυφαίας βαθμολογίας: μόνο η κορυφαία κατηγορία αν >= 70%, οι δύο
κορυφαίες αν >= 40%, αλλιώς (όλες οι βαθμολογίες χαμηλές) οι τρεις
κορυφαίες. Πληκτρολογώντας ετικέτες παρακάμπτεις την πρόταση, 'n' τη
σημειώνει ρητά ως ουδέτερη. Κάθε προστιθέμενη εγγραφή καταγράφεται στο
ιστορικό με το πεδίο "labeling_method", ώστε να φαίνεται καθαρά ποιες
ήταν αυτόματες (και σε ποιο επίπεδο) και ποιες χειροκίνητες.

Πριν την επιθεώρηση, αφαιρούνται αυτόματα υποβολές με κείμενο ΑΚΡΙΒΩΣ
ίδιο (όχι απλώς παρόμοιο) με κάποιο ήδη επιθεωρημένο δίλημμα ή με άλλη
υποβολή μέσα στην ίδια παρτίδα — π.χ. το ίδιο τεστ που υποβλήθηκε
πολλές φορές στη διάρκεια δοκιμών δεν χρειάζεται να επιθεωρηθεί ξανά.

Εκτέλεση:
    python -m model.review_log
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import json
import os

from model.training_data import CATEGORIES

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "usage_log.jsonl")
REVIEWED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "usage_log_reviewed.jsonl")
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.py")
INSERT_MARKER = "    # --- neutral / low-signal examples"

# Κατώφλια εμπιστοσύνης για την αυτόματη ετικέτηση όταν ο χρήστης απλώς
# πατάει Enter χωρίς να πληκτρολογήσει ο ίδιος ετικέτες. Ο άνθρωπος
# εξακολουθεί να βλέπει κάθε δίλημμα πριν αποφασίσει να πατήσει Enter (ή
# να το παρακάμψει πληκτρολογώντας κάτι άλλο) — αυτό απλώς κάνει την
# "αποδοχή της πρότασης του μοντέλου" την προεπιλεγμένη ενέργεια αντί να
# απαιτεί επαναπληκτρολόγηση των ίδιων ετικετών κάθε φορά.
AUTO_LABEL_HIGH_CONFIDENCE = 0.70  # σκορ >= αυτό -> μόνο η κορυφαία κατηγορία
AUTO_LABEL_MEDIUM_CONFIDENCE = 0.40  # σκορ >= αυτό -> οι 2 κορυφαίες κατηγορίες
# κάτω από το MEDIUM -> θεωρούνται "όλα χαμηλά", οι 3 κορυφαίες κατηγορίες


# -----------------------------------------------------------------------
# Αυτόματη επιλογή ετικετών βάσει της κατανομής βαθμολογιών του μοντέλου,
# με τρία επίπεδα εμπιστοσύνης. Επιστρέφει επίσης μια σύντομη ένδειξη
# ("μέθοδο") για το αρχείο ιστορικού, ώστε να ξεχωρίζουν καθαρά οι
# αυτόματα ετικετημένες εγγραφές από τις χειροκίνητες.
# -----------------------------------------------------------------------
def _auto_label_from_scores(sorted_scores):
    if not sorted_scores:
        return [], "auto_no_scores"

    top_score = sorted_scores[0][1]
    if top_score >= AUTO_LABEL_HIGH_CONFIDENCE:
        labels = [sorted_scores[0][0]]
        method = "auto_top1_high_confidence"
    elif top_score >= AUTO_LABEL_MEDIUM_CONFIDENCE:
        labels = [cat for cat, _ in sorted_scores[:2]]
        method = "auto_top2_medium_confidence"
    else:
        labels = [cat for cat, _ in sorted_scores[:3]]
        method = "auto_top3_low_confidence"
    return labels, method


# -----------------------------------------------------------------------
# Φόρτωση των εκκρεμών (μη επιθεωρημένων ακόμη) καταγραφών
# -----------------------------------------------------------------------
def _load_pending():
    if not os.path.isfile(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# -----------------------------------------------------------------------
# Σύνολο των κειμένων διλημμάτων που έχουν ήδη επιθεωρηθεί σε
# προηγούμενη εκτέλεση — χρησιμοποιείται για τον εντοπισμό ΑΚΡΙΒΩΣ
# πανομοιότυπων (όχι απλώς παρόμοιων) διπλότυπων υποβολών
# -----------------------------------------------------------------------
def _load_reviewed_dilemmas():
    if not os.path.isfile(REVIEWED_PATH):
        return set()
    with open(REVIEWED_PATH, "r", encoding="utf-8") as f:
        return {json.loads(line)["dilemma"] for line in f if line.strip()}


# -----------------------------------------------------------------------
# Ξαναγράφει το αρχείο εκκρεμών, αφαιρώντας τις ήδη επιθεωρημένες
# εγγραφές (index-based, με τη σειρά που επιστράφηκαν από _load_pending)
# -----------------------------------------------------------------------
def _rewrite_pending(remaining_entries):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for entry in remaining_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------
# Καταγράφει μια ήδη επιθεωρημένη εγγραφή (με τις τελικές, ανθρώπινα
# επιβεβαιωμένες ετικέτες) στο αρχείο ιστορικού επιθεωρήσεων
# -----------------------------------------------------------------------
def _append_reviewed(entry, final_labels, added_to_training, method):
    entry = dict(entry)
    entry["reviewed"] = True
    entry["final_labels"] = final_labels
    entry["added_to_training"] = added_to_training
    # "Σημείωση" καταγωγής: πώς προέκυψαν αυτές οι ετικέτες — χειροκίνητα
    # πληκτρολογημένες από τον χρήστη, ή αυτόματα βάσει σκορ (και σε ποιο
    # επίπεδο εμπιστοσύνης) — ώστε να ξεχωρίζουν καθαρά στο ιστορικό.
    entry["labeling_method"] = method
    with open(REVIEWED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------
# Προσθήκη μίας εγκεκριμένης εγγραφής στο model/training_data.py, στο
# ίδιο σημείο εισαγωγής που χρησιμοποιείται και χειροκίνητα
# -----------------------------------------------------------------------
def _append_to_training_data(text, labels):
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
    labels_str = ", ".join(f'"{l}"' for l in labels)
    new_line = f'    ("{escaped_text}", [{labels_str}]),\n'

    if INSERT_MARKER not in content:
        print("  [!] Δεν βρέθηκε το σημείο εισαγωγής στο training_data.py — προσθήκη παραλείφθηκε.")
        return False

    content = content.replace(INSERT_MARKER, new_line + "\n" + INSERT_MARKER, 1)
    with open(TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return True


# -----------------------------------------------------------------------
# Κύρια διαδραστική ροή
# -----------------------------------------------------------------------
def main():
    pending = _load_pending()
    if not pending:
        print("Δεν υπάρχουν καταγεγραμμένες υποβολές προς επιθεώρηση (data/usage_log.jsonl).")
        return

    # ---------------------------------------------------------------
    # Αυτόματη αφαίρεση ΑΚΡΙΒΩΣ πανομοιότυπων διλημμάτων — μόνο όταν το
    # κείμενο ταιριάζει χαρακτήρα-προς-χαρακτήρα με κάτι που είτε (α)
    # έχει ήδη επιθεωρηθεί σε προηγούμενη εκτέλεση, είτε (β) εμφανίζεται
    # ξανά μέσα στην ίδια αυτή παρτίδα. Καμία θολή/κατά προσέγγιση
    # αντιστοίχιση — αν διαφέρει έστω και ένας χαρακτήρας, θεωρείται
    # διαφορετικό δίλημμα και εμφανίζεται κανονικά.
    # ---------------------------------------------------------------
    already_reviewed = _load_reviewed_dilemmas()
    seen_this_batch = set()
    deduplicated = []
    duplicate_count = 0
    for entry in pending:
        text = entry["dilemma"]
        if text in already_reviewed or text in seen_this_batch:
            duplicate_count += 1
            continue
        seen_this_batch.add(text)
        deduplicated.append(entry)

    if duplicate_count:
        print(f"Αφαιρέθηκαν αυτόματα {duplicate_count} ακριβώς διπλότυπα διλήμματα (ήδη επιθεωρημένα ή επαναλαμβανόμενα).")
        _rewrite_pending(deduplicated)  # ώστε τα διπλότυπα να μη γράφονται ξανά στο αρχείο εκκρεμών

    pending = deduplicated
    if not pending:
        print("Δεν απομένουν νέες (μη διπλότυπες) υποβολές προς επιθεώρηση.")
        return

    print(f"Βρέθηκαν {len(pending)} καταγεγραμμένες υποβολές.")
    print(f"Έγκυρες ετικέτες: {', '.join(CATEGORIES)}")
    print("Σε κάθε ερώτηση:")
    print(f"  Enter          -> αυτόματη ετικέτηση βάσει σκορ: κορυφαία κατηγορία αν >= {AUTO_LABEL_HIGH_CONFIDENCE:.0%},")
    print(f"                    δύο κορυφαίες αν >= {AUTO_LABEL_MEDIUM_CONFIDENCE:.0%}, αλλιώς τρεις κορυφαίες")
    print("  ετικέτες,ετικέτες -> χειροκίνητη επιλογή (παρακάμπτει την αυτόματη πρόταση)")
    print("  n              -> ρητά ουδέτερο (καμία ηθική διάσταση)")
    print("  d              -> απόρριψη (δοκιμές, spam, ακατάληπτο ή αμφίσημο κείμενο)")
    print("  s              -> παράλειψη (θα ξαναεμφανιστεί την επόμενη φορά)")
    print("  q              -> έξοδος\n")

    remaining = list(pending)
    added_count = 0
    dropped_count = 0

    for entry in pending:
        print("-" * 70)
        print(f"Δίλημμα: {entry['dilemma']}")
        scores = entry.get("scores", {})
        sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])
        print("Βαθμολογίες μοντέλου:")
        for cat, score in sorted_scores:
            print(f"    {cat}: {score}")

        answer = input("Ετικέτες (ή Enter/n/d/s/q): ").strip()

        if answer.lower() == "q":
            print("\nΈξοδος. Οι μη επιθεωρημένες εγγραφές παραμένουν στο αρχείο για επόμενη φορά.")
            break

        if answer.lower() == "s":
            continue  # παραμένει στο pending log, δεν αφαιρείται

        if answer.lower() == "d":
            # Απόρριψη: δεν μπαίνει στην εκπαίδευση, αλλά καταγράφεται στο
            # ιστορικό ως "discarded". Επειδή πλέον θεωρείται επιθεωρημένη,
            # ένα ακριβώς ίδιο κείμενο που θα υποβληθεί ξανά αργότερα θα
            # αφαιρεθεί αυτόματα ως διπλότυπο και δεν θα ξαναεμφανιστεί.
            _append_reviewed(entry, None, False, "discarded")
            remaining.remove(entry)
            dropped_count += 1
            print("  -> Απορρίφθηκε (δεν προστέθηκε στην εκπαίδευση).")
            continue

        if answer.lower() == "n":
            final_labels, method = [], "manual_neutral"
        elif answer == "":
            final_labels, method = _auto_label_from_scores(sorted_scores)
            print(f"  -> Αυτόματη ετικέτηση ({method}): {final_labels}")
        else:
            final_labels = [l.strip() for l in answer.split(",") if l.strip()]
            method = "manual"

        invalid = [l for l in final_labels if l not in CATEGORIES]
        if invalid:
            print(f"  [!] Άγνωστες ετικέτες: {invalid} — η εγγραφή παραλείφθηκε, δοκίμασε ξανά αργότερα.")
            continue

        added = _append_to_training_data(entry["dilemma"], final_labels)
        if added:
            added_count += 1
            print(f"  -> Προστέθηκε στο training_data.py με ετικέτες {final_labels}.")

        _append_reviewed(entry, final_labels, added, method)
        remaining.remove(entry)

    _rewrite_pending(remaining)
    print(f"\nΣύνολο: {added_count} νέα παραδείγματα προστέθηκαν στο model/training_data.py"
          f" | {dropped_count} απορρίφθηκαν | {len(remaining)} εκκρεμούν.")
    if added_count:
        print("Τρέξε ξανά python -m model.train_classifier για να τα ενσωματώσεις στο μοντέλο.")


if __name__ == "__main__":
    main()
