"""
Αξιολόγηση της γενίκευσης του ταξινομητή ηθικών διαστάσεων.

Δύο συμπληρωματικοί τρόποι μέτρησης:

  1. k-fold διασταυρούμενη επικύρωση (cross-validation) πάνω στο ίδιο
     σύνολο δεδομένων του model/training_data.py: εκπαιδεύει σε k-1
     τμήματα, αξιολογεί στο τμήμα που έμεινε έξω, k φορές, και βγάζει
     μέσο όρο — δίνει μια αμερόληπτη εκτίμηση της απόδοσης πάνω σε
     δεδομένα με το ίδιο γενικό "ύφος" διατύπωσης.

  2. Ένα εντελώς ξεχωριστό, χειρόγραφο σύνολο δοκιμής (πιο κάτω στο
     αρχείο), με διατυπώσεις που ΔΕΝ εμφανίζονται πουθενά στο
     training_data.py — αξιολογεί το ήδη αποθηκευμένο μοντέλο παραγωγής
     (saved_model/ethics_classifier_head) πάνω σε πραγματικά άγνωστη
     διατύπωση. Αυτό είναι η πιο αυστηρή δοκιμή γενίκευσης, καθώς καμία
     από αυτές τις προτάσεις δεν έχει δει ποτέ το μοντέλο, ούτε καν σε
     διαφορετική μορφή.

Εκτέλεση (από τη ρίζα του project, με ενεργό το virtual environment):
    python -m model.evaluate_classifier

Απαιτεί scikit-learn (προστέθηκε στο requirements.txt) για τις μετρικές.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών και εσωτερικών ενοτήτων
# -----------------------------------------------------------------------
import os

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support

from model.ethics_classifier import build_classifier_head, embed_texts, MODEL_DIR
from model.training_data import get_texts_and_labels, CATEGORIES
from model.training_data import TRAINING_EXAMPLES as _TRAINING_EXAMPLES


def get_training_examples():
    return _TRAINING_EXAMPLES

SEED = 42
N_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 8
# Ίδιο κατώφλι ανίχνευσης με το engine/advisor.py, για συνέπεια με το
# πραγματικό όριο που καθορίζει τι εμφανίζεται στην εφαρμογή.
THRESHOLD = 0.45


# -----------------------------------------------------------------------
# Ξεχωριστό, χειρόγραφο σύνολο δοκιμής γενίκευσης. ΚΑΜΙΑ από αυτές τις
# προτάσεις δεν υπάρχει στο model/training_data.py — γράφτηκαν σκόπιμα
# με διαφορετική διατύπωση από τα παραδείγματα εκπαίδευσης, ώστε να
# ελέγχουν αν το μοντέλο κατανοεί το νόημα ή απλώς απομνημονεύει λέξεις.
# Κάθε πλειάδα: (κείμενο διλήμματος, αναμενόμενες κατηγορίες [gold labels]).
# -----------------------------------------------------------------------
HELD_OUT_EXAMPLES = [
    ("Our company wants to install smart cameras with emotion recognition to monitor staff wellbeing.",
     ["non_maleficence", "privacy"]),
    ("Should we let an algorithm auto-reject loan applications without anyone reviewing the outcome?",
     ["accountability"]),
    ("We're thinking of scraping social media profiles to enrich our candidate screening tool.",
     ["privacy", "fairness"]),
    ("Is it okay that our pricing algorithm charges different amounts based on browser history?",
     ["fairness", "transparency"]),
    ("A customer asked why our chatbot denied their refund and we couldn't give a real answer.",
     ["transparency", "accountability"]),
    ("We want to keep using a hiring algorithm even though nobody checked it for bias in over a year.",
     ["fairness", "accountability"]),
    ("Θέλουμε να χρησιμοποιήσουμε τεχνητή νοημοσύνη για να αξιολογούμε αυτόματα τις άδειες ασθενείας των εργαζομένων μας.",
     ["privacy", "non_maleficence"]),
    ("Μπορούμε να χρησιμοποιήσουμε δεδομένα πλοήγησης πελατών χωρίς να τους ενημερώσουμε ρητά;",
     ["privacy", "transparency"]),
    ("Ένας πελάτης ρώτησε γιατί το σύστημα απέρριψε την αίτησή του και δεν έχουμε σαφή απάντηση.",
     ["transparency", "accountability"]),
    ("What's a good way to organize our quarterly budget review?", []),
    ("Can you suggest icebreakers for a team lunch?", []),

    # --- 14 επιπλέον, για πιο στατιστικά αξιόπιστη ένδειξη γενίκευσης
    # (σύνολο 25 αντί για 11) — νέα, διαφορετικά σενάρια/τομείς ---
    ("A ride-sharing app's AI sets higher prices in neighborhoods with lower average incomes.",
     ["fairness"]),
    ("Our AI-powered security system flags people loitering near the building, but seems to flag delivery workers of certain ethnicities more often.",
     ["fairness", "non_maleficence"]),
    ("Should a subscription service use AI to make it deliberately harder to find the cancel button?",
     ["transparency", "non_maleficence"]),
    ("Nobody at our company can say who approved letting the AI assistant access customer payment details.",
     ["accountability", "privacy"]),
    ("Our AI writing assistant sometimes generates plausible-sounding but false citations in reports.",
     ["non_maleficence", "transparency"]),
    ("Should an AI system that denies insurance claims explain its reasoning to the policyholder?",
     ["transparency", "accountability"]),
    ("We want to sell aggregated, anonymized location data collected through our app to advertisers.",
     ["privacy"]),
    ("An AI-powered toy for children records audio and sends it to the manufacturer's servers for 'quality improvement'.",
     ["privacy", "transparency"]),
    ("Θέλουμε να χρησιμοποιήσουμε ΑΙ για να κατατάξουμε τους υποψήφιους ενοικιαστές, αλλά δεν έχουμε ελέγξει αν κάνει διακρίσεις.",
     ["fairness"]),
    ("Μπορούμε να χρησιμοποιήσουμε chatbot ΑΙ για ψυχολογική υποστήριξη χωρίς εποπτεία από ειδικό;",
     ["non_maleficence", "accountability"]),
    ("What time zone should we use for our internal team meetings?", []),
    ("Can you help me write a thank-you note to a colleague?", []),
    ("Our self-checkout AI occasionally misidentifies produce, overcharging customers by a small amount each time.",
     ["non_maleficence", "fairness"]),
    ("Should we let an AI-powered background-check tool automatically reject candidates with any criminal record, regardless of context?",
     ["fairness", "non_maleficence"]),

    # =====================================================================
    # ΔΙΕΥΡΥΝΣΗ ΣΕ ~60 (ΠΑΓΩΜΕΝΟ ΣΥΝΟΛΟ): στόχος ~15 θετικά ανά κατηγορία,
    # ώστε μια διαφορά ενός παραδείγματος να μην αλλάζει το F1 κατά 0,10.
    # Τα θέματα «διασταυρώνονται» σκόπιμα με ετικέτες που το μοντέλο δεν
    # συνδέει συνήθως με αυτά (προσλήψεις × privacy/transparency/
    # accountability, παιδιά × fairness/accountability/privacy, υγεία ×
    # fairness/accountability/privacy), ώστε το σύνολο να μετρά ρητά τις
    # συντομεύσεις θέματος (topic shortcuts). ΜΗΝ το αλλάξεις από εδώ και
    # πέρα, αλλιώς τα αποτελέσματα παύουν να είναι συγκρίσιμα μεταξύ τους.
    # =====================================================================

    # --- transparency ---
    ("A job board uses AI to rewrite applicants' cover letters before employers see them, but neither side is told.",
     ["transparency"]),
    ("Our children's reading app mixes human-written and AI-generated stories without marking which is which.",
     ["transparency"]),
    ("Η κλινική χρησιμοποιεί ΑΙ για να συντάσσει τις απαντήσεις στα email των ασθενών, χωρίς να αναφέρεται πουθενά ότι γράφτηκαν από ΑΙ.",
     ["transparency"]),
    ("Our real-estate listings use AI-generated photos of furnished rooms, with no note that the furniture isn't real.",
     ["transparency"]),
    ("We never tell restaurant owners that the review summary shown on their page was written by an AI.",
     ["transparency"]),

    # --- fairness ---
    ("Our AI tutor gives more detailed feedback to students who write in standard English than to those who use dialect.",
     ["fairness"]),
    ("A hospital's AI scheduling system gives earlier appointments to patients with private insurance.",
     ["fairness"]),
    ("Το σύστημα ΑΙ κατανομής βαρδιών δίνει σταθερά τις χειρότερες βάρδιες σε εργαζόμενους που δεν μιλούν καλά ελληνικά.",
     ["fairness"]),
    ("A dating app's AI shows users from certain ethnic backgrounds to far fewer people.",
     ["fairness"]),
    ("Our AI-driven delivery app offers slower delivery windows to rural addresses at the same price.",
     ["fairness"]),

    # --- non_maleficence ---
    ("Our AI job-matching tool sometimes recommends roles with unsafe working conditions to desperate applicants.",
     ["non_maleficence"]),
    ("A smart-home AI occasionally turns the heating off overnight to save energy, even in homes with elderly residents.",
     ["non_maleficence"]),
    ("Η εφαρμογή ΑΙ για δρομείς προτείνει σε αρχάριους προγράμματα προπόνησης τόσο απαιτητικά που υπάρχει κίνδυνος τραυματισμού.",
     ["non_maleficence"]),
    ("Our AI recipe assistant sometimes suggests ingredients that users listed as allergies.",
     ["non_maleficence"]),
    ("An AI parental-control app blocks a teenager's access to crisis helpline websites by mistake.",
     ["non_maleficence"]),

    # --- accountability ---
    ("Our AI grading system has changed hundreds of students' marks, and the school can't say who configured its rubric.",
     ["accountability"]),
    ("When the AI hiring tool rejected an internal candidate by mistake, HR, IT and the vendor each said it wasn't their call.",
     ["accountability"]),
    ("Η πλατφόρμα τηλεϊατρικής άλλαξε τον αλγόριθμο διαλογής ασθενών, αλλά κανείς δεν ξέρει ποιος το ενέκρινε.",
     ["accountability"]),
    ("Our AI chatbot promised a customer a refund the company never offered, and no team will take ownership of honouring it.",
     ["accountability"]),
    ("An autonomous forklift in our warehouse damaged goods twice, and the vendor and our operations team blame each other.",
     ["accountability"]),
    ("Κανείς δεν έχει οριστεί υπεύθυνος για να ενημερώνει το σύστημα ΑΙ όταν αλλάζει η νομοθεσία.",
     ["accountability"]),

    # --- privacy ---
    ("Our internal AI search tool lets any employee look up other employees' home addresses from HR records.",
     ["privacy"]),
    ("A school's AI attendance system runs facial recognition on pupils every morning at the gate.",
     ["privacy"]),
    ("A gym's AI app sells members' workout and body-weight data to a supplement company.",
     ["privacy"]),
    ("Η εφαρμογή ΑΙ του σούπερ μάρκετ συνδέει τις αγορές φαρμάκων των πελατών με το προφίλ τους για στοχευμένες διαφημίσεις.",
     ["privacy"]),
    ("Our car-rental app's AI keeps tracking the driver's location for weeks after the car is returned.",
     ["privacy"]),
    ("Ο εργοδότης θέλει να ζητά από τους υποψηφίους πρόσβαση στο ιστορικό τοποθεσίας του κινητού τους κατά τη συνέντευξη.",
     ["privacy"]),

    # --- δύο ετικέτες ---
    ("Our AI loan chatbot now collects applicants' social media handles, and nobody knows who approved adding that question.",
     ["privacy", "accountability"]),
    ("A children's app's AI ranks kids' drawings in a public gallery, consistently placing those from wealthier schools higher.",
     ["fairness", "non_maleficence"]),
    ("An AI symptom checker gives less urgent advice to older users, and the company won't explain the logic behind it.",
     ["fairness", "transparency"]),
    ("Η εφαρμογή ΑΙ ενοικιάσεων ζητά από τους ενοικιαστές ανάλυση τραπεζικών κινήσεων, χωρίς να λέει πώς χρησιμοποιείται στη βαθμολογία τους.",
     ["privacy", "transparency"]),
    ("A delivery company's AI docks drivers' pay for route deviations caused by road closures, and managers say they can't override it.",
     ["fairness", "accountability"]),
    ("Our AI tax-filing assistant files returns with errors that lead to fines, and the terms say the user alone is responsible.",
     ["non_maleficence", "accountability"]),

    # --- ουδέτερα ---
    ("Which spreadsheet template works best for tracking the team's holiday requests?", []),
    ("Can you suggest a name for our new internal newsletter?", []),
    ("Ποια είναι μια καλή δομή για την εβδομαδιαία συνάντηση του τμήματος;", []),
    ("Μπορείς να μου προτείνεις δραστηριότητες για την εορταστική εκδήλωση της εταιρείας;", []),
]


# -----------------------------------------------------------------------
# Υπολογισμός και εκτύπωση μετρικών πολυ-ετικέτας ταξινόμησης: precision/
# recall/F1 ανά κατηγορία, micro/macro μέσοι όροι, subset accuracy
# (ακριβής αντιστοιχία όλων των ετικετών) και Hamming loss
# -----------------------------------------------------------------------
def print_metrics(y_true, y_pred, label_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    subset_acc = accuracy_score(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)

    print(f"{'Κατηγορία':<18}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    for i, name in enumerate(label_names):
        print(f"{name:<18}{precision[i]:>12.2f}{recall[i]:>12.2f}{f1[i]:>12.2f}")
    print("-" * 54)
    print(f"{'Micro-average':<18}{micro_p:>12.2f}{micro_r:>12.2f}{micro_f1:>12.2f}")
    print(f"{'Macro-average':<18}{macro_p:>12.2f}{macro_r:>12.2f}{macro_f1:>12.2f}")
    print(f"\nSubset accuracy (ακριβής αντιστοιχία όλων των ετικετών): {subset_acc:.2f}")
    print(f"Hamming loss (όσο χαμηλότερο, τόσο καλύτερο): {h_loss:.3f}")


# -----------------------------------------------------------------------
# Μέρος 1: k-fold διασταυρούμενη επικύρωση πάνω στο training_data.py.
# Σε κάθε πτυχή (fold) εκπαιδεύεται ΝΕΟ μοντέλο από την αρχή μόνο στα
# δεδομένα εκπαίδευσης εκείνης της πτυχής, και αξιολογείται στα
# δεδομένα δοκιμής που έμειναν έξω — άρα κάθε πρόβλεψη αφορά παράδειγμα
# που το συγκεκριμένο μοντέλο δεν είδε ποτέ κατά την εκπαίδευσή του.
# -----------------------------------------------------------------------
def cross_validate():
    print("=" * 60)
    print(f"ΜΕΡΟΣ 1: {N_FOLDS}-fold διασταυρούμενη επικύρωση")
    print("=" * 60)

    texts, labels = get_texts_and_labels()
    y = np.array(labels, dtype="float32")

    print(f"Σύνολο παραδειγμάτων: {len(texts)}")
    print("Υπολογισμός ενσωματώσεων (μία φορά, επαναχρησιμοποιείται σε όλες τις πτυχές)...")
    embeddings = embed_texts(texts)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_pred, all_prob = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings), 1):
        # Σταθερό seed πριν από κάθε πτυχή, ώστε η αρχικοποίηση βαρών του
        # μοντέλου (όχι μόνο ο διαχωρισμός σε πτυχές) να είναι επίσης
        # αναπαραγώγιμη — τα ίδια δεδομένα εισόδου δίνουν πάντα τα ίδια
        # νούμερα εξόδου σε κάθε επανάληψη του σεναρίου.
        import tensorflow as tf
        tf.random.set_seed(SEED + fold)
        np.random.seed(SEED + fold)

        model = build_classifier_head(embedding_dim=embeddings.shape[1])
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        model.fit(
            embeddings[train_idx], y[train_idx],
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.15, callbacks=[early_stopping], verbose=0,
        )
        probs = model.predict(embeddings[test_idx], verbose=0)
        preds = (probs >= THRESHOLD).astype(int)

        all_true.append(y[test_idx])
        all_pred.append(preds)
        all_prob.append(probs)
        print(f"  Πτυχή {fold}/{N_FOLDS}: {len(test_idx)} παραδείγματα δοκιμής")

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    y_prob = np.vstack(all_prob)
    print()
    print_metrics(y_true, y_pred, CATEGORIES)
    optimal_thresholds = find_optimal_thresholds(y_true, y_prob, CATEGORIES)
    return optimal_thresholds


# -----------------------------------------------------------------------
# Σάρωση υποψήφιων κατωφλίων (0.05 έως 0.95) ξεχωριστά για κάθε
# κατηγορία, πάνω στις ήδη υπολογισμένες πιθανότητες του cross-validation
# (όχι στα ήδη κατωφλιωμένα 0/1), και επιλογή αυτού που μεγιστοποιεί το
# F1 της συγκεκριμένης κατηγορίας. Ενιαίο κατώφλι για όλες τις κατηγορίες
# είναι βολικό αλλά όχι βέλτιστο, όταν κάθε κατηγορία έχει διαφορετική
# "δυσκολία" ανίχνευσης — όπως δείχνουν ήδη τα per-κατηγορία F1 παραπάνω.
# -----------------------------------------------------------------------
def find_optimal_thresholds(y_true, y_prob, label_names):
    print("\n" + "=" * 60)
    print("Βέλτιστο κατώφλι ανά κατηγορία (μεγιστοποίηση F1)")
    print("=" * 60)

    candidate_thresholds = np.arange(0.05, 1.0, 0.05)
    optimal = {}

    print(f"{'Κατηγορία':<18}{'Τρέχον F1':>12}{'Βέλτιστο κατώφλι':>20}{'Νέο F1':>12}")
    for i, name in enumerate(label_names):
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]

        current_pred = (col_prob >= THRESHOLD).astype(int)
        _, _, current_f1, _ = precision_recall_fscore_support(
            col_true, current_pred, average="binary", zero_division=0
        )

        best_t, best_f1 = THRESHOLD, current_f1
        for t in candidate_thresholds:
            pred = (col_prob >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                col_true, pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_t, best_f1 = t, f1

        optimal[name] = round(float(best_t), 2)
        print(f"{name:<18}{current_f1:>12.2f}{best_t:>20.2f}{best_f1:>12.2f}")

    print("\nΓια να τα εφαρμόσεις στην εφαρμογή, αντικατέστησε το DETECTION_THRESHOLD")
    print("στο engine/advisor.py με ένα dict ανά κατηγορία:")
    print(f"\nDETECTION_THRESHOLDS = {optimal}")
    return optimal


# -----------------------------------------------------------------------
# Μέρος 2: αξιολόγηση του ήδη εκπαιδευμένου μοντέλου παραγωγής πάνω στο
# χειρόγραφο σύνολο δοκιμής γενίκευσης (HELD_OUT_EXAMPLES). Απαιτεί να
# έχει ήδη τρέξει το `python -m model.train_classifier`.
# -----------------------------------------------------------------------
def evaluate_held_out_set():
    print("\n" + "=" * 60)
    print(f"ΜΕΡΟΣ 2: Χειρόγραφο σύνολο δοκιμής γενίκευσης ({len(HELD_OUT_EXAMPLES)} παραδείγματα, μοντέλο παραγωγής)")
    print("=" * 60)

    # Έλεγχος μόλυνσης: αν κάποιο held-out κείμενο έχει μπει (π.χ. μέσω
    # review_log ή επέκτασης LLM) στο σύνολο εκπαίδευσης, η μέτρηση
    # γενίκευσης δεν είναι πλέον έγκυρη για αυτό το παράδειγμα.
    train_texts = {t.strip().lower() for t, _ in get_training_examples()}
    leaked = [t for t, _ in HELD_OUT_EXAMPLES if t.strip().lower() in train_texts]
    if leaked:
        print(f"[!] ΠΡΟΣΟΧΗ: {len(leaked)} held-out παραδείγματα υπάρχουν και στο training_data.py —")
        print("    αφαίρεσέ τα από το training_data.py για έγκυρη μέτρηση γενίκευσης:")
        for t in leaked:
            print(f"    - {t[:80]}")

    head_path = os.path.join(MODEL_DIR, "ethics_classifier_head.keras")
    if not os.path.isfile(head_path):
        print("Δεν βρέθηκε εκπαιδευμένο μοντέλο.")
        print("Τρέξε πρώτα: python -m model.train_classifier")
        return

    import tensorflow as tf
    model = tf.keras.models.load_model(head_path)

    texts = [t for t, _ in HELD_OUT_EXAMPLES]
    gold_labels = [labels for _, labels in HELD_OUT_EXAMPLES]
    y_true = np.array([[1 if cat in labels else 0 for cat in CATEGORIES] for labels in gold_labels])

    embeddings = embed_texts(texts)
    probs = model.predict(embeddings, verbose=0)
    preds = (probs >= THRESHOLD).astype(int)

    print(f"\n{'':<3}{'Δίλημμα':<62}{'Αναμενόμενο':<28}{'Πρόβλεψη':<28}")
    for text, gold, pred_row in zip(texts, gold_labels, preds):
        predicted = [CATEGORIES[i] for i, v in enumerate(pred_row) if v == 1]
        match = "OK " if set(predicted) == set(gold) else "ΔΙΑΦ"
        print(f"{match:<3}{text[:59]:<62}{str(gold):<28}{str(predicted):<28}")

    print()
    print_metrics(y_true, preds, CATEGORIES)


if __name__ == "__main__":
    cross_validate()
    evaluate_held_out_set()
