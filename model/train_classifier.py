"""
Train and save the Ethical TensorFlow ethics-dimension classifier head.

Usage (from the project root, with the virtual environment active):

    python -m model.train_classifier

This will:
  1. Load the synthetic labelled dataset from model/training_data.py
     (English and Greek examples)
  2. Compute a multilingual sentence embedding for each example using
     sentence-transformers (downloads the embedding model on first run,
     roughly 470MB, then caches it locally)
  3. Train a small Keras classifier head on those embeddings
  4. Save the trained head to saved_model/ethics_classifier_head.keras so
     app.py can load it via EthicsClassifier() at request time.

Note: only step 2's model download needs an internet connection, and only
the first time. Training itself (step 3) runs in well under a minute on a
laptop CPU, since the head is small and the embeddings are precomputed.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών και εσωτερικών ενοτήτων
# -----------------------------------------------------------------------
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.ethics_classifier import build_classifier_head, embed_texts, MODEL_DIR
from model.training_data import get_texts_and_labels, CATEGORIES

# -----------------------------------------------------------------------
# Υπερπαράμετροι εκπαίδευσης
# -----------------------------------------------------------------------
EPOCHS = 60
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.15
SEED = 42


def main():
    # Σταθεροποίηση τυχαιότητας για αναπαραγωγίσιμα αποτελέσματα
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ---------------------------------------------------------------
    # Φόρτωση του συνόλου εκπαίδευσης και υπολογισμός των πολυγλωσσικών
    # ενσωματώσεων (embeddings) για κάθε παράδειγμα
    # ---------------------------------------------------------------
    texts, label_vectors = get_texts_and_labels()
    y = np.array(label_vectors, dtype="float32")

    print(f"Training examples: {len(texts)} | Categories: {CATEGORIES}")
    print("Computing multilingual sentence embeddings (downloads the embedding model on first run)...")
    x = embed_texts(texts)
    print(f"Embedding shape: {x.shape}")

    # ---------------------------------------------------------------
    # Δημιουργία και εκπαίδευση της κεφαλής ταξινόμησης (Keras)
    # ---------------------------------------------------------------
    # ΣΗΜΕΙΩΣΗ: εσκεμμένα ΔΕΝ χρησιμοποιείται το validation_split του
    # model.fit(). Το Keras, όταν του δοθεί validation_split, δεν παίρνει
    # τυχαίο δείγμα — απλώς κόβει το ΤΕΛΕΥΤΑΙΟ ποσοστό του πίνακα όπως
    # είναι ήδη διατεταγμένος. Επειδή το training_data.py είναι
    # οργανωμένο σε ομαδοποιημένα μπλοκ ανά κατηγορία/θέμα, αυτό θα
    # σήμαινε ότι το σύνολο επικύρωσης (και άρα το early stopping που
    # βασίζεται πάνω του) θα μπορούσε να καταλήξει συστηματικά
    # μεροληπτικό αντί για αντιπροσωπευτικό δείγμα. Κάνουμε τον
    # διαχωρισμό εμείς οι ίδιοι, με πραγματική (αλλά seeded, άρα
    # αναπαραγώγιμη) ανακάτεμα.
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=VALIDATION_SPLIT, random_state=SEED, shuffle=True
    )

    model = build_classifier_head(embedding_dim=x.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    # ---------------------------------------------------------------
    # Αποθήκευση του εκπαιδευμένου μοντέλου στον δίσκο
    # ---------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "ethics_classifier_head.keras")
    model.save(save_path)
    print(f"\nSaved trained classifier head to: {save_path}")

    # ---------------------------------------------------------------
    # Δοκιμαστικός έλεγχος (sanity check) σε παραδείγματα εκτός του
    # συνόλου εκπαίδευσης, συμπεριλαμβανομένου ενός ελληνικού
    # ---------------------------------------------------------------
    samples = [
        "Our AI hiring tool seems to reject older applicants more often and nobody checks why.",
        "Θέλουμε να βάλουμε κάμερες παρακολούθησης και αναγνώριση συναισθήματος στους υπαλλήλους μας.",
        "Is it fine to share client records with a third-party chatbot for a quick summary?",
        "What's a good agenda for our weekly stand-up?",
    ]
    sample_embeddings = embed_texts(samples)
    preds = model.predict(sample_embeddings, verbose=0)
    print("\nSanity check (includes a Greek-language example and an off-training-set phrasing):")
    for text, pred in zip(samples, preds):
        scored = sorted(zip(CATEGORIES, pred), key=lambda t: -t[1])
        top = ", ".join(f"{cat}={score:.2f}" for cat, score in scored)
        print(f"  '{text[:60]}...' -> {top}")


if __name__ == "__main__":
    main()
