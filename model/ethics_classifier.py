"""
TensorFlow/Keras multi-label text classifier for the Ethical prototype.

Given a short free-text description of a business dilemma — in English or
Greek — the model predicts a probability, between 0 and 1, for each of
five ethical dimensions (see training_data.CATEGORIES). Those
probabilities drive the retrieval step in engine/advisor.py: dimensions
above a threshold pull in matching principles, regulations and actions
from the knowledge base.

Architecture (and why it looks the way it does):
- Multilingual sentence embeddings come from `sentence-transformers`
  (specifically a multilingual MiniLM model covering 50+ languages,
  including Greek), which maps semantically similar sentences to nearby
  points in the same embedding space regardless of language. This is what
  makes typing a dilemma in Greek meaningfully supported, rather than just
  accepted without being understood.
- An earlier version of this module used a TensorFlow Hub encoder (LaBSE)
  loaded directly inside the Keras graph. That was switched out because
  LaBSE's TF Hub preprocessor depends on the `tensorflow-text` package,
  which does not ship official Windows wheels (Linux/macOS only) — a real
  blocker for anyone developing on Windows, as this thesis project is.
  `sentence-transformers` (PyTorch-backed) installs cleanly on Windows,
  macOS, and Linux alike.
- The classifier you actually train and evaluate — the thing with weights,
  a training loop, a saved model, and metrics worth reporting in the
  thesis — is the small Dense/Keras head trained on top of those frozen
  embeddings (see build_classifier_head() and train_classifier.py). The
  embedding step is a fixed multilingual feature extractor, not something
  this project trains.
- It is a *routing* classifier, not the source of ethical or legal advice.
  The actual guidance text always comes from the curated, bilingual
  knowledge base, which keeps the system auditable: every sentence shown
  to the user can be traced back to a source in data/knowledge_base.json.
- If no trained model is found on disk, EthicsClassifier falls back to a
  transparent keyword-matching heuristic (with both English and Greek
  keywords), so the web app still runs before
  `python -m model.train_classifier` has been executed.
- The embedding model is downloaded from Hugging Face the first time it's
  used (roughly 470MB) and cached locally afterward, so an internet
  connection is needed once, but not for every run.
"""

# -----------------------------------------------------------------------
# Εισαγωγές και ρυθμίσεις διαδρομών/μοντέλου
# -----------------------------------------------------------------------
import os

from model.training_data import CATEGORIES

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_model")
# Το Keras 3 (που έρχεται πλέον μαζί με το TensorFlow) απαιτεί ρητή
# κατάληξη αρχείου κατά την αποθήκευση/φόρτωση μοντέλων — δεν δέχεται
# πια φάκελο χωρίς κατάληξη (παλιά μορφή SavedModel). Χρησιμοποιούμε τη
# εγγενή μορφή .keras (ένα ενιαίο αρχείο, προτεινόμενη μορφή).
CLASSIFIER_HEAD_PATH = os.path.join(MODEL_DIR, "ethics_classifier_head.keras")

# Πολυγλωσσικό μοντέλο (50+ γλώσσες, συμπεριλαμβανομένων των Ελληνικών),
# καλή ισορροπία μεγέθους/ποιότητας για ένα πρωτότυπο διατριβής που
# εκπαιδεύεται σε φορητό υπολογιστή.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedder = None  # lazy singleton: loaded at most once per process


# -----------------------------------------------------------------------
# Φόρτωση (μία φορά ανά διεργασία) του πολυγλωσσικού μοντέλου ενσωμάτωσης
# -----------------------------------------------------------------------
def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


# -----------------------------------------------------------------------
# Μετατροπή λίστας κειμένων σε πίνακα πολυγλωσσικών ενσωματώσεων (embeddings)
# -----------------------------------------------------------------------
def embed_texts(texts):
    """Return an (N, embedding_dim) numpy array of multilingual sentence
    embeddings for the given list of strings."""
    embedder = _get_embedder()
    return embedder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


# -----------------------------------------------------------------------
# Εφεδρικές λέξεις-κλειδιά (fallback) σε Αγγλικά και Ελληνικά, για χρήση
# πριν την εκπαίδευση του μοντέλου ή σε περίπτωση σφάλματος
# -----------------------------------------------------------------------
_FALLBACK_KEYWORDS = {
    "transparency": [
        "explain", "disclose", "tell", "know", "understand", "why", "label", "byline", "black box",
        "εξηγ", "γνωστοποι", "ενημερ", "διαφάνει", "κατανο", "ετικέτ",
    ],
    "fairness": [
        "bias", "biased", "discriminat", "unfair", "fair", "gender", "age", "ethnic", "race", "postcode", "zip code",
        "μερολη", "διάκρι", "άδικ", "δίκαι", "φύλο", "ηλικία", "εθνικ", "φυλή", "ταχυδρομικό",
    ],
    "non_maleficence": [
        "harm", "danger", "risky", "unsafe", "hurt", "injur", "self-harm", "overheat", "loss", "wrong advice",
        "camera", "facial recognition", "monitor", "surveillance", "emotion recognition",
        "βλάβη", "κίνδυν", "επικίνδυν", "ανασφαλ", "τραυματ", "ζημιά", "λάθος συμβουλή",
        "παρακολούθηση", "κάμερ", "επιτήρηση", "συναίσθημα",
    ],
    "accountability": [
        "responsib", "accountab", "own", "sign off", "approve", "audit", "liable", "review", "governance",
        "υπεύθυν", "λογοδοσ", "εγκρίν", "έλεγχο", "διακυβέρνηση", "επισκόπηση",
    ],
    "privacy": [
        "personal data", "confidential", "gdpr", "privacy", "customer data", "patient", "record", "upload", "database",
        "facial recognition", "biometric", "camera", "employee monitoring",
        "προσωπικά δεδομένα", "εμπιστευτικ", "ιδιωτικότητα", "δεδομένα πελατ", "ασθεν", "αρχείο", "βάση δεδομένων",
    ],
}


# -----------------------------------------------------------------------
# Υπολογισμός βαθμολογίας ανά κατηγορία, βάσει πλήθους λέξεων-κλειδιών
# -----------------------------------------------------------------------
def _keyword_fallback_scores(text: str) -> dict:
    lowered = text.lower()
    scores = {}
    for category, keywords in _FALLBACK_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        scores[category] = min(1.0, hits * 0.35)
    return scores


# -----------------------------------------------------------------------
# Κύρια κλάση ταξινομητή: φορτώνει το εκπαιδευμένο μοντέλο αν υπάρχει,
# διαφορετικά λειτουργεί μέσω του εφεδρικού μηχανισμού λέξεων-κλειδιών
# -----------------------------------------------------------------------
class EthicsClassifier:
    """Loads (or lazily builds a fallback for) the ethics-dimension model."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.using_fallback = True
        self._try_load()

    def _try_load(self):
        head_path = os.path.join(self.model_dir, "ethics_classifier_head.keras")
        if os.path.isfile(head_path):
            try:
                import tensorflow as tf  # imported lazily so the app can run without TF installed
                self.model = tf.keras.models.load_model(head_path)
                self.using_fallback = False
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[EthicsClassifier] Could not load trained model, using fallback: {exc}")
                self.model = None
                self.using_fallback = True
        else:
            self.using_fallback = True

    def predict(self, text: str) -> dict:
        """Return {category: probability_0_to_1} for the given dilemma text
        (English or Greek)."""
        text = (text or "").strip()
        if not text:
            return {cat: 0.0 for cat in CATEGORIES}

        # Προτεραιότητα στο εκπαιδευμένο μοντέλο· σε περίπτωση αποτυχίας,
        # χρησιμοποιείται αθόρυβα ο εφεδρικός μηχανισμός λέξεων-κλειδιών
        if self.model is not None:
            try:
                embedding = embed_texts([text])
                probs = self.model.predict(embedding, verbose=0)[0]
                return {cat: float(probs[i]) for i, cat in enumerate(CATEGORIES)}
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[EthicsClassifier] Embedding/prediction failed, using fallback: {exc}")

        return {cat: _keyword_fallback_scores(text).get(cat, 0.0) for cat in CATEGORIES}


# -----------------------------------------------------------------------
# Δημιουργία του εκπαιδεύσιμου Keras μοντέλου (κεφαλή ταξινόμησης) που
# δέχεται ως είσοδο έτοιμες ενσωματώσεις (embeddings), όχι ακατέργαστο κείμενο
# -----------------------------------------------------------------------
def build_classifier_head(embedding_dim: int):
    """Build a fresh, untrained Keras model: a small classifier head that
    takes precomputed sentence embeddings as input (not raw text).

    Deliberately smaller and more regularized than earlier versions. With
    repeated data-addition cycles we kept seeing the same pattern: strong
    cross-validation scores but much weaker performance on the held-out
    generalization set (e.g. privacy: 0.82 CV vs 0.46 held-out) — a classic
    sign the head has enough capacity to memorise the specific phrasing of
    the (still small, ~200-example) training set rather than learning a
    genuinely general boundary. Shrinking the layers, raising dropout, and
    adding L2 weight decay all push in the same direction: make it harder
    for the head to fit narrow, batch-specific patterns.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers

    l2 = regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=(embedding_dim,), name="embedding")
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CATEGORIES), activation="sigmoid", name="ethics_dimensions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ethical_classifier_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model
