"""
Semantic retrieval (a small RAG layer) over a bibliography of AI-ethics
literature, complementing the category-tagged "Relevant principles"
section (engine/advisor.py) with broader, similarity-based matches.

Unlike the curated principle/regulation/action entries — which are
retrieved by an exact match on ethical dimension, country and sector —
these bibliography entries are retrieved purely by semantic similarity
between the dilemma text and each entry's summary, using the same
multilingual sentence-transformers model already used by the classifier
(model/ethics_classifier.py). This is deliberately kept as a separate,
clearly-labeled section in the report ("Further literature — semantic
match") rather than merged into "Relevant principles", so the reader can
tell the difference between an exact, tag-based match and a broader,
similarity-based one.

Growing the bibliography: add an entry to data/bibliography.json with an
id, citation, optional url, and a bilingual {"en", "el"} summary written
in your own words — not a copied abstract, consistent with the rest of
this project's copyright-conscious approach to sourced content. No other
code changes are needed; new entries are picked up automatically the
next time the app starts.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import json
import os

import numpy as np

from model.ethics_classifier import embed_texts

# -----------------------------------------------------------------------
# Ρυθμίσεις: διαδρομή corpus, ελάχιστη ομοιότητα, πλήθος αποτελεσμάτων
# -----------------------------------------------------------------------
BIBLIOGRAPHY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bibliography.json")
MIN_SIMILARITY = 0.35
TOP_K = 3

# Καθυστερημένη φόρτωση: το corpus και οι ενσωματώσεις του υπολογίζονται
# μία φορά ανά διεργασία, την πρώτη φορά που ζητηθεί ανάκτηση
_entries = None
_embeddings = None


# -----------------------------------------------------------------------
# Φόρτωση του corpus και προϋπολογισμός των ενσωματώσεών του
# -----------------------------------------------------------------------
def _load():
    global _entries, _embeddings
    if _entries is not None:
        return
    with open(BIBLIOGRAPHY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _entries = data["entries"]
    summaries_en = [e["summary"]["en"] for e in _entries]
    _embeddings = embed_texts(summaries_en)


# -----------------------------------------------------------------------
# Υπολογισμός συνημιτονικής ομοιότητας ανάμεσα σε πίνακα ενσωματώσεων
# και ένα μεμονωμένο διάνυσμα ερωτήματος
# -----------------------------------------------------------------------
def _cosine_sim(matrix, vector):
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    vector_norm = vector / (np.linalg.norm(vector) + 1e-8)
    return matrix_norm @ vector_norm


# -----------------------------------------------------------------------
# Δημόσια συνάρτηση: επιστρέφει τις πιο σημασιολογικά κοντινές εγγραφές
# βιβλιογραφίας στο δοσμένο δίλημμα
# -----------------------------------------------------------------------
def retrieve_bibliography(dilemma: str, lang: str = "en") -> list:
    """Return up to TOP_K bibliography entries whose summary is
    semantically similar to the dilemma, above MIN_SIMILARITY.

    Fails silently (returns an empty list) on any error, since this is an
    additive enhancement — a broken bibliography lookup should never break
    the rest of the report.
    """
    if not dilemma or not dilemma.strip():
        return []
    try:
        _load()
        query_embedding = embed_texts([dilemma])[0]
        sims = _cosine_sim(_embeddings, query_embedding)
        ranked = sorted(zip(_entries, sims), key=lambda pair: -pair[1])

        results = []
        for entry, score in ranked[:TOP_K]:
            if score < MIN_SIMILARITY:
                continue
            results.append({
                "id": entry["id"],
                "citation": entry["citation"],
                "url": entry.get("url"),
                "summary": entry["summary"].get(lang) or entry["summary"]["en"],
                "similarity": round(float(score), 3),
            })
        return results
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[bibliography] retrieval failed: {exc}")
        return []
