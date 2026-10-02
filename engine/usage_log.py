"""
Καταγραφή πραγματικών χρήσεων της εφαρμογής (δίλημμα + πρόβλεψη
μοντέλου) ώστε να μπορούν αργότερα να αξιολογηθούν και, αν εγκριθούν,
να επεκτείνουν το σύνολο εκπαίδευσης.

ΣΗΜΑΝΤΙΚΟ — γιατί δεν «μαθαίνει» αυτόματα το μοντέλο από αυτό: αν
προσθέταμε απευθείας κάθε (δίλημμα, πρόβλεψη) ως νέο παράδειγμα
εκπαίδευσης, το μοντέλο θα άρχιζε σταδιακά να επιβεβαιώνει τα δικά του
λάθη αντί να μαθαίνει από αυτά (feedback loop) — ιδίως επικίνδυνο με
τόσο μικρό σύνολο εκπαίδευσης, όπου λίγες λανθασμένες προσθήκες
αρκούν για να μετατοπίσουν αισθητά τη συμπεριφορά του ταξινομητή. Γι'
αυτό η καταγραφή εδώ είναι αυστηρά μονόδρομη προς ένα αρχείο· η
προαγωγή μιας καταγραφής σε πραγματικό παράδειγμα εκπαίδευσης γίνεται
ΜΟΝΟ μέσω ανθρώπινης επιθεώρησης (βλ. model/review_log.py).

Το αρχείο καταγραφής (data/usage_log.jsonl) περιέχει το πραγματικό
κείμενο που υπέβαλε κάποιος χρήστης της εφαρμογής — ενδέχεται να
περιλαμβάνει ευαίσθητες επιχειρηματικές πληροφορίες. Δεν πρέπει ποτέ
να ανέβει σε δημόσιο αποθετήριο (ήδη προστέθηκε στο .gitignore).
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "usage_log.jsonl")


# -----------------------------------------------------------------------
# Δημόσια συνάρτηση: προσθήκη μίας εγγραφής χρήσης στο τοπικό αρχείο
# καταγραφής (append-only, μορφή JSON Lines — μία εγγραφή ανά γραμμή)
# -----------------------------------------------------------------------
def log_prediction(dilemma: str, all_scores: dict, country: str, sector: str, lang: str):
    """Append one usage record to the local, gitignored JSONL log.

    Fails silently on any error — logging is a best-effort side channel
    and must never break the actual user-facing request.

    Disabled on the public deployment: Cloud Run sets K_SERVICE, and there
    visitors' dilemmas are not stored at all (USAGE_LOG=true overrides).
    """
    if os.environ.get("K_SERVICE") and os.environ.get("USAGE_LOG", "").lower() != "true":
        return
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dilemma": dilemma,
            "scores": all_scores,
            "country": country,
            "sector": sector,
            "lang": lang,
            "reviewed": False,
        }
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[usage_log] failed to log prediction: {exc}")
