"""
AI-thical — Πρωτότυπο συμβουλευτικής ΑΙ για ηθικά & νομικά ζητήματα.

Backend Flask για το πρωτότυπο της διατριβής
"Aithical: The AI Consulting LLM for Ethical and Legal issues about AI"
(Κ. Σολδάτος). Ο χρήστης περιγράφει ένα επιχειρηματικό δίλημμα που αφορά
την ΑΙ, επιλέγει χώρα και επιχειρηματική λειτουργία, και λαμβάνει μια
δομημένη αναφορά συμβουλευτικής:

    κείμενο διλήμματος --> ταξινομητής ηθικών διαστάσεων (TensorFlow)
                        --> ανάκτηση από τη βάση γνώσης (φιλτραρισμένη
                            ανά χώρα και κλάδο)
                        --> δομημένη, τεκμηριωμένη αναφορά + αποποίηση
                            ευθύνης

Τόσο το περιβάλλον χρήστη όσο και το περιεχόμενο της βάσης γνώσης
υποστηρίζουν Αγγλικά και Ελληνικά (βλ. data/knowledge_base.json και την
παράμετρο `lang` στο engine/advisor.py). Ο ταξινομητής είναι πολυγλωσσικός
(βλ. model/ethics_classifier.py), οπότε τα διλήμματα μπορούν να γραφτούν
και στις δύο γλώσσες.

Εκτέλεση:
    python app.py
Και άνοιγμα του http://127.0.0.1:5000
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών και εσωτερικών ενοτήτων της εφαρμογής
# -----------------------------------------------------------------------
import os
import time
from collections import defaultdict, deque
from datetime import date

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from engine.advisor import EU_MEMBER_STATES, US_STATES, analyze, detect_lang, load_knowledge_base
from engine.bibliography import retrieve_bibliography
from engine.live_search import LiveSearchUnavailable, search_recent_legal_updates
from engine.risk_tier import assess_risk_tier
from engine.llm_client import LLMUnavailable, generate_narrative
from engine.usage_log import log_prediction
from model.ethics_classifier import EthicsClassifier

# -----------------------------------------------------------------------
# Αρχικοποίηση εφαρμογής
# -----------------------------------------------------------------------
# Φόρτωση μεταβλητών περιβάλλοντος από το αρχείο .env, εφόσον υπάρχει·
# η κλήση είναι ακίνδυνη (no-op) αν το αρχείο δεν βρεθεί.
load_dotenv()

app = Flask(__name__)

# Η βάση γνώσης και ο ταξινομητής φορτώνονται μία φορά κατά την εκκίνηση
# και επαναχρησιμοποιούνται σε κάθε αίτημα, αντί να ξαναφορτώνονται κάθε
# φορά — σημαντικό ειδικά για τον ταξινομητή, καθώς η φόρτωσή του απαιτεί
# υπολογιστικό κόστος.
knowledge_base = load_knowledge_base()
classifier = EthicsClassifier()

MAX_DILEMMA_LENGTH = 2000
SUPPORTED_LANGS = ("en", "el")


# -----------------------------------------------------------------------
# Κύρια σελίδα: απόδοση του προτύπου με τις διαθέσιμες επιλογές
# χώρας/κλάδου και τις ρυθμίσεις που αφορούν τον πάροχο LLM
# -----------------------------------------------------------------------
@app.route("/")
def index():
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    return render_template(
        "index.html",
        countries=knowledge_base["countries"],
        sectors=knowledge_base["sectors"],
        country_labels=knowledge_base["country_labels"],
        sector_labels=knowledge_base["sector_labels"],
        # Δύο υπο-λίστες, αλφαβητικά ταξινομημένες (κατά το αγγλικό όνομα,
        # ώστε η σειρά στο DOM να μένει σταθερή ανεξαρτήτως γλώσσας
        # εμφάνισης), για το διβάθμιο μενού επιλογής χώρας/πολιτείας.
        eu_countries=sorted(EU_MEMBER_STATES),
        us_states=sorted(US_STATES),
        using_fallback_classifier=classifier.using_fallback,
        llm_provider_is_local=(provider == "ollama"),
        llm_provider_configured=bool(provider),
    )


# -----------------------------------------------------------------------
# Απλός περιοριστής ρυθμού αιτημάτων (rate limiting), χωρίς εξωτερική
# εξάρτηση: προστατεύει το /api/analyze από κατάχρηση (π.χ. αυτοματοποιημένο
# "βομβαρδισμό" που θα εξαντλούσε τα δωρεάν credits της Tavily/του LLM API,
# ή θα επιβάρυνε υπερβολικά τον τοπικό υπολογιστή). Κρατάει, ανά διεύθυνση
# IP, τις χρονοσφραγίδες των τελευταίων αιτημάτων μέσα σε ένα παράθυρο
# χρόνου· αν ξεπεραστεί το όριο, επιστρέφει 429 αντί να επεξεργαστεί το
# αίτημα. Λειτουργεί σωστά και πίσω από Cloudflare Tunnel, γιατί διαβάζει
# πρώτα την κεφαλίδα CF-Connecting-IP (η πραγματική IP του επισκέπτη),
# πριν καταφύγει στο request.remote_addr (πάντα 127.0.0.1 μέσω tunnel).
# -----------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
_request_log = defaultdict(deque)


def _client_ip() -> str:
    return request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "unknown"


def _rate_limited() -> bool:
    ip = _client_ip()
    now = time.monotonic()
    recent = _request_log[ip]
    while recent and now - recent[0] > RATE_LIMIT_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    recent.append(now)
    return False


# -----------------------------------------------------------------------
# Κύριο endpoint ανάλυσης: λαμβάνει το δίλημμα και επιστρέφει τη
# δομημένη αναφορά συμβουλευτικής σε μορφή JSON
# -----------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if _rate_limited():
        return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429

    payload = request.get_json(silent=True) or {}
    dilemma = (payload.get("dilemma") or "").strip()
    country = (payload.get("country") or "").strip()
    sector = (payload.get("sector") or "").strip()

    # Η γλώσσα απόκρισης καθορίζεται πρωτίστως από το ίδιο το κείμενο του
    # διλήμματος (ανίχνευση ελληνικού αλφαβήτου), όχι μόνο από το κουμπί
    # γλώσσας του περιβάλλοντος χρήστη· έτσι η απάντηση ταιριάζει με ό,τι
    # γράφτηκε πραγματικά, ακόμα κι αν ο χρήστης ξέχασε να αλλάξει γλώσσα.
    lang = (payload.get("lang") or "en").strip().lower()
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    # Όταν ο χρήστης πάτησε ρητά το κουμπί γλώσσας (επανυποβολή της ίδιας
    # αναφοράς σε άλλη γλώσσα), η επιλογή του υπερισχύει της αυτόματης
    # ανίχνευσης· αλλιώς ένα αγγλικό δίλημμα θα γύριζε πάντα στα αγγλικά.
    if not payload.get("lang_explicit"):
        lang = detect_lang(dilemma, fallback=lang)

    # ---------------------------------------------------------------
    # Έλεγχοι εγκυρότητας εισόδου
    # ---------------------------------------------------------------
    if not dilemma:
        message = "Περιγράψτε ένα επιχειρηματικό δίλημμα πριν την υποβολή." if lang == "el" \
            else "Please describe a business dilemma before submitting."
        return jsonify({"error": message}), 400
    if len(dilemma) > MAX_DILEMMA_LENGTH:
        message = f"Κρατήστε την περιγραφή κάτω από {MAX_DILEMMA_LENGTH} χαρακτήρες." if lang == "el" \
            else f"Please keep the description under {MAX_DILEMMA_LENGTH} characters."
        return jsonify({"error": message}), 400

    # ---------------------------------------------------------------
    # Κύρια αλυσίδα επεξεργασίας: ταξινόμηση ηθικών διαστάσεων και
    # ανάκτηση σχετικών αρχών, κανονισμών και προτεινόμενων ενεργειών
    # από την επιμελημένη βάση γνώσης
    # ---------------------------------------------------------------
    result = analyze(dilemma, country, sector, classifier, kb=knowledge_base, lang=lang)

    # ---------------------------------------------------------------
    # Καταγραφή (μόνο) της πρόβλεψης για πιθανή μελλοντική επέκταση του
    # συνόλου εκπαίδευσης — δεν επηρεάζει καθόλου τον ταξινομητή από
    # μόνη της. Βλ. engine/usage_log.py και model/review_log.py.
    # ---------------------------------------------------------------
    log_prediction(dilemma, result["all_scores"], country, sector, lang)

    # ---------------------------------------------------------------
    # Εκτίμηση της βαθμίδας κινδύνου κατά την πυραμίδα του Κανονισμού
    # ΤΝ της ΕΕ (απαγορευμένος / υψηλός / περιορισμένος / ελάχιστος),
    # με τη νομική βάση κάθε αντιστοίχισης. Βλ. engine/risk_tier.py.
    # ---------------------------------------------------------------
    result["ai_act_risk"] = assess_risk_tier(
        dilemma,
        sector=result["sector"],
        in_eu=not (result["country"] in US_STATES or result["country"] == "United States"),
        lang=lang,
    )

    # ---------------------------------------------------------------
    # Σημασιολογική ανάκτηση βιβλιογραφίας (μικρό στρώμα RAG), πάντα
    # ενεργή — δεν έχει κόστος ή ζήτημα ιδιωτικότητας, καθώς εκτελείται
    # πλήρως τοπικά πάνω στο ήδη φορτωμένο πολυγλωσσικό μοντέλο
    # ---------------------------------------------------------------
    result["further_reading"] = retrieve_bibliography(dilemma, lang=lang)

    # ---------------------------------------------------------------
    # Προαιρετική σύνθεση φυσικής γλώσσας μέσω LLM· ενεργοποιείται μόνο
    # εφόσον έχει επιλεγεί ρητά από τον χρήστη, καθώς είναι το μοναδικό
    # σημείο της αλυσίδας όπου το κείμενο του διλήμματος ενδέχεται να
    # αποσταλεί εκτός του τοπικού συστήματος
    # ---------------------------------------------------------------
    result["narrative"] = None
    result["narrative_error"] = None
    if payload.get("generate_narrative"):
        try:
            result["narrative"] = generate_narrative(
                dilemma=result["dilemma"],
                country=result["country"],
                sector=result["sector"],
                detected_dimensions=result["detected_dimensions"],
                principles=result["principles"],
                regulations=result["regulations"],
                actions=result["actions"],
                literature=result["further_reading"],
                risk=result["ai_act_risk"],
                lang=lang,
            )
        except LLMUnavailable as exc:
            result["narrative_error"] = str(exc)

    # ---------------------------------------------------------------
    # Προαιρετική ζωντανή αναζήτηση πρόσφατων πηγών στο διαδίκτυο· τα
    # αποτελέσματα παρουσιάζονται ξεχωριστά από την επιμελημένη βάση
    # γνώσης, καθώς δεν έχουν επαληθευτεί
    # ---------------------------------------------------------------
    result["live_sources"] = None
    result["live_sources_error"] = None
    if payload.get("include_live_sources"):
        try:
            # Πρώτα το πιο συγκεκριμένο ερώτημα (χώρα + έως 2 ανιχνευμένες
            # ηθικές διαστάσεις). Όσο πιο πολλές/συγκεκριμένες διαστάσεις
            # ανιχνεύει ο ταξινομητής, τόσο πιο στενό γίνεται το ερώτημα —
            # και για μικρότερες χώρες αυτό εύκολα βρίσκει μηδέν πραγματικά
            # συναφή αποτελέσματα. Αν συμβεί αυτό, ξαναδοκιμάζουμε με πιο
            # ευρύ ερώτημα (μόνο χώρα + έτος) πριν παραδεχτούμε ότι δεν
            # υπάρχει τίποτα σχετικό.
            query = _build_live_search_query(result, lang, include_topic=True)
            raw_results = search_recent_legal_updates(query)
            filtered = _filter_relevant_sources(raw_results, result["country"], lang)

            if not filtered:
                broader_query = _build_live_search_query(result, lang, include_topic=False)
                raw_results = search_recent_legal_updates(broader_query)
                filtered = _filter_relevant_sources(raw_results, result["country"], lang)

            result["live_sources"] = filtered
        except LiveSearchUnavailable as exc:
            result["live_sources_error"] = str(exc)

    return jsonify(result)


# -----------------------------------------------------------------------
# Φίλτρο συνάφειας: η ζωντανή αναζήτηση (Tavily) δεν κάνει πάντα τέλειο σημασιολογικό
# ταίριασμα — όταν δεν βρίσκει κάτι ακριβές για μια λιγότερο "δημοφιλή"
# χώρα/πολιτεία, μπορεί να επιστρέψει χαλαρά συναφή αποτελέσματα (π.χ.
# ένα δημοφιλές άρθρο για άλλη χώρα) ως "καλύτερη διαθέσιμη" απάντηση.
# Κρατάμε μόνο αποτελέσματα όπου το όνομα της επιλεγμένης χώρας
# εμφανίζεται πράγματι στον τίτλο ή το απόσπασμα — προτιμότερο κενό
# αποτέλεσμα από παραπλανητικά άσχετο περιεχόμενο.
# -----------------------------------------------------------------------
# Επιθετικοί τύποι (demonyms) για τα κράτη-μέλη ΕΕ — τα δημοσιογραφικά
# άρθρα γράφουν συχνά "Greek AI law" ή "Italian data authority" αντί για
# το ίδιο το όνομα της χώρας, οπότε το φίλτρο συνάφειας πρέπει να
# αναγνωρίζει και αυτούς τους τύπους, όχι μόνο το ουσιαστικό.
COUNTRY_DEMONYMS = {
    "Austria": "Austrian", "Belgium": "Belgian", "Bulgaria": "Bulgarian",
    "Croatia": "Croatian", "Cyprus": "Cypriot", "Czech Republic": "Czech",
    "Denmark": "Danish", "Estonia": "Estonian", "Finland": "Finnish",
    "France": "French", "Germany": "German", "Greece": "Greek",
    "Hungary": "Hungarian", "Ireland": "Irish", "Italy": "Italian",
    "Latvia": "Latvian", "Lithuania": "Lithuanian", "Malta": "Maltese",
    "Netherlands": "Dutch", "Poland": "Polish", "Portugal": "Portuguese",
    "Romania": "Romanian", "Slovakia": "Slovak", "Slovenia": "Slovenian",
    "Spain": "Spanish", "Sweden": "Swedish",
}


def _filter_relevant_sources(sources: list, country: str, lang: str) -> list:
    labels = knowledge_base["country_labels"].get(country, {})
    needles = {
        n.lower() for n in [country, labels.get("en"), labels.get("el"), COUNTRY_DEMONYMS.get(country)]
        if n
    }
    if not needles:
        return sources

    relevant = []
    for item in sources:
        haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        if any(needle in haystack for needle in needles):
            relevant.append(item)
    return relevant


# -----------------------------------------------------------------------
# Βοηθητική συνάρτηση: σύνθεση ερωτήματος αναζήτησης για την εύρεση
# πρόσφατων νομοθετικών ενημερώσεων, στη γλώσσα της απόκρισης
# -----------------------------------------------------------------------
def _build_live_search_query(result: dict, lang: str, include_topic: bool = True) -> str:
    year = date.today().year
    country = result["country"]
    country_label = knowledge_base["country_labels"].get(country, {}).get(lang, country)
    dim_labels = [d["label"] for d in result["detected_dimensions"][:2]] if include_topic else []

    if lang == "el":
        topic = " ".join(dim_labels) if dim_labels else "τεχνητή νοημοσύνη"
        return f"νέα νομοθεσία {topic} {country_label} {year}"
    topic = " ".join(dim_labels) if dim_labels else "artificial intelligence"
    return f"new AI regulation {topic} {country_label} {year}"


# -----------------------------------------------------------------------
# Endpoint ελέγχου κατάστασης λειτουργίας (health check)
# -----------------------------------------------------------------------
@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "using_fallback_classifier": classifier.using_fallback,
    })


# -----------------------------------------------------------------------
# Σημείο εκκίνησης της εφαρμογής
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Το debug=True ενεργοποιεί τον διαδραστικό debugger του Werkzeug, ο
    # οποίος επιτρέπει εκτέλεση αυθαίρετου κώδικα σε όποιον έχει πρόσβαση
    # στη σελίδα — ακίνδυνο τοπικά, επικίνδυνο αν η εφαρμογή εκτεθεί
    # δημόσια (π.χ. μέσω tunnel). Η μεταβλητή FLASK_DEBUG=1 στο .env
    # ενεργοποιεί το debug mode μόνο για τοπική ανάπτυξη· διαφορετικά
    # παραμένει απενεργοποιημένο από προεπιλογή.
    debug_mode = os.environ.get("FLASK_DEBUG", "").strip() == "1"
    app.run(debug=debug_mode)
