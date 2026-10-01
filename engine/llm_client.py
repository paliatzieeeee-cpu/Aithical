"""
LLM client for Aithicist, used for two optional, clearly separated jobs:

1. generate_narrative(...) - phrase already-retrieved knowledge-base
   sources into a natural-language paragraph. Never introduces new claims.
2. classify_dilemma(...) - a fallback classifier, used only when the local
   TensorFlow model has low confidence across every category, i.e. exactly
   the "keyword doesn't ring, model isn't sure" case. This does NOT
   replace the TensorFlow classifier; it only fills in when it is unsure,
   and its scores are combined with (not substituted for) the TF scores
   in engine/advisor.py.

Three providers are supported, chosen via the LLM_PROVIDER environment
variable:

  - "ollama"    - a locally running Ollama server (no API key, no data
                  leaves the machine). Since there is no cost or privacy
                  trade-off, this is the only provider used automatically
                  for the classification fallback (see advisor.py).
  - "openai"    - OpenAI's API (requires OPENAI_API_KEY).
  - "anthropic" - Anthropic's API (requires ANTHROPIC_API_KEY).
  - "groq"      - Groq's OpenAI-compatible API (requires GROQ_API_KEY;
                  model chosen with GROQ_MODEL). Called with plain
                  `requests`, so no extra SDK is needed.

Both openai/anthropic SDKs and the `requests` calls to Ollama/Groq are
imported lazily, so the app runs fine with none of this configured.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import json
import os
import re

from model.training_data import CATEGORIES

# -----------------------------------------------------------------------
# Οδηγίες συστήματος (system prompts) για τη σύνθεση φυσικής γλώσσας,
# ξεχωριστές για Αγγλικά και Ελληνικά
# -----------------------------------------------------------------------
NARRATIVE_SYSTEM_PROMPT_EN = (
    "You are a compliance-writing assistant. You will be given a business "
    "dilemma involving AI, a set of detected ethical dimensions, a list "
    "of sourced principles, regulations, and recommended actions retrieved "
    "from a curated knowledge base, and optionally a short list of semantically "
    "matched academic literature. Write a short (120-200 word) advisory "
    "paragraph in plain, professional English that weaves these points "
    "together for the reader, drawing on the literature where it genuinely adds "
    "useful context (e.g. why a tension exists, or what the wider debate says), "
    "not just on the principles/regulations/actions alone. Rules: "
    "(1) Use ONLY the information given to you below — do not introduce any "
    "new fact, law, statistic, or recommendation that is not already present "
    "in the provided sources. "
    "(2) Do not invent citations or sources; when drawing on the literature, "
    "refer to it only by the author name(s) already given, never a page number "
    "or quote you were not given. "
    "(3) If the provided material is thin, keep the paragraph short rather "
    "than padding it with invented content. "
    "(4) Never state or imply that something is allowed, permitted, legal, "
    "acceptable, compliant or safe, and never answer a 'can we do this?' "
    "question with yes or no — not even conditionally. Do not use phrasings "
    "such as 'is allowed if', 'is acceptable only if', 'you can', 'you may', "
    "'this ensures compliance' or 'this guarantees legality'. Instead write "
    "what the given rules require and what the reader must check, document or "
    "disclose, e.g. 'whether this is possible depends on …, which a qualified "
    "professional should confirm'. "
    "(5) Each regulation below is listed with its source and article. Attribute "
    "an obligation only to the source and article of the item that states it. "
    "National implementing laws (for example Greek Law 5321/2026) designate "
    "supervisory authorities, sanctions and registries; never write that such "
    "a law itself imposes an EU Regulation's obligations — attribute those to "
    "the EU Regulation. Attribute a recommendation to an author only if it is "
    "listed under that author's name. "
    "(6) The detected ethical dimensions are an automated estimate and may be "
    "wrong, so some items below may not fit the dilemma: leave out any item "
    "that does not clearly apply to it. Mention data protection rules (e.g. "
    "the GDPR) only if the dilemma itself involves personal data such as "
    "people's names, faces, voices, or customer, employee or patient data. "
    "(7) The EU AI Act risk level given below was assessed by the application: "
    "if you mention a risk level, state exactly that level and its legal basis, "
    "never a higher or lower one, and present it as an automated indication. "
    "(8) End with a one-sentence reminder that this is general guidance, "
    "not legal advice."
)

NARRATIVE_SYSTEM_PROMPT_EL = (
    "Είσαι βοηθός σύνταξης κειμένων κανονιστικής συμμόρφωσης. Θα σου δοθεί ένα "
    "επιχειρηματικό δίλημμα που αφορά την ΤΝ, ένα σύνολο εντοπισμένων ηθικών "
    "διαστάσεων, μια λίστα τεκμηριωμένων αρχών, κανονισμών και προτεινόμενων "
    "ενεργειών από μια επιμελημένη βάση γνώσης, και προαιρετικά μια σύντομη "
    "λίστα σημασιολογικά αντιστοιχισμένης ακαδημαϊκής βιβλιογραφίας. Γράψε μια "
    "σύντομη παράγραφο (120-200 λέξεων) σε απλά, επαγγελματικά ελληνικά που "
    "συνδέει αυτά τα σημεία για τον αναγνώστη, αξιοποιώντας τη βιβλιογραφία "
    "όπου προσθέτει πραγματικά χρήσιμο πλαίσιο (π.χ. γιατί υπάρχει μια ένταση, "
    "ή τι λέει η ευρύτερη συζήτηση), όχι μόνο τις αρχές/κανονισμούς/ενέργειες. "
    "Κανόνες: "
    "(1) Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες που σου δίνονται παρακάτω — μην "
    "εισάγεις κανένα νέο γεγονός, νόμο, στατιστικό στοιχείο ή σύσταση που δεν "
    "υπάρχει ήδη στις παρεχόμενες πηγές. "
    "(2) Μην επινοήσεις παραπομπές ή πηγές· όταν αναφέρεσαι στη βιβλιογραφία, "
    "χρησιμοποίησε μόνο το όνομα συγγραφέα που ήδη σου δόθηκε, ποτέ αριθμό "
    "σελίδας ή απόσπασμα που δεν σου δόθηκε. "
    "(3) Αν το παρεχόμενο υλικό είναι λιγοστό, κράτησε την παράγραφο σύντομη "
    "αντί να την γεμίσεις με επινοημένο περιεχόμενο. "
    "(4) Μην γράψεις ούτε να υπονοήσεις ποτέ ότι κάτι επιτρέπεται, είναι "
    "νόμιμο, αποδεκτό, σύμφωνο με τον νόμο ή ασφαλές, και μην απαντάς με ναι ή "
    "όχι σε ερώτημα του τύπου «μπορούμε να το κάνουμε;» — ούτε υπό όρους. Μη "
    "χρησιμοποιείς διατυπώσεις όπως «επιτρέπεται εφόσον», «είναι αποδεκτό μόνο "
    "εφόσον», «μπορείτε να», «διασφαλίζει τη συμμόρφωση» ή «εξασφαλίζει τη "
    "νομιμότητα». Γράψε αντίθετα τι απαιτούν οι κανόνες που σου δόθηκαν και τι "
    "πρέπει να ελέγξει, να τεκμηριώσει ή να γνωστοποιήσει ο αναγνώστης, π.χ. "
    "«το αν είναι εφικτό εξαρτάται από …, κάτι που πρέπει να επιβεβαιώσει "
    "εξειδικευμένος επαγγελματίας». "
    "(5) Κάθε κανονισμός παρακάτω δίνεται με την πηγή και το άρθρο του. "
    "Απόδιδε μια υποχρέωση μόνο στην πηγή και στο άρθρο της εγγραφής που τη "
    "διατυπώνει. Οι εθνικοί νόμοι εφαρμογής (π.χ. ο ελληνικός Ν. 5321/2026) "
    "ορίζουν εποπτικές αρχές, κυρώσεις και μητρώα· μη γράψεις ποτέ ότι ένας "
    "τέτοιος νόμος επιβάλλει ο ίδιος τις υποχρεώσεις ενός Κανονισμού της ΕΕ — "
    "απόδωσέ τες στον Κανονισμό. Απόδιδε μια σύσταση σε συγγραφέα μόνο αν "
    "δίνεται με το όνομά του. "
    "(6) Οι εντοπισμένες ηθικές διαστάσεις είναι αυτοματοποιημένη εκτίμηση "
    "και μπορεί να είναι λάθος, οπότε κάποια στοιχεία παρακάτω ίσως δεν "
    "ταιριάζουν στο δίλημμα: παράλειψε όποιο δεν το αφορά σαφώς. Ανέφερε "
    "κανόνες προστασίας δεδομένων (π.χ. τον ΓΚΠΔ) μόνο αν το ίδιο το δίλημμα "
    "αφορά προσωπικά δεδομένα, όπως ονόματα, πρόσωπα ή φωνές ανθρώπων, ή "
    "δεδομένα πελατών, εργαζομένων ή ασθενών. "
    "(7) Η βαθμίδα κινδύνου του AI Act που δίνεται παρακάτω υπολογίστηκε από "
    "την εφαρμογή: αν αναφέρεις βαθμίδα κινδύνου, ανέφερε ακριβώς αυτήν και τη "
    "νομική της βάση, ποτέ υψηλότερη ή χαμηλότερη, και παρουσίασέ την ως "
    "αυτοματοποιημένη ένδειξη. "
    "(8) Κλείσε με μία πρόταση που υπενθυμίζει ότι πρόκειται για γενική "
    "καθοδήγηση, όχι νομική συμβουλή."
)

# -----------------------------------------------------------------------
# Οδηγία συστήματος για την ενισχυτική ταξινόμηση μέσω LLM
# -----------------------------------------------------------------------
CLASSIFY_SYSTEM_PROMPT = (
    "You classify short descriptions of business dilemmas involving AI into "
    "five ethical dimensions: transparency, fairness, non_maleficence, "
    "accountability, privacy. The dilemma may be written in English, Greek, "
    "or another language — classify it regardless of language. For the "
    "dilemma given, respond with ONLY a "
    "JSON object (no prose, no markdown fences) mapping each of those five "
    "exact keys to a probability between 0 and 1 representing how strongly "
    "that dimension is present in the dilemma. Multiple dimensions can score "
    "high at once; unrelated dimensions should score near 0. "
    'Example output: {"transparency": 0.1, "fairness": 0.7, '
    '"non_maleficence": 0.2, "accountability": 0.6, "privacy": 0.0}'
)


# -----------------------------------------------------------------------
# Εξαίρεση που σηματοδοτεί ότι το LLM δεν είναι διαθέσιμο (χωρίς ρύθμιση
# παρόχου, ή αποτυχία κλήσης/ανάλυσης απόκρισης)
# -----------------------------------------------------------------------
class LLMUnavailable(Exception):
    """Raised when no provider is configured, or a call/parse fails."""


# -----------------------------------------------------------------------
# Κεντρική δρομολόγηση προς τον κατάλληλο πάροχο LLM, βάσει της
# μεταβλητής περιβάλλοντος LLM_PROVIDER
# -----------------------------------------------------------------------
def _dispatch(system_prompt: str, user_prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "ollama":
        return _call_ollama(system_prompt, user_prompt)
    if provider == "openai":
        return _call_openai(system_prompt, user_prompt)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    if provider == "groq":
        return _call_groq(system_prompt, user_prompt)
    raise LLMUnavailable(
        "Set LLM_PROVIDER to 'ollama', 'openai', 'anthropic' or 'groq' in your environment to enable this feature."
    )


# -----------------------------------------------------------------------
# Κλήση σε τοπικά εκτελούμενο μοντέλο Ollama (χωρίς κλειδί API)
# -----------------------------------------------------------------------
def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    import requests  # lazy import

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")

    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=int(os.environ.get("OLLAMA_TIMEOUT", "120")),
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnavailable(
            f"Could not reach Ollama at {host}. Is `ollama serve` running?"
        ) from exc
    except Exception as exc:  # pragma: no cover - network/HTTP errors
        raise LLMUnavailable(f"Ollama call failed: {exc}") from exc

    data = response.json()
    try:
        return data["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        raise LLMUnavailable(f"Unexpected Ollama response shape: {data}") from exc


# -----------------------------------------------------------------------
# Κλήση στο API του OpenAI
# -----------------------------------------------------------------------
def _call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # lazy import

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        max_tokens=350,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# -----------------------------------------------------------------------
# Κλήση στο API του Anthropic
# -----------------------------------------------------------------------
def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    import anthropic  # lazy import

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    response = client.messages.create(
        model=model,
        max_tokens=350,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


# -----------------------------------------------------------------------
# Κλήση στο API του Groq (συμβατό με το OpenAI)
# -----------------------------------------------------------------------
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    import requests  # lazy import

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMUnavailable("GROQ_API_KEY is not set.")
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Τα ελληνικά χρειάζονται αρκετά περισσότερα tokens ανά λέξη από τα
        # αγγλικά, και στα μοντέλα συλλογισμού μετρά και ο συλλογισμός
        "max_completion_tokens": 1024,
        # Χαμηλή τυχαιότητα: το μοντέλο τηρεί σταθερότερα τους κανόνες της
        # οδηγίας (απόδοση πηγών, όχι «άδειες») και διατυπώνει πιο συνεπώς
        "temperature": 0.2,
    }
    # Τα μοντέλα του Groq είναι μοντέλα συλλογισμού (reasoning): ο
    # συλλογισμός δεν πρέπει να εμφανίζεται στη σύνοψη, και κρατιέται
    # σύντομος ώστε η απάντηση να έρχεται γρήγορα
    if model.startswith("openai/gpt-oss"):
        payload["include_reasoning"] = False
        payload["reasoning_effort"] = "low"
    elif model.startswith("qwen/"):
        payload["reasoning_format"] = "hidden"
        payload["reasoning_effort"] = "none"

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=int(os.environ.get("GROQ_TIMEOUT", "30")),
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""
    except requests.exceptions.HTTPError as exc:
        # Μόνο ο κωδικός κατάστασης, ποτέ το σώμα του αιτήματος (κλειδί, δίλημμα)
        status = exc.response.status_code if exc.response is not None else "?"
        raise LLMUnavailable(f"Groq request failed (HTTP {status}).") from exc
    except Exception as exc:  # δίκτυο, timeout, μη αναμενόμενη μορφή απόκρισης
        raise LLMUnavailable(f"Groq call failed: {type(exc).__name__}") from exc

    content = _THINK_RE.sub("", content).strip()
    if not content:
        raise LLMUnavailable("Groq returned an empty response.")
    return content


# -----------------------------------------------------------------------
# Σύνθεση του κειμένου εντολής χρήστη (user prompt) από τα ανακτημένα
# στοιχεία: δίλημμα, εντοπισμένες διαστάσεις, αρχές, κανονισμοί, ενέργειες
# -----------------------------------------------------------------------
def _build_narrative_prompt(dilemma, country, sector, detected_dimensions, principles, regulations, actions, literature=None, risk=None):
    def fmt(entries):
        if not entries:
            return "  (none retrieved)"
        # Το άρθρο δίνεται ρητά, ώστε το μοντέλο να αποδίδει κάθε υποχρέωση
        # στη σωστή διάταξη και όχι σε όποιον νόμο αναφέρεται κοντά της
        return "\n".join(
            f"  - {e['text']} [source: {e['source']}"
            + (f"; article: {e['article']}" if e.get("article") else "") + "]"
            for e in entries
        )

    def fmt_literature(entries):
        if not entries:
            return "  (none matched)"
        return "\n".join(f"  - {e['summary']} [source: {e['citation']}]" for e in entries)

    dims = ", ".join(d["label"] for d in detected_dimensions) or "none strongly detected"

    # Η βαθμίδα κινδύνου του AI Act που υπολόγισε η εφαρμογή, ώστε η σύνοψη
    # να μην έρχεται σε αντίφαση με την ενότητα της πυραμίδας (π.χ. να γράφει
    # «υψηλού κινδύνου» για μια πρακτική που η εφαρμογή κρίνει απαγορευμένη)
    risk_text = "  (not assessed)"
    if risk:
        refs = "; ".join(f"{m['ref']} ({m['label']})" for m in risk.get("matches", [])) or "no specific provision matched"
        notes = " ".join(risk.get("notes", [])[:-1])  # η τελευταία σημείωση είναι η γενική αποποίηση
        risk_text = f"  {risk.get('verdict', '')}\n  Legal basis: {refs}" + (f"\n  Notes: {notes}" if notes else "")

    return (
        f"Country: {country}\n"
        f"Business function: {sector}\n"
        f"Dilemma: {dilemma}\n\n"
        f"Detected ethical dimensions: {dims}\n\n"
        f"EU AI Act risk level (automated indication):\n{risk_text}\n\n"
        f"Principles:\n{fmt(principles)}\n\n"
        f"Regulatory considerations:\n{fmt(regulations)}\n\n"
        f"Recommended actions:\n{fmt(actions)}\n\n"
        f"Related academic literature (semantically matched, use only where it adds real context):\n{fmt_literature(literature)}\n"
    )


# -----------------------------------------------------------------------
# Δημόσια συνάρτηση: παραγωγή παραγράφου φυσικής γλώσσας, βασισμένης
# αποκλειστικά στις ήδη ανακτημένες πηγές — πλέον συμπεριλαμβανομένης
# και της σημασιολογικά αντιστοιχισμένης βιβλιογραφίας (RAG)
# -----------------------------------------------------------------------
def generate_narrative(dilemma, country, sector, detected_dimensions, principles, regulations, actions, literature=None, lang: str = "en", risk=None) -> str:
    """Return an LLM-phrased paragraph grounded in the given retrieved sources
    (knowledge-base entries) and, when provided, the semantically matched
    bibliography entries from engine/bibliography.py.

    Raises LLMUnavailable if no provider/key is configured or the call fails.
    """
    user_prompt = _build_narrative_prompt(dilemma, country, sector, detected_dimensions, principles, regulations, actions, literature, risk)
    system_prompt = NARRATIVE_SYSTEM_PROMPT_EL if lang == "el" else NARRATIVE_SYSTEM_PROMPT_EN
    return _strip_preamble(_dispatch(system_prompt, user_prompt))


# Μικρά τοπικά μοντέλα ξεκινούν συχνά με σχόλιο για την ίδια την οδηγία
# (π.χ. «Here is a 120-200 word advisory paragraph:»), που δεν ανήκει στη σύνοψη
_PREAMBLE_RE = re.compile(
    r"^\s*(here is|here's|below is|sure[,!]?|ορίστε|παρακάτω|ακολουθεί)[^\n]{0,100}:\s*\n*",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    return _PREAMBLE_RE.sub("", text, count=1).strip()


# -----------------------------------------------------------------------
# Δημόσια συνάρτηση: ζητά από το LLM να ταξινομήσει απευθείας το δίλημμα
# στις πέντε ηθικές διαστάσεις, επιστρέφοντας δομημένο JSON
# -----------------------------------------------------------------------
def classify_dilemma(dilemma: str) -> dict:
    """Ask the configured LLM to score the dilemma against the five ethical
    dimensions directly. Used only as a fallback when the local TensorFlow
    classifier is not confident about anything (see advisor.py).

    Raises LLMUnavailable if no provider is configured, the call fails, or
    the response cannot be parsed as the expected JSON shape.
    """
    raw = _dispatch(CLASSIFY_SYSTEM_PROMPT, f"Dilemma: {dilemma}")

    # Ανοχή σε τυχόν markdown περιθώρια (```) που προσθέτουν κάποια μοντέλα
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"Could not parse classification response as JSON: {raw[:200]}") from exc

    # Επικύρωση ότι υπάρχουν όλες οι αναμενόμενες κατηγορίες με έγκυρες τιμές
    scores = {}
    for category in CATEGORIES:
        value = parsed.get(category)
        if not isinstance(value, (int, float)):
            raise LLMUnavailable(f"Classification response missing/invalid key '{category}': {parsed}")
        scores[category] = max(0.0, min(1.0, float(value)))
    return scores
