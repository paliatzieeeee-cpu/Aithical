"""
Συστηματική επέκταση του συνόλου εκπαίδευσης μέσω LLM, σε πλέγμα
θέμα × ηθική διάσταση.

Γιατί υπάρχει: η αξιολόγηση έδειξε ότι ο ταξινομητής είχε μάθει
«συντομεύσεις θέματος» (topic shortcuts) — π.χ. προσλήψεις -> fairness,
παιδιά -> non_maleficence, υγεία -> non_maleficence — επειδή στα
χειρόγραφα δεδομένα κάθε θέμα εμφανιζόταν σχεδόν πάντα με την ίδια
ετικέτα. Οι παγωμένες πολυγλωσσικές ενσωματώσεις κωδικοποιούν έντονα το
θέμα ενός κειμένου, οπότε η κεφαλή ταξινόμησης «πιάνεται» από αυτό αντί
για το ηθικό ζήτημα. Μικρές χειροκίνητες προσθήκες μετατόπιζαν απλώς τα
λάθη. Η λύση εδώ είναι ΣΧΕΔΙΑΣΤΙΚΗ: κάθε θέμα παράγεται ρητά με κάθε
ετικέτα, ώστε κανένα θέμα να μη συσχετίζεται πλέον με μία μόνο διάσταση.

Τι ΔΕΝ κάνει: δεν γράφει τίποτα στο model/training_data.py. Όλα τα
υποψήφια πάνε στο data/augmentation_candidates.jsonl και μπαίνουν στο
σύνολο εκπαίδευσης ΜΟΝΟ αφού εγκριθούν ένα-ένα από άνθρωπο μέσω του
model/review_augmentation.py. Αυτό διατηρεί την ίδια αρχή με το
review_log: κανένα αυτόματα παραγόμενο δεδομένο χωρίς ανθρώπινο έλεγχο.

Προστασίες:
  - απόρριψη ακριβών διπλοτύπων (έναντι εκπαίδευσης, υποψηφίων, run)
  - απόρριψη οτιδήποτε μοιάζει με παράδειγμα του παγωμένου held-out
    συνόλου (έλεγχος λεξιλογικής επικάλυψης), ώστε η μέτρηση γενίκευσης
    να μη «μολυνθεί»
  - έλεγχος γλώσσας (ελληνικό κείμενο όταν ζητήθηκε ελληνικό κ.ο.κ.)

Εκτέλεση (από τη ρίζα του project, με ενεργό το venv):
    python -m model.augment_with_llm --dry-run          # δες το prompt, χωρίς κλήσεις
    python -m model.augment_with_llm --max-cells 3      # γρήγορη δοκιμή
    python -m model.augment_with_llm                    # πλήρες πλέγμα
    python -m model.augment_with_llm --langs en         # μόνο αγγλικά

Σημείωση για τα ελληνικά: μικρά τοπικά μοντέλα (π.χ. llama3.1 8B) γράφουν
αισθητά χειρότερα ελληνικά από αγγλικά. Αν χρησιμοποιείς Ollama, είτε
τρέξε με --langs en είτε απόρριψε με άνεση κακά ελληνικά στον έλεγχο.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών και εσωτερικών ενοτήτων
# -----------------------------------------------------------------------
import argparse
import itertools
import json
import os
import random
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

from engine.llm_client import LLMUnavailable, _dispatch
from model.training_data import CATEGORIES, TRAINING_EXAMPLES

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
CANDIDATES_PATH = os.path.join(ROOT, "data", "augmentation_candidates.jsonl")

# Όριο λεξιλογικής επικάλυψης (Jaccard λέξεων) πάνω από το οποίο ένα
# υποψήφιο θεωρείται «πολύ κοντά» σε held-out παράδειγμα και απορρίπτεται.
HELDOUT_SIMILARITY_LIMIT = 0.45
# Αντίστοιχο όριο έναντι της εκπαίδευσης: πολύ κοντινά = πλεονάζοντα.
TRAINING_SIMILARITY_LIMIT = 0.70

# -----------------------------------------------------------------------
# Θέματα (επιχειρηματικά πλαίσια). Σκόπιμα ποικίλα, ώστε κάθε ηθική
# διάσταση να εμφανίζεται σε πολλά διαφορετικά πλαίσια.
# -----------------------------------------------------------------------
TOPICS = {
    "hiring": "recruitment and hiring (job applicants, CV screening, interviews)",
    "education": "schools, children and education (pupils, students, learning apps for kids)",
    "health": "healthcare and wellbeing (patients, clinics, health or fitness apps)",
    "banking": "banking, lending and credit",
    "insurance": "insurance (premiums, claims, underwriting)",
    "retail": "retail, e-commerce and marketing",
    "customer_service": "customer service and chatbots",
    "workplace": "workplace management of existing employees (performance, scheduling, monitoring)",
    "housing": "housing and rentals (landlords, tenants, property platforms)",
    "public_sector": "public sector and security (government services, policing, building security)",
    "transport": "transport, delivery and logistics",
    "media": "media, publishing and online platforms (news, social media, content)",
}

# -----------------------------------------------------------------------
# Σύντομοι ορισμοί των πέντε διαστάσεων — δίνονται στο LLM ώστε να
# παράγει κείμενα όπου το ηθικό ζήτημα είναι σαφώς η ζητούμενη διάσταση.
# -----------------------------------------------------------------------
CATEGORY_DEFINITIONS = {
    "transparency": "people are not told that AI is involved, or cannot get an explanation of how an AI decision or output was produced",
    "fairness": "an AI system treats some groups of people worse than others (the problem is the DIFFERENCE between groups, e.g. by age, gender, ethnicity, language, income, location)",
    "non_maleficence": "an AI system could cause physical, psychological, financial or safety harm to the people it affects, regardless of which group they belong to (e.g. errors, unsafe advice, malfunctions)",
    "accountability": "it is unclear who is responsible for an AI system, its decisions or its mistakes, or nobody oversees it",
    "privacy": "personal data is collected, used, shared, combined, retained or accessed inappropriately or without real consent",
}

# Κριτήρια διάκρισης για τα ζεύγη που μπερδεύονται συχνότερα, τόσο από τα
# μικρά μοντέλα παραγωγής όσο και από τον ίδιο τον ταξινομητή. Δίνονται
# στο prompt όταν μια από τις δύο διαστάσεις του ζεύγους ζητείται.
DISTINCTIONS = [
    ({"fairness", "non_maleficence"},
     "Test for fairness vs non_maleficence: if everyone were treated exactly the same, would there still be a problem? "
     "If NO, it is fairness (unequal treatment between groups). If YES, it is non_maleficence (harm to anyone). "
     "A non_maleficence example must NOT mention any group being treated differently; a fairness example must "
     "clearly compare how different groups are treated."),
    ({"transparency", "accountability"},
     "Test for transparency vs accountability: transparency is about what people are TOLD or can have EXPLAINED to them; "
     "accountability is about WHO is responsible, who approved it, who oversees it or who fixes mistakes. "
     "A transparency example must not be about unclear responsibility; an accountability example must not be "
     "about people not being informed."),
]

LANG_NAMES = {"en": "English", "el": "Greek (Modern Greek, natural business language)"}

SYSTEM_PROMPT = (
    "You write realistic training examples for a text classifier that detects ethical issues in "
    "business uses of AI. Each example is one or two sentences describing a concrete dilemma a "
    "company faces, written the way an employee would ask about it. Output ONLY a JSON array of "
    "strings, with no commentary, no numbering, and no markdown."
)


# -----------------------------------------------------------------------
# Κατασκευή του prompt για ένα κελί του πλέγματος
# -----------------------------------------------------------------------
def build_prompt(topic_key, labels, n, lang):
    topic = TOPICS[topic_key]
    if not labels:
        # Ουδέτερο κελί: συνηθισμένες επιχειρηματικές ερωτήσεις που αφορούν
        # ΑΙ και το ίδιο θέμα, αλλά ΧΩΡΙΣ ηθικό ζήτημα. Είναι τα πιο χρήσιμα
        # αρνητικά παραδείγματα: διδάσκουν ότι η λέξη "AI" ή ένα θέμα όπως
        # "προσλήψεις" δεν αρκούν από μόνα τους για να υπάρχει πρόβλημα.
        all_issues = "\n".join(f"- {c}: {CATEGORY_DEFINITIONS[c]}" for c in CATEGORIES)
        return (
            f"Write {n} different examples in {LANG_NAMES[lang]}.\n\n"
            f"Business context: {topic}.\n\n"
            "Every example is an ordinary, practical question or request from an employee about using AI "
            "in this business context (for example choosing a tool, setting it up, costs, speed, formatting, "
            "training staff, integrating it with existing software).\n\n"
            f"The examples must raise NONE of these ethical issues:\n{all_issues}\n\n"
            "Requirements:\n"
            "- Each example mentions AI and fits the business context above.\n"
            "- Each example describes a different, specific situation.\n"
            "- Do not name real companies or products.\n"
            "- 10 to 35 words each.\n"
            f'Return exactly a JSON array of {n} strings, e.g. ["...", "..."].'
        )
    wanted = "\n".join(f"- {c}: {CATEGORY_DEFINITIONS[c]}" for c in labels)
    unwanted = [c for c in CATEGORIES if c not in labels]
    not_wanted = "\n".join(f"- {c}: {CATEGORY_DEFINITIONS[c]}" for c in unwanted)
    # Κριτήριο διάκρισης μόνο όταν ζητείται η ΜΙΑ από τις δύο διαστάσεις
    # ενός ζεύγους που μπερδεύεται (αν ζητούνται και οι δύο, δεν ισχύει).
    tests = [text for pair, text in DISTINCTIONS if len(pair & set(labels)) == 1]
    distinction = ("\n".join(tests) + "\n\n") if tests else ""
    return (
        f"Write {n} different examples in {LANG_NAMES[lang]}.\n\n"
        f"Business context: {topic}.\n\n"
        f"Every example MUST clearly raise ALL of these ethical issues:\n{wanted}\n\n"
        f"Every example must NOT primarily be about these other issues:\n{not_wanted}\n\n"
        f"{distinction}"
        "Requirements:\n"
        "- Each example describes a different, specific situation (vary the scenario, not just the wording).\n"
        "- Do not name real companies or products.\n"
        "- Do not use the words transparency, fairness, accountability, privacy or harm themselves; "
        "describe the situation so the issue is implied.\n"
        "- 15 to 45 words each.\n"
        f'Return exactly a JSON array of {n} strings, e.g. ["...", "..."].'
    )


# -----------------------------------------------------------------------
# Ανάγνωση της απάντησης του LLM ως λίστας κειμένων (ανθεκτικά σε
# περιττό κείμενο ή markdown γύρω από το JSON)
# -----------------------------------------------------------------------
def parse_examples(raw, n=None):
    """Ανάγνωση της απάντησης ως λίστας κειμένων. Ανθεκτική στα συνηθισμένα
    λάθη μικρών μοντέλων: markdown γύρω από το JSON, δύο ή περισσότεροι
    πίνακες στη σειρά ("Extra data"), κείμενο μετά τον πίνακα, και
    ξεχασμένα κόμματα ανάμεσα σε strings ("Expecting ',' delimiter").
    Κρατά το πολύ n κείμενα, ώστε ένα κελί να μη «φουσκώνει» το πλέγμα."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # 1) Κάθε έγκυρος πίνακας JSON μέσα στο κείμενο
    decoder = json.JSONDecoder()
    items, i = [], 0
    while True:
        start = cleaned.find("[", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, list):
            items.extend(obj)
        i = end

    # 2) Εναλλακτικά: όλα τα strings σε εισαγωγικά (καλύπτει τα χαμένα κόμματα)
    if not any(isinstance(x, (str, dict)) for x in items):
        items = []
        s, e = cleaned.find("["), cleaned.rfind("]")
        region = cleaned[s:e + 1] if s != -1 and e > s else cleaned
        for m in re.findall(r'"((?:[^"\\]|\\.)*)"', region):
            try:
                items.append(json.loads(f'"{m}"'))
            except json.JSONDecodeError:
                items.append(m)

    out = []
    for item in items:
        if isinstance(item, dict):  # μερικά μοντέλα επιστρέφουν αντικείμενα
            item = item.get("text") or item.get("example") or ""
        if isinstance(item, str):
            text = " ".join(item.split())
            if text:
                out.append(text)
    if not out:
        raise ValueError("no JSON array found")
    return out[:n] if n else out


# -----------------------------------------------------------------------
# Βοηθητικά για κανονικοποίηση και λεξιλογική ομοιότητα
# -----------------------------------------------------------------------
def normalize(text):
    return " ".join(text.lower().split())


def token_set(text):
    return set(re.findall(r"\w{3,}", text.lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def has_greek(text):
    return bool(re.search(r"[α-ωΑ-Ωά-ώ]", text))


def load_reviewed():
    path = os.path.join(ROOT, "data", "augmentation_reviewed.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_candidates():
    if not os.path.isfile(CANDIDATES_PATH):
        return []
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# -----------------------------------------------------------------------
# Κατασκευή του πλέγματος: κάθε θέμα × κάθε μεμονωμένη διάσταση, συν
# μερικά τυχαία (αλλά αναπαραγώγιμα) ζεύγη διαστάσεων ανά θέμα
# -----------------------------------------------------------------------
def build_cells(topic_keys, pairs_per_topic, seed, neutral_per_topic=1):
    rng = random.Random(seed)
    all_pairs = list(itertools.combinations(CATEGORIES, 2))
    cells = []
    for topic in topic_keys:
        for cat in CATEGORIES:
            cells.append((topic, [cat]))
        for pair in rng.sample(all_pairs, min(pairs_per_topic, len(all_pairs))):
            cells.append((topic, list(pair)))
        for _ in range(neutral_per_topic):
            cells.append((topic, []))  # ουδέτερο κελί: καμία ηθική διάσταση
    return cells


def main():
    parser = argparse.ArgumentParser(description="Systematic LLM data augmentation (topic x dimension grid).")
    parser.add_argument("--per-cell", type=int, default=3, help="examples requested per grid cell (default 3)")
    parser.add_argument("--pairs-per-topic", type=int, default=2, help="two-label cells per topic (default 2)")
    parser.add_argument("--neutral-per-topic", type=int, default=1, help="neutral (no-issue) cells per topic (default 1)")
    parser.add_argument("--topics", type=str, default="", help="comma-separated topic keys (default: all)")
    parser.add_argument("--langs", type=str, default="en,el", help="languages to alternate, e.g. 'en' or 'en,el'")
    parser.add_argument("--max-cells", type=int, default=0, help="stop after this many cells (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="print one example prompt and exit")
    parser.add_argument("--fill-missing", action="store_true",
                        help="run only grid cells that have no candidates yet (e.g. cells that failed earlier)")
    parser.add_argument("--cells-for", type=str, default="",
                        help="run only single-dimension cells for these labels, e.g. 'fairness,accountability,neutral'")
    args = parser.parse_args()

    topic_keys = [t.strip() for t in args.topics.split(",") if t.strip()] or list(TOPICS)
    unknown = [t for t in topic_keys if t not in TOPICS]
    if unknown:
        raise SystemExit(f"Unknown topics: {unknown}. Available: {', '.join(TOPICS)}")
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    if any(l not in LANG_NAMES for l in langs):
        raise SystemExit("--langs accepts only 'en' and/or 'el'")

    cells = build_cells(topic_keys, args.pairs_per_topic, args.seed, args.neutral_per_topic)

    if args.cells_for:
        # Στοχευμένη συμπλήρωση: μόνο μονά κελιά για τις κατηγορίες που
        # υστερούν (και ουδέτερα αν ζητηθούν) — χωρίς ζεύγη, ώστε να μην
        # ανεβαίνουν ταυτόχρονα και οι ήδη πλεονάζουσες κατηγορίες.
        wanted = [w.strip() for w in args.cells_for.split(",") if w.strip()]
        bad = [w for w in wanted if w not in CATEGORIES and w != "neutral"]
        if bad:
            raise SystemExit(f"Unknown labels in --cells-for: {bad}. Use {', '.join(CATEGORIES)}, neutral")
        cells = []
        for topic in topic_keys:
            for w in wanted:
                cells.append((topic, [] if w == "neutral" else [w]))

    if args.fill_missing:
        # Κελιά χωρίς κανένα υποψήφιο (ούτε εκκρεμές ούτε ήδη ελεγμένο)
        done = {(r.get("topic"), tuple(r.get("labels", []))) for r in load_existing_candidates()}
        done |= {(r.get("topic"), tuple(r.get("labels", []))) for r in load_reviewed()}
        before = len(cells)
        cells = [(t, l) for t, l in cells if (t, tuple(l)) not in done]
        print(f"--fill-missing: {len(cells)} από {before} κελιά δεν έχουν ακόμα υποψήφια.")

    if args.max_cells:
        cells = cells[: args.max_cells]

    if args.dry_run:
        topic, labels = cells[0]
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT + "\n")
        print("USER PROMPT (πρώτο κελί):\n" + build_prompt(topic, labels, args.per_cell, langs[0]))
        print(f"\nΣύνολο κελιών: {len(cells)} -> έως {len(cells) * args.per_cell} υποψήφια.")
        return

    # Late import: το held-out σύνολο ζει στο evaluate_classifier.py
    from model.evaluate_classifier import HELD_OUT_EXAMPLES

    heldout_tokens = [token_set(t) for t, _ in HELD_OUT_EXAMPLES]
    heldout_norm = {normalize(t) for t, _ in HELD_OUT_EXAMPLES}
    training_tokens = [token_set(t) for t, _ in TRAINING_EXAMPLES]
    seen = {normalize(t) for t, _ in TRAINING_EXAMPLES} | {normalize(c["text"]) for c in load_existing_candidates()}

    provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or "(not set)"
    model_name = {
        "ollama": os.environ.get("OLLAMA_MODEL", "llama3.1"),
        "openai": os.environ.get("OPENAI_MODEL", ""),
        "anthropic": os.environ.get("ANTHROPIC_MODEL", ""),
    }.get(provider, "")

    print(f"Πάροχος: {provider} {model_name}".strip())
    print(f"Κελιά: {len(cells)} | ζητούμενα ανά κελί: {args.per_cell} | γλώσσες: {', '.join(langs)}\n")

    stats = {"kept": 0, "duplicate": 0, "near_heldout": 0, "near_training": 0,
             "wrong_language": 0, "bad_length": 0, "failed_cells": 0}
    os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)

    with open(CANDIDATES_PATH, "a", encoding="utf-8") as out:
        for i, (topic, labels) in enumerate(cells, 1):
            lang = langs[(i - 1) % len(langs)]
            label_str = "+".join(labels) or "neutral"
            print(f"[{i}/{len(cells)}] {topic} × {label_str} ({lang}) ... ", end="", flush=True)
            prompt = build_prompt(topic, labels, args.per_cell, lang)
            texts = None
            for attempt in (1, 2):
                try:
                    raw = _dispatch(SYSTEM_PROMPT, prompt)
                    texts = parse_examples(raw, n=args.per_cell)
                    break
                except LLMUnavailable as exc:
                    print("ΑΠΟΤΥΧΙΑ")
                    raise SystemExit(f"\nΟ πάροχος LLM δεν είναι διαθέσιμος: {exc}")
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    # δεύτερη προσπάθεια με πιο αυστηρή υπενθύμιση μορφής
                    prompt = build_prompt(topic, labels, args.per_cell, lang) + (
                        "\n\nIMPORTANT: output ONE valid JSON array of strings and nothing else. "
                        "Separate strings with commas. Do not add any text before or after it."
                    )
            if texts is None:
                stats["failed_cells"] += 1
                print(f"παραλείφθηκε μετά από 2 προσπάθειες ({last_error})")
                continue

            kept_here = 0
            for text in texts:
                norm = normalize(text)
                words = len(text.split())
                if words < 8 or words > 70:
                    stats["bad_length"] += 1
                    continue
                if (lang == "el") != has_greek(text):
                    stats["wrong_language"] += 1
                    continue
                if norm in seen or norm in heldout_norm:
                    stats["duplicate"] += 1
                    continue
                toks = token_set(text)
                if any(jaccard(toks, h) >= HELDOUT_SIMILARITY_LIMIT for h in heldout_tokens):
                    stats["near_heldout"] += 1
                    continue
                if any(jaccard(toks, t) >= TRAINING_SIMILARITY_LIMIT for t in training_tokens):
                    stats["near_training"] += 1
                    continue

                seen.add(norm)
                record = {
                    "text": text,
                    "labels": labels,
                    "topic": topic,
                    "lang": lang,
                    "provider": provider,
                    "model": model_name,
                    "created": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stats["kept"] += 1
                kept_here += 1
            print(f"κρατήθηκαν {kept_here}/{len(texts)}")

    print("\nΣύνοψη:")
    print(f"  Νέα υποψήφια για έλεγχο:        {stats['kept']}")
    print(f"  Απορρίφθηκαν ως διπλότυπα:       {stats['duplicate']}")
    print(f"  Απορρίφθηκαν (κοντά σε held-out): {stats['near_heldout']}")
    print(f"  Απορρίφθηκαν (κοντά σε εκπαίδευση): {stats['near_training']}")
    print(f"  Λάθος γλώσσα / μήκος:            {stats['wrong_language']} / {stats['bad_length']}")
    print(f"  Κελιά που απέτυχαν:              {stats['failed_cells']}")
    print(f"\nΤα υποψήφια είναι στο {CANDIDATES_PATH}")
    if stats["failed_cells"]:
        print("Για να ξαναδοκιμάσεις μόνο τα κελιά που απέτυχαν: python -m model.augment_with_llm --fill-missing "
              f"--langs {','.join(langs)}")
    print("Επόμενο βήμα: python -m model.review_augmentation")


if __name__ == "__main__":
    main()
