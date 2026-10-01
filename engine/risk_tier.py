"""
EU AI Act risk-tier estimation: places a dilemma on the four-level "risk
pyramid" of Regulation (EU) 2024/1689 (the AI Act):

    unacceptable  -> prohibited practices            (Art. 5)
    high          -> high-risk systems               (Art. 6, Annex I & III)
    limited       -> transparency obligations        (Art. 50)
    minimal       -> no specific obligations         (voluntary codes, Art. 95)

Like the rest of the advisory engine, this is deliberately rule-based and
traceable rather than a free-form LLM judgement: each tier is described by
a small set of "use cases", each tied to the specific article or Annex
point it comes from. A dilemma is scored against every use case using two
signals:

    1. bilingual keyword stems (en/el), accent-insensitive, and
    2. semantic similarity between the dilemma and a short English
       description of the use case, using the same multilingual
       sentence-transformers model as the classifier and bibliography
       (so Greek dilemmas match English descriptions too).

The highest tier with a use case scoring above its threshold wins; with no
match the system defaults to "minimal". The result is an *indication* for
the user to verify, never a legal classification — the report says so.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import re
import unicodedata

import numpy as np

from model.ethics_classifier import embed_texts

# -----------------------------------------------------------------------
# Ρυθμίσεις: σειρά βαθμίδων (από την υψηλότερη), κατώφλια ανά βαθμίδα
# -----------------------------------------------------------------------
TIER_ORDER = ("unacceptable", "high", "limited", "minimal")

# Οι απαγορευμένες πρακτικές απαιτούν ισχυρότερη ένδειξη: μια λανθασμένη
# ένδειξη «απαγορεύεται» είναι πολύ πιο ανησυχητική για τον χρήστη από
# μια λανθασμένη ένδειξη «περιορισμένος κίνδυνος».
TIER_THRESHOLDS = {"unacceptable": 0.6, "high": 0.5, "limited": 0.5}

# Το πολυγλωσσικό μοντέλο δίνει ομοιότητα ~0.5 σχεδόν σε κάθε ζεύγος
# κειμένων «επιχείρηση + ΑΙ», οπότε η απόλυτη τιμή δεν λέει πολλά.
# Χρησιμοποιείται αντί αυτής το *περιθώριο*: πόσο ξεχωρίζει μια περίπτωση
# από τον μέσο όρο όλων. Κάτω από SEM_FLOOR θεωρείται θόρυβος, στο SEM_CEIL
# ισχυρή αντιστοίχιση· η σημασιολογική βαθμολογία μόνη της φτάνει έως SEM_MAX.
SEM_FLOOR = 0.10
SEM_CEIL = 0.24
SEM_MAX = 0.75
SECTOR_BOOST = 0.05

# -----------------------------------------------------------------------
# Περιπτώσεις χρήσης ανά βαθμίδα, με την ακριβή νομική τους βάση.
# `keywords`: ρίζες λέξεων, πεζά και χωρίς τόνους (το κείμενο του
# διλήμματος κανονικοποιείται με τον ίδιο τρόπο πριν τη σύγκριση).
# `combos`: συνδυασμοί ομάδων λέξεων· ένας συνδυασμός ταιριάζει όταν κάθε
# ομάδα του έχει τουλάχιστον μία λέξη οπουδήποτε στο κείμενο (π.χ.
# «συναίσθημα» + «εργαζόμενοι»). Πιάνει διατυπώσεις που μια σταθερή φράση
# όπως «employees' emotions» χάνει, π.χ. «analyzes employees' facial
# expressions ... to infer their emotions».
# `prototype`: σύντομη αγγλική περιγραφή για τη σημασιολογική σύγκριση.
# `sectors`: λειτουργίες επιχείρησης όπου η περίπτωση είναι πιθανότερη.
# -----------------------------------------------------------------------

# Κοινές ομάδες λέξεων για τους συνδυασμούς
_EMOTION_TERMS = [
    "emotion", "emotional state", "stress level", "stressed", "mood", "feelings", "facial expression",
    "tone of voice", "voice tone", "affect recognition", "anxiety", "frustration", "anger",
    "συναισθ", "αγχος", "αγχους", "στρες", "διαθεση", "ψυχολογικη κατασταση",
    "εκφρασεις του προσωπου", "εκφρασεις προσωπου", "εκφραση του προσωπου", "τονος φωνης",
    "τονο φωνης", "τονο της φωνης",
]
_WORK_EDU_TERMS = [
    "employee", "worker", "workplace", "staff", "personnel", "at work", "colleague",
    "student", "pupil", "classroom", "school", "university", "learner",
    "εργαζομεν", "υπαλληλ", "προσωπικο", "χωρο εργασιας", "χωρο της εργασιας", "στη δουλεια",
    "μαθητ", "φοιτητ", "σχολει", "πανεπιστημ", "σχολικη ταξη",
]

USE_CASES = [
    # ---------------- Απαγορευμένες πρακτικές — Άρθρο 5 ----------------
    {
        "id": "social_scoring", "tier": "unacceptable", "ref": "Art. 5(1)(c)",
        "label": {"en": "Social scoring of people", "el": "Κοινωνική βαθμολόγηση ατόμων"},
        "keywords": ["social scor", "social credit", "trustworthiness score", "citizen score",
                     "κοινωνικη βαθμολογ", "κοινωνικο πιστω", "βαθμολογια αξιοπιστιας πολιτ"],
        "combos": [[["score", "rating", "rank", "βαθμολογ", "κατατα"],
                    ["social behaviour", "social behavior", "personal characteristics", "personality traits",
                     "lifestyle", "κοινωνικη συμπεριφορα", "προσωπικα χαρακτηριστικα", "τροπο ζωης"]]],
        "prototype": "Scoring or ranking people based on their social behaviour or personal traits, "
                     "leading to unfavourable treatment in unrelated contexts (social scoring).",
    },
    {
        "id": "manipulation", "tier": "unacceptable", "ref": "Art. 5(1)(a)",
        "label": {"en": "Subliminal or manipulative techniques", "el": "Υποσυνείδητες ή χειριστικές τεχνικές"},
        "keywords": ["subliminal", "manipulative", "manipulate users", "manipulate customers", "manipulate people", "dark pattern", "deceptive techni",
                     "υποσυνειδ", "χειραγωγ", "παραπλανητικ τεχνικ"],
        "prototype": "AI that uses subliminal, manipulative or deceptive techniques to distort people's "
                     "behaviour and push them into decisions that cause them significant harm.",
    },
    {
        "id": "exploit_vulnerable", "tier": "unacceptable", "ref": "Art. 5(1)(b)",
        "label": {"en": "Exploiting vulnerabilities (age, disability, social situation)",
                  "el": "Εκμετάλλευση ευαλωτότητας (ηλικία, αναπηρία, κοινωνική κατάσταση)"},
        "keywords": ["exploit vulnerab", "target children", "targeting children", "vulnerable elderly",
                     "εκμεταλλευ ευαλωτ", "στοχευ παιδια", "ευαλωτους ηλικιωμεν"],
        "combos": [[["children", "minors", "elderly", "disabled", "disability", "financially vulnerable",
                     "παιδι", "ανηλικ", "ηλικιωμεν", "αναπηρ", "ευαλωτ"],
                    ["exploit", "manipulat", "pressure them", "addict", "εκμεταλλευ", "χειραγωγ", "εθισ"]]],
        "prototype": "AI that exploits the vulnerabilities of children, elderly or disabled people or people "
                     "in financial hardship to materially distort their behaviour in a harmful way.",
    },
    {
        "id": "emotion_work_edu", "tier": "unacceptable", "ref": "Art. 5(1)(f)",
        "label": {"en": "Emotion recognition at work or in education",
                  "el": "Αναγνώριση συναισθημάτων στην εργασία ή την εκπαίδευση"},
        "keywords": ["employees' emotion", "employee emotion", "emotions of employees", "emotions of students",
                     "students' emotion", "mood of employees", "facial expressions of employees",
                     "συναισθηματα εργαζομεν", "συναισθηματα των εργαζομεν",
                     "συναισθηματα μαθητ", "συναισθηματα φοιτητ"],
        "combos": [[_EMOTION_TERMS, _WORK_EDU_TERMS]],
        # Όταν τα συναισθήματα ανήκουν ρητά σε πελάτες/ασθενείς, η αναφορά σε
        # «εργαζόμενους» είναι απλώς το πλαίσιο (π.χ. call center), όχι το
        # υποκείμενο της αναγνώρισης — δεν εμπίπτει στο Άρθρο 5(1)(f)
        "combo_exclude": ["emotions of customers", "emotions of clients", "emotions of callers",
                          "emotions of patients", "emotions of visitors", "customers' emotion", "customer emotion",
                          "clients' emotion", "callers' emotion", "patients' emotion", "customer sentiment",
                          "customer mood", "συναισθηματα των πελατ", "συναισθηματα πελατ",
                          "συναισθηματα των ασθεν", "συναισθηματα ασθεν", "συναισθηματα των επισκεπτ"],
        "prototype": "Using AI to infer or recognise the emotions of employees in the workplace or of "
                     "students in schools and universities, for example from their faces or voices.",
        "sectors": ["Human Resources", "Education"],
    },
    {
        "id": "biometric_categorisation", "tier": "unacceptable", "ref": "Art. 5(1)(g)",
        "label": {"en": "Biometric categorisation by sensitive traits",
                  "el": "Βιομετρική κατηγοριοποίηση βάσει ευαίσθητων χαρακτηριστικών"},
        "keywords": ["infer sexual orientation", "infer religion", "infer race", "infer political",
                     "predict sexual orientation", "συμπερανει σεξουαλικο προσανατολ", "συμπερανει θρησκευμα",
                     "σεξουαλικο προσανατολισμο απο", "θρησκευμα απο το προσωπο"],
        "combos": [[["face", "facial", "biometric", "voice", "προσωπ", "βιομετρικ", "φωνη"],
                    ["infer", "predict", "categoris", "categoriz", "classify", "συμπερα", "κατηγοριοπ", "ταξινομ"],
                    ["sexual orientation", "religio", "political opinion", "political views", "race", "trade union",
                     "σεξουαλικο προσανατολ", "θρησκ", "πολιτικες πεποιθησ", "πολιτικες αποψ", "φυλη", "συνδικαλ"]]],
        "prototype": "Categorising people from their biometric data, such as face images, to infer their "
                     "race, political opinions, religion or sexual orientation.",
    },
    {
        "id": "face_scraping", "tier": "unacceptable", "ref": "Art. 5(1)(e)",
        "label": {"en": "Untargeted scraping of facial images", "el": "Μη στοχευμένη συλλογή εικόνων προσώπων"},
        "keywords": ["scrape face", "scraping face", "scrape facial", "scraping facial", "facial images from the internet",
                     "cctv footage to build", "facial recognition database",
                     "συλλογη εικονων προσωπ", "βαση δεδομενων αναγνωρισης προσωπ"],
        "prototype": "Building or expanding a facial recognition database by untargeted scraping of facial "
                     "images from the internet or CCTV footage.",
    },
    {
        "id": "realtime_rbi", "tier": "unacceptable", "ref": "Art. 5(1)(h)",
        "label": {"en": "Real-time remote biometric identification in public (law enforcement)",
                  "el": "Βιομετρική ταυτοποίηση εξ αποστάσεως σε πραγματικό χρόνο σε δημόσιους χώρους (επιβολή του νόμου)"},
        "keywords": ["real-time facial recognition", "live facial recognition", "facial recognition in public",
                     "αναγνωριση προσωπου σε πραγματικο χρονο", "αναγνωριση προσωπων σε δημοσιους χωρους"],
        "prototype": "Real-time remote biometric identification such as live facial recognition of people in "
                     "publicly accessible spaces for law enforcement purposes.",
    },
    {
        "id": "predictive_policing", "tier": "unacceptable", "ref": "Art. 5(1)(d)",
        "label": {"en": "Predicting crime risk from profiling alone",
                  "el": "Πρόβλεψη εγκληματικότητας αποκλειστικά βάσει κατάρτισης προφίλ"},
        "keywords": ["predict who will commit", "predictive policing", "likely to commit a crime", "criminal risk profil",
                     "προβλεψη εγκληματ", "πιθανοτητα να διαπραξ"],
        "combos": [[["predict", "forecast", "risk score", "likelihood", "προβλεψ", "πιθανοτητα"],
                    ["commit a crime", "commit crimes", "criminal offence", "reoffend", "criminal behaviour",
                     "εγκλημ", "παραβατικ", "υποτροπ"],
                    ["individual", "person", "profil", "personality", "ατομ", "προσωπο", "προφιλ", "προσωπικοτητ"]]],
        "prototype": "Predicting the risk that an individual will commit a crime based solely on profiling "
                     "or on their personality traits and characteristics.",
    },

    # ---------------- Υψηλού κινδύνου — Παράρτημα III ----------------
    {
        "id": "employment", "tier": "high", "ref": "Annex III, point 4",
        "label": {"en": "Employment & worker management", "el": "Απασχόληση & διαχείριση εργαζομένων"},
        "keywords": ["recruit", "hiring", "job applicant", "candidate", "screen cv", "screening cv", "cvs",
                     "resumes", "résumé", "shortlist", "job interview", "employee promotion", "promotion decision", "dismiss", "termination of employment", "fire employee", "firing",
                     "performance evaluation", "evaluate employee", "monitor employee", "task allocation",
                     "προσληψ", "υποψηφι", "βιογραφικ", "συνεντευξ", "προαγωγ", "απολυσ",
                     "αξιολογηση εργαζομεν", "αξιολογηση αποδοσης", "παρακολουθηση εργαζομεν", "επιλογη προσωπικου"],
        "prototype": "AI used to recruit or select job candidates, filter CVs, evaluate applicants, or make "
                     "decisions on promotion, termination, task allocation or monitoring of employees.",
        "sectors": ["Human Resources"],
    },
    {
        "id": "credit", "tier": "high", "ref": "Annex III, point 5(b)",
        "label": {"en": "Creditworthiness & credit scoring", "el": "Πιστοληπτική ικανότητα & βαθμολόγηση πιστοληπτικής ικανότητας"},
        "keywords": ["credit scor", "creditworth", "loan approv", "loan application", "approve loans", "mortgage", "lending decision",
                     "πιστοληπτικ", "δανει", "στεγαστικ", "πιστωτικη βαθμολογ"],
        "prototype": "AI that evaluates the creditworthiness of individuals or sets their credit score, "
                     "for example to approve or reject loan or mortgage applications.",
        "sectors": ["Finance"],
    },
    {
        "id": "insurance", "tier": "high", "ref": "Annex III, point 5(c)",
        "label": {"en": "Life & health insurance pricing", "el": "Τιμολόγηση ασφάλισης ζωής & υγείας"},
        "keywords": ["life insurance", "health insurance", "insurance premium", "insurance pricing", "underwriting",
                     "ασφαλιση ζωης", "ασφαλιση υγειας", "ασφαλιστρ"],
        "prototype": "An insurance company using AI to assess the risk of individual customers and set the premiums of their life or health insurance policies.",
        "sectors": ["Finance", "Healthcare"],
    },
    {
        "id": "public_benefits", "tier": "high", "ref": "Annex III, point 5(a), (d)",
        "label": {"en": "Access to public benefits & emergency services",
                  "el": "Πρόσβαση σε δημόσιες παροχές & υπηρεσίες έκτακτης ανάγκης"},
        "keywords": ["welfare", "social benefit", "public assistance", "eligibility for benefits", "emergency call",
                     "triage", "dispatch ambulance",
                     "επιδομ", "κοινωνικες παροχ", "επιλεξιμοτητα", "κλησεις εκτακτης αναγκης", "διαλογη ασθενων"],
        "prototype": "AI used by authorities to decide eligibility for public benefits and services, or to "
                     "classify emergency calls and prioritise emergency response or patient triage.",
        "sectors": ["Healthcare", "Public Sector"],
    },
    {
        "id": "education", "tier": "high", "ref": "Annex III, point 3",
        "label": {"en": "Education & vocational training", "el": "Εκπαίδευση & επαγγελματική κατάρτιση"},
        "keywords": ["admissions", "admission to", "grade student", "grading", "exams", "examination", "assess students", "evaluate students",
                     "proctor", "cheating", "learning outcome", "student placement",
                     "εισαγωγη φοιτητ", "εισαγωγη μαθητ", "βαθμολογηση μαθητ", "βαθμολογηση φοιτητ", "εξετασεις μαθητ", "εξετασεις φοιτητ", "πανελλαδικ", "γραπτα",
                     "αξιολογηση μαθητ", "αξιολογηση φοιτητ", "επιτηρηση εξετασ", "αντιγραφη στις εξετασ"],
        "prototype": "AI that decides admission to schools or universities, grades exams, evaluates "
                     "learning outcomes or monitors students for cheating during tests.",
        "sectors": ["Education"],
    },
    {
        "id": "biometric_id", "tier": "high", "ref": "Annex III, point 1",
        "label": {"en": "Biometric identification & emotion recognition",
                  "el": "Βιομετρική ταυτοποίηση & αναγνώριση συναισθημάτων"},
        "keywords": ["facial recognition", "face recognition", "biometric", "fingerprint", "voice print",
                     "emotion detect", "emotion recogn",
                     "αναγνωριση προσωπ", "βιομετρικ", "δακτυλικ αποτυπ", "αναγνωριση συναισθ"],
        # Αναγνώριση συναισθημάτων σε οποιοδήποτε άλλο πλαίσιο (Παράρτημα III, 1(c))
        "combos": [[_EMOTION_TERMS,
                    ["analys", "analyz", "detect", "recogni", "infer", "monitor", "track",
                     "αναλυ", "ανιχν", "αναγνωρ", "συμπερα", "παρακολουθ"]]],
        "prototype": "Remote biometric identification of people, for example facial recognition, or AI "
                     "systems that recognise emotions or categorise people from biometric data.",
        "sectors": ["Hospitality & Food Service"]
    },
    {
        "id": "critical_infra", "tier": "high", "ref": "Annex III, point 2",
        "label": {"en": "Critical infrastructure", "el": "Υποδομές ζωτικής σημασίας"},
        "keywords": ["power grid", "electricity grid", "water supply", "gas supply", "traffic control",
                     "road traffic", "heating supply", "critical infrastructure", "digital infrastructure",
                     "ηλεκτρικο δικτυο", "υδρευσ", "παροχη νερου", "φυσικο αεριο", "ελεγχος κυκλοφοριας",
                     "υποδομες ζωτικης"],
        "prototype": "AI used as a safety component in the management of critical infrastructure such as "
                     "electricity, water, gas or heating supply, road traffic or digital infrastructure.",
    },
    {
        "id": "law_enforcement", "tier": "high", "ref": "Annex III, point 6",
        "label": {"en": "Law enforcement", "el": "Επιβολή του νόμου"},
        "keywords": ["police", "law enforcement", "criminal investigation", "evidence reliability", "polygraph",
                     "αστυνομ", "επιβολη του νομου", "ποινικη ερευνα", "ανιχνευτη ψευδους"],
        "prototype": "AI used by police or law enforcement to assess victims or suspects, evaluate evidence "
                     "or support criminal investigations.",
        "sectors": ["Public Sector"]
    },
    {
        "id": "migration", "tier": "high", "ref": "Annex III, point 7",
        "label": {"en": "Migration, asylum & border control", "el": "Μετανάστευση, άσυλο & έλεγχος συνόρων"},
        "keywords": ["asylum", "visa application", "border control", "migrant", "residence permit",
                     "ασυλο", "βιζα", "ελεγχος συνορ", "μεταναστ", "αδεια διαμονης"],
        "prototype": "AI used to examine asylum, visa or residence permit applications, assess migration "
                     "risks, or detect and identify people at border control.",
        "sectors": ["Public Sector"]
    },
    {
        "id": "justice_democracy", "tier": "high", "ref": "Annex III, point 8",
        "label": {"en": "Justice & democratic processes", "el": "Δικαιοσύνη & δημοκρατικές διαδικασίες"},
        "keywords": ["judges", "assist judge", "court decision", "judicial", "sentencing", "dispute resolution",
                     "influence voters", "election campaign", "voting behaviour",
                     "δικαστ", "δικαστικ αποφασ", "επιλυση διαφορων", "ψηφοφορ", "εκλογ"],
        "prototype": "AI that assists judges in researching and applying the law to facts, or that is "
                     "intended to influence the outcome of elections or the voting behaviour of citizens.",
        "sectors": ["Legal Services", "Public Sector"]
    },
    # ---------------- Υψηλού κινδύνου — Παράρτημα I (ασφάλεια προϊόντων) ----------------
    {
        "id": "safety_component", "tier": "high", "ref": "Art. 6(1), Annex I",
        "label": {"en": "Safety component of a regulated product (e.g. medical device, vehicle, machinery)",
                  "el": "Κατασκευαστικό στοιχείο ασφάλειας ρυθμιζόμενου προϊόντος (π.χ. ιατροτεχνολογικό, όχημα, μηχάνημα)"},
        "keywords": ["medical device", "diagnos", "radiolog", "autonomous vehicle", "self-driving",
                     "machinery", "industrial robot", "toys", "aviation",
                     "ιατροτεχνολογικ", "διαγνωσ", "ακτινολογ", "αυτονομο οχημα", "αυτοοδηγουμεν",
                     "μηχανημα", "βιομηχανικο ρομποτ"],
        "prototype": "Medical AI software that analyses scans, X-rays or patient data to diagnose disease, or AI "
                     "that is a safety component of a regulated product such as a vehicle, machinery or a toy.",
        "sectors": ["Healthcare"],
    },

    # ---------------- Περιορισμένου κινδύνου — Άρθρο 50 ----------------
    {
        "id": "chatbot", "tier": "limited", "ref": "Art. 50(1)",
        "label": {"en": "AI interacting directly with people (e.g. chatbots)",
                  "el": "ΑΙ σε άμεση αλληλεπίδραση με ανθρώπους (π.χ. chatbots)"},
        "keywords": ["chatbot", "chat bot", "virtual assistant", "conversational", "voice assistant",
                     "customer service bot", "talk to customers",
                     "εικονικος βοηθ", "ψηφιακος βοηθ", "συνομιλι", "φωνητικος βοηθ"],
        "prototype": "A chatbot or virtual assistant that talks directly with customers or users, who may "
                     "not realise they are interacting with an AI system.",
        "sectors": ["Customer Support", "Marketing & Sales", "Hospitality & Food Service"],
    },
    {
        "id": "synthetic_content", "tier": "limited", "ref": "Art. 50(2), 50(4)",
        "label": {"en": "AI-generated or manipulated content (incl. deepfakes)",
                  "el": "Περιεχόμενο που παράγεται ή αλλοιώνεται από ΑΙ (συμπ. deepfakes)"},
        "keywords": ["deepfake", "deep fake", "synthetic image", "synthetic video", "synthetic voice",
                     "ai-generated", "ai generated", "generate images", "generate articles", "generate text",
                     "voice clon", "generative ai", "write articles",
                     "συνθετικ", "παραγεται απο ΑΙ", "παραγωγη εικονων", "παραγωγη κειμεν", "κλωνοποιηση φωνης",
                     "γενετικη ΑΙ", "παραγωγικη ΑΙ",
                     # Γνωστά εργαλεία παραγωγής μουσικής, φωνής και εικόνας
                     "suno", "udio", "elevenlabs", "midjourney", "stable diffusion", "dall-e", "dall·e",
                     "ai music", "ai-generated music", "ai song", "ai art", "image generator", "music generator",
                     "γεννητρια εικονων", "γεννητρια μουσικης"],
        # Μουσική, ήχος, εικόνα ή βίντεο που δημιουργείται: λέξη μέσου + λέξη
        # δημιουργίας οπουδήποτε στο κείμενο (π.χ. «creating some music tracks»)
        "combos": [[["music", "song", "track", "audio", "vocals", "voice", "melody", "soundtrack", "podcast",
                     "artwork", "illustration", "image", "video",
                     "τραγουδ", "μουσικ", "κομματ", "ηχογραφ", "φωνη", "μελωδ", "εικονογραφ", "εικον", "βιντεο"],
                    ["generat", "create", "creating", "compose", "composing", "produce", "producing",
                     "δημιουργ", "παραγ", "συνθε", "φτιαχν", "φτιαξ"]]],
        "prototype": "Using generative AI to create or manipulate images, audio, music, video or text, such as "
                     "AI-generated songs, deepfakes or AI-written articles published to the public.",
        # Η περιγραφή μοιάζει με κάθε κείμενο για εικόνες ή ήχο (π.χ. απλή
        # κατηγοριοποίηση φωτογραφιών), οπότε απαιτείται ρητή ένδειξη παραγωγής
        "requires_keyword": True,
        "sectors": ["Marketing & Sales", "Arts & Creative Industries"],
    },
    {
        "id": "emotion_disclosure", "tier": "limited", "ref": "Art. 50(3)",
        "label": {"en": "Emotion recognition / biometric categorisation (disclosure duty)",
                  "el": "Αναγνώριση συναισθημάτων / βιομετρική κατηγοριοποίηση (υποχρέωση ενημέρωσης)"},
        "keywords": ["sentiment of customers", "customer emotion", "συναισθηματα πελατ"],
        "combos": [[_EMOTION_TERMS + ["sentiment"],
                    ["customer", "client", "caller", "guest", "πελατ", "επισκεπτ"]]],
        # Η περιγραφή μοιάζει σημασιολογικά με κάθε κείμενο για «πελάτες»,
        # οπότε η ομοιότητα μόνη της δεν αρκεί για να δοθεί αυτή η ένδειξη
        "requires_keyword": True,
        "prototype": "Analysing the emotions or sentiment of customers from their voice or face, where "
                     "people must be informed that such a system is in use.",
        "sectors": ["Customer Support", "Marketing & Sales"],
    },
]

# -----------------------------------------------------------------------
# Περιγραφή κάθε βαθμίδας και βασικές υποχρεώσεις, στις δύο γλώσσες
# -----------------------------------------------------------------------
TIER_INFO = {
    "unacceptable": {
        "verdict": {
            "en": "Possibly a PROHIBITED practice under the EU AI Act (unacceptable risk).",
            "el": "Πιθανώς ΑΠΑΓΟΡΕΥΜΕΝΗ πρακτική βάσει του Κανονισμού ΤΝ της ΕΕ (μη αποδεκτός κίνδυνος).",
        },
        "obligations": {
            "en": [
                "Prohibited practices may not be placed on the EU market, put into service or used (Art. 5), in force since 2 February 2025.",
                "Fines can reach €35 million or 7% of worldwide annual turnover (Art. 99(3)).",
                "Check whether a narrow exception in Art. 5 applies (e.g. medical or safety reasons for emotion recognition).",
            ],
            "el": [
                "Οι απαγορευμένες πρακτικές δεν επιτρέπεται να διατίθενται στην αγορά της ΕΕ, να τίθενται σε λειτουργία ή να χρησιμοποιούνται (Άρθρο 5), ισχύει από 2 Φεβρουαρίου 2025.",
                "Τα πρόστιμα φτάνουν τα 35 εκατ. € ή το 7% του παγκόσμιου ετήσιου κύκλου εργασιών (Άρθρο 99(3)).",
                "Ελέγξτε αν ισχύει κάποια στενή εξαίρεση του Άρθρου 5 (π.χ. ιατρικοί λόγοι ή λόγοι ασφάλειας για την αναγνώριση συναισθημάτων).",
            ],
        },
    },
    "high": {
        "verdict": {
            "en": "Likely a HIGH-RISK AI system under the EU AI Act.",
            "el": "Πιθανώς σύστημα ΤΝ ΥΨΗΛΟΥ ΚΙΝΔΥΝΟΥ βάσει του Κανονισμού ΤΝ της ΕΕ.",
        },
        "obligations": {
            "en": [
                "Providers: risk management system, data governance, technical documentation, logging, accuracy & robustness, conformity assessment, CE marking and EU database registration (Arts. 9–17, 43, 49).",
                "Deployers: use per instructions, human oversight by competent staff, monitor operation, keep logs, inform affected people and workers (Art. 26).",
                "Public bodies and some private deployers (e.g. credit, insurance) must run a fundamental-rights impact assessment (Art. 27).",
                "Affected people have a right to an explanation of decisions taken with the system's help (Art. 86).",
            ],
            "el": [
                "Πάροχοι: σύστημα διαχείρισης κινδύνων, διακυβέρνηση δεδομένων, τεχνική τεκμηρίωση, καταγραφή, ακρίβεια & στιβαρότητα, αξιολόγηση συμμόρφωσης, σήμανση CE και καταχώριση στη βάση δεδομένων της ΕΕ (Άρθρα 9–17, 43, 49).",
                "Φορείς εφαρμογής: χρήση σύμφωνα με τις οδηγίες, ανθρώπινη εποπτεία από ικανό προσωπικό, παρακολούθηση λειτουργίας, τήρηση αρχείων, ενημέρωση θιγόμενων προσώπων και εργαζομένων (Άρθρο 26).",
                "Δημόσιοι φορείς και ορισμένοι ιδιώτες (π.χ. πίστωση, ασφάλιση) οφείλουν να διενεργούν εκτίμηση επιπτώσεων στα θεμελιώδη δικαιώματα (Άρθρο 27).",
                "Τα θιγόμενα πρόσωπα έχουν δικαίωμα εξήγησης για αποφάσεις που λήφθηκαν με τη βοήθεια του συστήματος (Άρθρο 86).",
            ],
        },
    },
    "limited": {
        "verdict": {
            "en": "Likely LIMITED risk: transparency obligations apply under the EU AI Act.",
            "el": "Πιθανώς ΠΕΡΙΟΡΙΣΜΕΝΟΣ κίνδυνος: ισχύουν υποχρεώσεις διαφάνειας βάσει του Κανονισμού ΤΝ της ΕΕ.",
        },
        "obligations": {
            "en": [
                "Tell people they are interacting with an AI system, unless it is obvious (Art. 50(1)).",
                "Mark AI-generated audio, image, video or text in a machine-readable way (Art. 50(2)).",
                "Disclose deepfakes and AI-generated text published to inform the public (Art. 50(4)).",
            ],
            "el": [
                "Ενημερώστε τους ανθρώπους ότι αλληλεπιδρούν με σύστημα ΤΝ, εκτός αν είναι προφανές (Άρθρο 50(1)).",
                "Σημάνετε σε μηχαναγνώσιμη μορφή τον ήχο, την εικόνα, βίντεο ή κείμενο που παράγεται από ΤΝ (Άρθρο 50(2)).",
                "Γνωστοποιήστε τα deepfakes και τα κείμενα ΤΝ που δημοσιεύονται για την ενημέρωση του κοινού (Άρθρο 50(4)).",
            ],
        },
    },
    "minimal": {
        "verdict": {
            "en": "Likely MINIMAL risk: no specific AI Act obligations beyond AI literacy.",
            "el": "Πιθανώς ΕΛΑΧΙΣΤΟΣ κίνδυνος: καμία ειδική υποχρέωση από τον Κανονισμό ΤΝ πέρα από τον γραμματισμό ΤΝ.",
        },
        "obligations": {
            "en": [
                "Ensure sufficient AI literacy of staff who operate or use the system (Art. 4).",
                "Voluntary codes of conduct are encouraged (Art. 95); other laws (e.g. GDPR, consumer, anti-discrimination law) still apply.",
            ],
            "el": [
                "Εξασφαλίστε επαρκή γραμματισμό ΤΝ στο προσωπικό που λειτουργεί ή χρησιμοποιεί το σύστημα (Άρθρο 4).",
                "Ενθαρρύνονται εθελοντικοί κώδικες δεοντολογίας (Άρθρο 95)· άλλοι νόμοι (π.χ. ΓΚΠΔ, προστασία καταναλωτή, νομοθεσία κατά των διακρίσεων) εξακολουθούν να ισχύουν.",
            ],
        },
    },
}

TIER_NAMES = {
    "unacceptable": {"en": "unacceptable-risk (prohibited)", "el": "μη αποδεκτού κινδύνου (απαγορευμένες πρακτικές)"},
    "high": {"en": "high-risk", "el": "υψηλού κινδύνου"},
    "limited": {"en": "limited-risk", "el": "περιορισμένου κινδύνου"},
}

NOTES = {
    "also_high": {
        "en": "Even if an Art. 5 exception applied, the system would still be high-risk: {label} ({ref}).",
        "el": "Ακόμη κι αν ίσχυε κάποια εξαίρεση του Άρθρου 5, το σύστημα θα ήταν υψηλού κινδύνου: {label} ({ref}).",
    },
    "borderline": {
        "en": "Your description is close to the {tier} tier — check carefully whether that tier applies.",
        "el": "Η περιγραφή σας βρίσκεται κοντά στη βαθμίδα {tier} — ελέγξτε προσεκτικά αν ισχύει εκείνη η βαθμίδα.",
    },
    "high_exception": {
        "en": "An Annex III system is not high-risk if it only performs a narrow procedural or preparatory task and does not materially influence decisions (Art. 6(3)) — but it is always high-risk if it profiles natural persons.",
        "el": "Ένα σύστημα του Παραρτήματος III δεν θεωρείται υψηλού κινδύνου αν εκτελεί μόνο στενή διαδικαστική ή προπαρασκευαστική εργασία χωρίς ουσιώδη επίδραση στις αποφάσεις (Άρθρο 6(3)) — είναι όμως πάντα υψηλού κινδύνου όταν καταρτίζει προφίλ φυσικών προσώπων.",
    },
    "high_dates": {
        "en": "Under Art. 113, high-risk obligations apply from 2 August 2026 (Annex III) and 2 August 2027 (Annex I products); check whether later amendments have shifted these dates.",
        "el": "Σύμφωνα με το Άρθρο 113, οι υποχρεώσεις υψηλού κινδύνου εφαρμόζονται από 2 Αυγούστου 2026 (Παράρτημα III) και 2 Αυγούστου 2027 (προϊόντα Παραρτήματος I)· ελέγξτε αν μεταγενέστερες τροποποιήσεις έχουν μετατοπίσει αυτές τις ημερομηνίες.",
    },
    "non_eu": {
        "en": "You selected a jurisdiction outside the EU. The AI Act still applies if the system is placed on the EU market or its output is used in the EU (Art. 2).",
        "el": "Επιλέξατε δικαιοδοσία εκτός ΕΕ. Ο Κανονισμός ΤΝ εφαρμόζεται παρ' όλα αυτά αν το σύστημα διατίθεται στην αγορά της ΕΕ ή τα αποτελέσματά του χρησιμοποιούνται στην ΕΕ (Άρθρο 2).",
    },
    "indicative": {
        "en": "This is an automated indication based on your description, not a legal classification.",
        "el": "Πρόκειται για αυτοματοποιημένη ένδειξη με βάση την περιγραφή σας, όχι για νομική κατάταξη.",
    },
}

# Καθυστερημένη φόρτωση των ενσωματώσεων των περιγραφών (μία φορά ανά διεργασία)
_prototype_embeddings = None


# -----------------------------------------------------------------------
# Κανονικοποίηση κειμένου: πεζά και αφαίρεση τόνων/διαλυτικών, ώστε
# «Προσλήψεις» και «προσληψεις» να ταιριάζουν με την ίδια ρίζα
# -----------------------------------------------------------------------
def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Οι λέξεις-κλειδιά κανονικοποιούνται και μεταγλωττίζονται μία φορά. Η
# αντιστοίχιση γίνεται μόνο στην αρχή λέξης, ώστε π.χ. το «exam» να μην
# ταιριάζει μέσα στο «example»· οι ρίζες εξακολουθούν να καλύπτουν καταλήξεις.
def _compile(words):
    return [re.compile(r"(?<!\w)" + re.escape(_normalize(k))) for k in words]


for _case in USE_CASES:
    _case["_patterns"] = _compile(_case["keywords"])
    _case["_combos"] = [[_compile(group) for group in combo] for combo in _case.get("combos", [])]
    _case["_combo_exclude"] = _compile(_case.get("combo_exclude", []))

# Ένας συνδυασμός σε απαγορευμένη πρακτική μετράει μόνο αν συμφωνεί και η
# σημασιολογική ομοιότητα. Λέξεις όπως «emotion» και «employees» μπορεί να
# συνυπάρχουν σε άσχετο κείμενο (π.χ. «οι εργαζόμενοί μας αναλύουν τα
# συναισθήματα των πελατών»), ενώ η περιγραφή της περίπτωσης όχι.
COMBO_MIN_SEM = 0.3


def _combo_matches(case: dict, norm_text: str) -> int:
    if any(p.search(norm_text) for p in case["_combo_exclude"]):
        return 0
    return sum(
        1 for combo in case["_combos"]
        if all(any(p.search(norm_text) for p in group) for group in combo)
    )


# -----------------------------------------------------------------------
# Σημασιολογικές ομοιότητες με κάθε περιγραφή περίπτωσης χρήσης· επιστρέφει
# None αν το μοντέλο ενσωμάτωσης δεν είναι διαθέσιμο (μόνο λέξεις-κλειδιά)
# -----------------------------------------------------------------------
def _semantic_similarities(dilemma: str):
    global _prototype_embeddings
    try:
        if _prototype_embeddings is None:
            _prototype_embeddings = embed_texts([c["prototype"] for c in USE_CASES])
        query = embed_texts([dilemma])[0]
        matrix = _prototype_embeddings / (np.linalg.norm(_prototype_embeddings, axis=1, keepdims=True) + 1e-8)
        sims = matrix @ (query / (np.linalg.norm(query) + 1e-8))
        return sims - sims.mean()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[risk_tier] semantic matching unavailable, using keywords only: {exc}")
        return None


# -----------------------------------------------------------------------
# Βαθμολογία μίας περίπτωσης χρήσης από τα δύο σήματα
# -----------------------------------------------------------------------
def _score_case(case: dict, norm_text: str, margin, sector: str) -> float:
    hits = sum(1 for pattern in case["_patterns"] if pattern.search(norm_text))

    sem_score = 0.0
    if margin is not None:
        sem_score = float(np.clip((margin - SEM_FLOOR) / (SEM_CEIL - SEM_FLOOR), 0.0, 1.0)) * SEM_MAX

    # Κάθε συνδυασμός που ταιριάζει μετράει όσο δύο ρητές λέξεις-κλειδιά,
    # αφού δένει δύο (ή περισσότερες) ανεξάρτητες ενδείξεις μεταξύ τους
    combos = _combo_matches(case, norm_text)
    if combos and (case["tier"] != "unacceptable" or margin is None or sem_score >= COMBO_MIN_SEM):
        hits += 2 * combos
    kw_score = 0.0 if hits == 0 else (0.6 if hits == 1 else (0.8 if hits == 2 else 0.9))

    if margin is not None:
        # Η ένδειξη «απαγορευμένη πρακτική» δεν δίνεται ποτέ μόνο από
        # σημασιολογική ομοιότητα· χωρίς ρητή λέξη-κλειδί, το σήμα μένει
        # κάτω από το κατώφλι της βαθμίδας (π.χ. ένα deepfake για διαφήμιση
        # μοιάζει σημασιολογικά με «συλλογή εικόνων προσώπων», αλλά δεν είναι)
        # Το ίδιο ισχύει για περιπτώσεις με ρητή σήμανση requires_keyword.
        if (case["tier"] == "unacceptable" or case.get("requires_keyword")) and hits == 0:
            sem_score = min(sem_score, TIER_THRESHOLDS["unacceptable"] - 0.15)

    score = max(kw_score, sem_score)
    # Όταν συμφωνούν και τα δύο σήματα, η ένδειξη είναι ισχυρότερη
    if kw_score > 0 and sem_score > 0.3:
        score += 0.1
    # Η επιλεγμένη λειτουργία επιχείρησης ενισχύει ελαφρά μια ήδη υπαρκτή
    # ένδειξη, αλλά δεν τη δημιουργεί από μόνη της
    if score > 0 and sector in case.get("sectors", []):
        score += SECTOR_BOOST
    return min(score, 1.0)


# -----------------------------------------------------------------------
# Δημόσια συνάρτηση: εκτίμηση της βαθμίδας κινδύνου του AI Act
# -----------------------------------------------------------------------
def assess_risk_tier(dilemma: str, sector: str = "", in_eu: bool = True, lang: str = "en") -> dict:
    """Place the dilemma on the AI Act risk pyramid.

    Returns the winning tier, a confidence in [0, 1], a `position` in
    [0, 1] for the marker inside that tier's band on the pyramid (1 = top
    of the band, i.e. strongest indication), the matched use cases with
    their legal reference, and the key obligations for that tier.
    """
    lang = lang if lang in ("en", "el") else "en"
    norm_text = _normalize(dilemma or "")
    sims = _semantic_similarities(dilemma) if dilemma and dilemma.strip() else None

    scored = []
    for i, case in enumerate(USE_CASES):
        sim = None if sims is None else float(sims[i])
        scored.append((case, _score_case(case, norm_text, sim, sector)))

    # Η υψηλότερη βαθμίδα με αντιστοίχιση πάνω από το κατώφλι της κερδίζει
    tier, confidence = "minimal", 0.0
    for candidate in TIER_ORDER[:-1]:
        best = max((s for c, s in scored if c["tier"] == candidate), default=0.0)
        if best >= TIER_THRESHOLDS[candidate]:
            tier, confidence = candidate, best
            break

    if tier == "minimal":
        # Βεβαιότητα για «ελάχιστο»: όσο πιο αδύναμη η ισχυρότερη ένδειξη
        # υψηλότερης βαθμίδας, τόσο πιο σίγουρα ανήκει εδώ
        strongest_other = max((s for _, s in scored), default=0.0)
        confidence = round(1.0 - strongest_other * 0.8, 3)

    # Αντιστοιχίσεις που εμφανίζονται στον χρήστη: οι ισχυρότερες της
    # βαθμίδας που επιλέχθηκε (κοντά στην κορυφαία), με τη νομική τους βάση
    floor = max(TIER_THRESHOLDS.get(tier, 1.0) - 0.05, confidence - 0.15)
    matches = [
        {
            "id": c["id"],
            "tier": c["tier"],
            "ref": c["ref"],
            "label": c["label"][lang],
            "score": round(s, 3),
        }
        for c, s in sorted(scored, key=lambda cs: -cs[1])
        if c["tier"] == tier and s >= floor
    ][:3]

    # Θέση του δείκτη μέσα στη ζώνη της βαθμίδας στην πυραμίδα: όσο πιο
    # κοντά στο κατώφλι της αμέσως υψηλότερης βαθμίδας βρίσκεται η
    # ισχυρότερη ένδειξή της, τόσο πιο ψηλά μέσα στη ζώνη (0 = βάση, 1 = κορυφή).
    # Για την κορυφή της πυραμίδας, η θέση ακολουθεί απλώς τη βεβαιότητα.
    tier_index = TIER_ORDER.index(tier)
    borderline_with = None
    if tier_index == 0:
        position = confidence
    else:
        above = TIER_ORDER[tier_index - 1]
        best_above = max((s for c, s in scored if c["tier"] == above), default=0.0)
        position = best_above / TIER_THRESHOLDS[above]
        if best_above >= TIER_THRESHOLDS[above] - 0.15:
            borderline_with = above
    position = round(min(max(position, 0.1), 0.9), 3)

    notes = []
    if tier == "unacceptable":
        # Αν ταιριάζει και περίπτωση υψηλού κινδύνου, η εξαίρεση του Άρθρου 5
        # (π.χ. ιατρικοί λόγοι) δεν αρκεί: το σύστημα θα παρέμενε υψηλού κινδύνου
        high_case, high_score = max(((c, s) for c, s in scored if c["tier"] == "high"),
                                    key=lambda cs: cs[1], default=(None, 0.0))
        if high_case and high_score >= TIER_THRESHOLDS["high"]:
            notes.append(NOTES["also_high"][lang].format(label=high_case["label"][lang], ref=high_case["ref"]))
    if borderline_with:
        notes.append(NOTES["borderline"][lang].format(tier=TIER_NAMES[borderline_with][lang]))
    if tier == "high":
        notes += [NOTES["high_exception"][lang], NOTES["high_dates"][lang]]
    if not in_eu:
        notes.append(NOTES["non_eu"][lang])
    notes.append(NOTES["indicative"][lang])

    return {
        "tier": tier,
        "is_high_risk": tier in ("unacceptable", "high"),
        "confidence": round(confidence, 3),
        "position": position,
        "borderline_with": borderline_with,
        "verdict": TIER_INFO[tier]["verdict"][lang],
        "obligations": TIER_INFO[tier]["obligations"][lang],
        "matches": matches,
        "notes": notes,
    }
