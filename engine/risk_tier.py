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
    "bored", "boredom", "unhappy", "happiness", "angry", "sad", "θυμωμεν", "λυπημεν",
    "συναισθ", "αγχος", "αγχους", "αγχων", "στρες", "διαθεση", "ψυχολογικη κατασταση", "βαριουν", "βαρεμαρ",
    "δυσαρεστ",
    "εκφρασεις του προσωπου", "εκφρασεις προσωπου", "εκφραση του προσωπου", "τονος φωνης",
    "τονο φωνης", "τονο της φωνης",
]
_WORK_EDU_TERMS = [
    "employee", "worker", "workplace", "staff", "personnel", "at work", "colleague", "supervisor",
    "call-centre agent", "call centre agent", "call-center agent", "call center agent", "support agent",
    "student", "pupil", "classroom", "school", "university", "learner", "teacher",
    "εργαζομεν", "υπαλληλ", "προσωπικο", "χωρο εργασιας", "χωρο της εργασιας", "στη δουλεια", "προισταμεν",
    "μαθητ", "φοιτητ", "σχολει", "πανεπιστημ", "σχολικη ταξη", "αιθουσες διδασκαλιας", "καθηγητ",
]

USE_CASES = [
    # ---------------- Απαγορευμένες πρακτικές — Άρθρο 5 ----------------
    {
        "id": "social_scoring", "tier": "unacceptable", "ref": "Art. 5(1)(c)",
        "label": {"en": "Social scoring of people", "el": "Κοινωνική βαθμολόγηση ατόμων"},
        "keywords": ["social scor", "social credit", "trustworthiness score", "citizen score",
                     "κοινωνικη βαθμολογ", "κοινωνικο πιστω", "βαθμολογια αξιοπιστιας πολιτ"],
        "combos": [[["score", "rating", "rank", "rate ", "βαθμολογ", "κατατα", "σκορ"],
                    ["social behaviour", "social behavior", "personal characteristics", "personality traits",
                     "lifestyle", "online behaviour", "online behavior", "social media", "payment history",
                     "complaints", "how responsible", "trustworth",
                     "κοινωνικη συμπεριφορα", "κοινωνικη τους συμπεριφορα", "προσωπικα χαρακτηριστικα",
                     "τροπο ζωης", "κοινωνικα δικτυα", "αξιοπιστ"]]],
        "prototype": "Scoring or ranking people based on their social behaviour or personal traits, "
                     "leading to unfavourable treatment in unrelated contexts (social scoring).",
    },
    {
        "id": "manipulation", "tier": "unacceptable", "ref": "Art. 5(1)(a)",
        "label": {"en": "Subliminal or manipulative techniques", "el": "Υποσυνείδητες ή χειριστικές τεχνικές"},
        "keywords": ["subliminal", "manipulative", "manipulate users", "manipulate customers", "manipulate people", "dark pattern", "deceptive techni",
                     "υποσυνειδ", "χειραγωγ", "παραπλανητικ τεχνικ"],
        # Κρυφή ή ασυνείδητη επιρροή + ώθηση σε δαπάνη/απόφαση
        "combos": [[["hidden", "subliminal", "unconscious", "consciously", "without realising", "without realizing",
                     "without noticing", "κρυφ", "ασυνειδ", "χωρις να το καταλαβ"],
                    ["spend", "bet", "gambl", "buy", "purchas", "nudg", "push", "pay more",
                     "ξοδευ", "αγορ", "στοιχημ", "τζογ", "πληρων"]]],
        "prototype": "AI that uses subliminal, manipulative or deceptive techniques to distort people's "
                     "behaviour and push them into decisions that cause them significant harm.",
    },
    {
        "id": "exploit_vulnerable", "tier": "unacceptable", "ref": "Art. 5(1)(b)",
        "label": {"en": "Exploiting vulnerabilities (age, disability, social situation)",
                  "el": "Εκμετάλλευση ευαλωτότητας (ηλικία, αναπηρία, κοινωνική κατάσταση)"},
        "keywords": ["exploit vulnerab", "target children", "targeting children", "vulnerable elderly",
                     "εκμεταλλευ ευαλωτ", "στοχευ παιδια", "ευαλωτους ηλικιωμεν"],
        "combos": [[["children", "child", "kids", "minors", "young people", "elderly", "dementia", "disabled",
                     "disability", "financially vulnerable", "financial distress", "in debt", "vulnerable",
                     "addicts", "gambling addict", "εθισμεν",
                     "παιδι", "ανηλικ", "ηλικιωμεν", "ανοια", "αναπηρ", "ευαλωτ", "οικονομικη δυσκολια", "χρεωμεν"],
                    ["exploit", "manipulat", "pressure", "addict", "push", "persuad", "convince", "upsell",
                     "dangerous", "εκμεταλλευ", "χειραγωγ", "εθισ", "πιεζ", "πεισ", "ωθ", "επικινδυν"]]],
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
        "combos": [[["face", "facial", "biometric", "voice", "photo", "picture", "προσωπ", "βιομετρικ", "φωνη",
                     "φωτογραφ"],
                    ["infer", "predict", "categoris", "categoriz", "classify", "guess", "determine", "tell whether",
                     "συμπερα", "κατηγοριοπ", "ταξινομ", "μαντε"],
                    ["sexual orientation", "gay", "lesbian", "homosexual", "religio", "political opinion",
                     "political views", "political belief", "race", "ethnic", "trade union",
                     "σεξουαλικο προσανατολ", "ομοφυλοφ", "θρησκ", "πολιτικες πεποιθησ", "πολιτικες αποψ",
                     "φυλη", "εθνοτ", "συνδικαλ"]]],
        "prototype": "Categorising people from their biometric data, such as face images, to infer their "
                     "race, political opinions, religion or sexual orientation.",
    },
    {
        "id": "face_scraping", "tier": "unacceptable", "ref": "Art. 5(1)(e)",
        "label": {"en": "Untargeted scraping of facial images", "el": "Μη στοχευμένη συλλογή εικόνων προσώπων"},
        "keywords": ["scrape face", "scraping face", "scrape facial", "scraping facial", "facial images from the internet",
                     "cctv footage to build", "facial recognition database",
                     "συλλογη εικονων προσωπ", "βαση δεδομενων αναγνωρισης προσωπ"],
        # Μαζική συλλογή εικόνων + πρόσωπα / αναγνώριση προσώπων
        "combos": [[["download", "scrape", "scraping", "collect", "harvest", "crawl", "gather", "millions of",
                     "κατεβα", "συλλεγ", "συλλογ", "εκατομμυρια"],
                    ["photo", "image", "picture", "cctv", "footage", "selfie", "φωτογραφ", "εικον", "βιντεο"],
                    ["face recognition", "facial recognition", "database of faces", "face database", "faces",
                     "face search",
                     "αναγνωριση προσωπ", "βαση δεδομενων προσωπ", "προσωπων"]]],
        "prototype": "Building or expanding a facial recognition database by untargeted scraping of facial "
                     "images from the internet or CCTV footage.",
    },
    {
        "id": "realtime_rbi", "tier": "unacceptable", "ref": "Art. 5(1)(h)",
        "label": {"en": "Real-time remote biometric identification in public (law enforcement)",
                  "el": "Βιομετρική ταυτοποίηση εξ αποστάσεως σε πραγματικό χρόνο σε δημόσιους χώρους (επιβολή του νόμου)"},
        "keywords": ["real-time facial recognition", "live facial recognition", "facial recognition in public",
                     "αναγνωριση προσωπου σε πραγματικο χρονο", "αναγνωριση προσωπων σε δημοσιους χωρους"],
        # Πρόσωπα + ζωντανά + δημόσιος χώρος + αστυνομία
        "combos": [[["face", "facial", "προσωπ"],
                    ["real time", "real-time", "live", "πραγματικο χρονο", "ζωντανα"],
                    ["public", "station", "street", "stadium", "crowd", "square", "metro", "everyone passing",
                     "δημοσι", "σταθμ", "γηπεδ", "πληθ", "δρομ", "πλατει", "μετρο"],
                    ["police", "law enforcement", "wanted", "suspect", "αστυνομ", "καταζητ", "υποπτ"]]],
        "prototype": "Real-time remote biometric identification such as live facial recognition of people in "
                     "publicly accessible spaces for law enforcement purposes.",
    },
    {
        "id": "predictive_policing", "tier": "unacceptable", "ref": "Art. 5(1)(d)",
        "label": {"en": "Predicting crime risk from profiling alone",
                  "el": "Πρόβλεψη εγκληματικότητας αποκλειστικά βάσει κατάρτισης προφίλ"},
        "keywords": ["predict who will commit", "predictive policing", "likely to commit a crime", "criminal risk profil",
                     "προβλεψη εγκληματ", "πιθανοτητα να διαπραξ"],
        "combos": [[["predict", "forecast", "risk score", "likelihood", "probabilit", "προβλεψ", "προβλεπ",
                     "πιθανοτητα"],
                    ["commit a crime", "commit crimes", "criminal offence", "reoffend", "criminal behaviour",
                     "εγκλημ", "παραβατικ", "υποτροπ"],
                    ["individual", "person", "profil", "personality", "ατομ", "προσωπο", "προφιλ", "προσωπικοτητ"]]],
        "prototype": "Predicting the risk that an individual will commit a crime based solely on profiling "
                     "or on their personality traits and characteristics.",
    },

    # ---------------- Υψηλού κινδύνου — Παράρτημα III ----------------
    {
        "id": "employment", "tier": "high", "ref": "Annex III, point 4",
        "label": {"en": "Employment & worker management", "el": "Απασχόληση και διαχείριση εργαζομένων"},
        "keywords": ["recruit", "hiring", "job applicant", "candidate", "screen cv", "screening cv", "cvs",
                     "resumes", "résumé", "shortlist", "job interview", "employee promotion", "promotion decision", "dismiss", "termination of employment", "fire employee", "firing",
                     "performance evaluation", "evaluate employee", "monitor employee", "task allocation",
                     "προσληψ", "υποψηφι", "βιογραφικ", "συνεντευξ", "προαγωγ", "απολυσ",
                     "αξιολογηση εργαζομεν", "αξιολογηση αποδοσης", "παρακολουθηση εργαζομεν", "επιλογη προσωπικου"],
        # Αιτήσεις/υποψήφιοι για θέση εργασίας, και απόφαση/αξιολόγηση +
        # εργαζόμενοι (και πλατφόρμες) + αντικείμενο εργασιακής απόφασης
        "combos": [[["promot", "προαγωγ", "προαγ"],
                    ["employee", "staff", "worker", "team leader", "manager", "performance",
                     "εργαζομεν", "υπαλληλ", "προσωπικ", "αποδοσ"]],
                   [["applic", "candidate", "cover letter", "αιτησ", "υποψηφι", "συνοδευτικ επιστολ"],
                    ["job", "position", "role", "vacanc", "hire", "hiring", "interview", "invite", "shortlist",
                     "recruit", "θεση εργασ", "θεσεις εργασ", "προσληψ", "συνεντευξ"]],
                   [["decide", "assign", "allocat", "rank", "evaluat", "assess", "deactivat", "suspend", "monitor",
                     "rate ", "rating", "score", "recommend", "αποφασ", "κατανεμ", "αξιολογ", "απενεργοπ",
                     "αναστελ", "παρακολουθ", "βαθμολογ"],
                    ["driver", "courier", "rider", "worker", "employee", "staff", "freelancer", "teacher",
                     "οδηγ", "διανομ", "εργαζομεν", "υπαλληλ", "προσωπικ", "εκπαιδευτικ", "καθηγητ"],
                    ["shift", "ride", "task", "contract", "performance", "productivity", "promot", "dismiss",
                     "terminat", "bonus", "salary", "βαρδι", "απολυσ", "αποδοσ", "παραγωγικ", "συμβασ",
                     "προαγωγ", "μπονους", "μισθ"]]],
        "prototype": "AI used to recruit or select job candidates, filter CVs, evaluate applicants, or make "
                     "decisions on promotion, termination, task allocation or monitoring of employees.",
        "sectors": ["Human Resources"],
    },
    {
        "id": "credit", "tier": "high", "ref": "Annex III, point 5(b)",
        "label": {"en": "Creditworthiness & credit scoring", "el": "Πιστοληπτική ικανότητα και πιστωτική βαθμολόγηση"},
        "keywords": ["credit scor", "creditworth", "loan approv", "loan application", "approve loans", "mortgage", "lending decision",
                     "personal loan", "interest rate",
                     "πιστοληπτικ", "δανει", "στεγαστικ", "πιστωτικη βαθμολογ"],
        "prototype": "AI that evaluates the creditworthiness of individuals or sets their credit score, "
                     "for example to approve or reject loan or mortgage applications.",
        "sectors": ["Finance"],
    },
    {
        "id": "insurance", "tier": "high", "ref": "Annex III, point 5(c)",
        "label": {"en": "Life & health insurance pricing", "el": "Τιμολόγηση ασφάλισης ζωής και υγείας"},
        "keywords": ["life insurance", "health insurance", "insurance premium", "insurance pricing", "underwriting",
                     "ασφαλιση ζωης", "ασφαλιση υγειας", "ασφαλιστρ"],
        # Τιμολόγηση/κινδυνος + ασφάλιση ζωής ή υγείας
        "combos": [[["price", "pricing", "premium", "risk", "decide", "set", "calculat", "τιμολογ", "τιμη", "κινδυν",
                     "υπολογ", "αποφασ"],
                    ["insur", "ασφαλ"],
                    ["life", "health", "medical", "ζωης", "υγει", "ιατρικ"]]],
        "prototype": "An insurance company using AI to assess the risk of individual customers and set the premiums of their life or health insurance policies.",
        "sectors": ["Finance", "Healthcare"],
    },
    {
        "id": "public_benefits", "tier": "high", "ref": "Annex III, point 5(a), (d)",
        "label": {"en": "Access to public benefits & emergency services",
                  "el": "Πρόσβαση σε δημόσιες παροχές και υπηρεσίες έκτακτης ανάγκης"},
        "keywords": ["welfare", "social benefit", "public assistance", "eligibility for benefits", "emergency call",
                     "triage", "dispatch ambulance",
                     "επιδομ", "κοινωνικες παροχ", "επιλεξιμοτητα", "κλησεις εκτακτης αναγκης", "διαλογη ασθενων",
                     "housing benefit", "disability benefit", "unemployment benefit", "child benefit", "social housing"],
        # Επιλεξιμότητα/χορήγηση + δημόσια παροχή
        "combos": [[["eligib", "qualif", "entitled", "who gets", "grant", "stop payment", "δικαιουχ", "δικαιουντ",
                     "επιλεξιμ", "χορηγ", "διακοπ"],
                    ["housing", "welfare", "social security", "pension", "disability", "unemployment",
                     "public assistance", "social benefit", "state benefit", "government benefit", "allowance",
                     "επιδομ", "στεγαστ", "συνταξ", "κοινωνικ παροχ", "προνοι", "αναπηρ", "ανεργ"]]],
        "prototype": "AI used by authorities to decide eligibility for public benefits and services, or to "
                     "classify emergency calls and prioritise emergency response or patient triage.",
        "sectors": ["Healthcare", "Public Sector"],
    },
    {
        "id": "education", "tier": "high", "ref": "Annex III, point 3",
        "label": {"en": "Education & vocational training", "el": "Εκπαίδευση και επαγγελματική κατάρτιση"},
        "keywords": ["admissions", "admission to", "grade student", "grading", "exams", "examination", "assess students", "evaluate students",
                     "proctor", "cheating", "learning outcome", "student placement",
                     "εισαγωγη φοιτητ", "εισαγωγη μαθητ", "βαθμολογηση μαθητ", "βαθμολογηση φοιτητ", "εξετασεις μαθητ", "εξετασεις φοιτητ", "πανελλαδικ", "γραπτα",
                     "αξιολογηση μαθητ", "αξιολογηση φοιτητ", "επιτηρηση εξετασ", "αντιγραφη στις εξετασ"],
        "combos": [[["admit", "admission", "accept", "select", "place", "εισαγωγ", "εισακτ", "επιλογ", "κατατασσ",
                     "τοποθετ"],
                    ["university", "school", "college", "programme", "program", "master", "course", "classes",
                     "pupil", "student", "πανεπιστημ", "σχολ", "μεταπτυχ", "προγραμμα σπουδ", "τμηματ", "μαθητ",
                     "φοιτητ"]],
                   [["grade", "grading", "mark", "score", "assess", "evaluat", "βαθμολογ", "αξιολογ"],
                    ["student", "pupil", "exam", "essay", "assignment", "coursework", "test",
                     "μαθητ", "φοιτητ", "εξετασ", "γραπτ", "εργασιες"]]],
        "prototype": "AI that decides admission to schools or universities, grades exams, evaluates "
                     "learning outcomes or monitors students for cheating during tests.",
        "sectors": ["Education"],
    },
    {
        "id": "biometric_id", "tier": "high", "ref": "Annex III, point 1",
        "label": {"en": "Biometric identification & emotion recognition",
                  "el": "Βιομετρική ταυτοποίηση και αναγνώριση συναισθημάτων"},
        "keywords": ["facial recognition", "face recognition", "biometric", "fingerprint", "voice print",
                     "emotion detect", "emotion recogn",
                     "αναγνωριση προσωπ", "βιομετρικ", "δακτυλικ αποτυπ", "αναγνωριση συναισθ"],
        # Αναγνώριση συναισθημάτων σε οποιοδήποτε άλλο πλαίσιο (Παράρτημα III, 1(c))
        "combos": [[_EMOTION_TERMS,
                    ["analys", "analyz", "detect", "recogni", "infer", "monitor", "track",
                     "αναλυ", "ανιχν", "αναγνωρ", "συμπερα", "παρακολουθ"]],
                   # Ταυτοποίηση ατόμων από το πρόσωπό τους
                   [["face", "faces", "facial", "προσωπ"],
                    ["recognis", "recogniz", "identif", "match", "αναγνωρ", "ταυτοπ"]]],
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
        "label": {"en": "Migration, asylum & border control", "el": "Μετανάστευση, άσυλο και έλεγχος συνόρων"},
        "keywords": ["asylum", "visa application", "border control", "migrant", "residence permit",
                     "border guard", "traveller", "traveler", "travel document", "passport control",
                     "ασυλο", "βιζα", "ελεγχος συνορ", "μεταναστ", "αδεια διαμονης"],
        "prototype": "AI used to examine asylum, visa or residence permit applications, assess migration "
                     "risks, or detect and identify people at border control.",
        "sectors": ["Public Sector"]
    },
    {
        "id": "justice_democracy", "tier": "high", "ref": "Annex III, point 8",
        "label": {"en": "Justice & democratic processes", "el": "Δικαιοσύνη και δημοκρατικές διαδικασίες"},
        "keywords": ["judges", "assist judge", "court decision", "judicial", "sentencing", "dispute resolution",
                     "judgment", "judgement", "for courts", "small claims", "rulings",
                     "influence voters", "election campaign", "voting behaviour", "voter", "how they vote",
                     "referendum",
                     "δικαστ", "δικαστικ αποφασ", "επιλυση διαφορων", "ψηφοφορ", "εκλογ"],
        "prototype": "AI that assists judges in researching and applying the law to facts, or that is "
                     "intended to influence the outcome of elections or the voting behaviour of citizens.",
        "sectors": ["Legal Services", "Public Sector"]
    },
    # ---------------- Υψηλού κινδύνου — Παράρτημα I (ασφάλεια προϊόντων) ----------------
    {
        "id": "safety_component", "tier": "high", "ref": "Art. 6(1), Annex I",
        "label": {"en": "Safety component of a regulated product (e.g. medical device, vehicle, machinery)",
                  "el": "Στοιχείο ασφάλειας ρυθμιζόμενου προϊόντος (π.χ. ιατροτεχνολογικό, όχημα, μηχάνημα)"},
        "keywords": ["medical device", "diagnos", "radiolog", "autonomous vehicle", "self-driving",
                     "machinery", "industrial robot", "toys", "aviation",
                     "ιατροτεχνολογικ", "διαγνωσ", "ακτινολογ", "αυτονομο οχημα", "αυτοοδηγουμεν",
                     "μηχανημα", "βιομηχανικο ρομποτ", "x-ray", "mri", "ct scan", "electrocardiogram", "ecg",
                     "ακτινογραφ", "ηλεκτροκαρδιογραφ", "αξονικ τομογραφ", "μαγνητικ τομογραφ", "διαγιγνωσκ"],
        # Ιατρική εξέταση/εικόνα + ανίχνευση + νόσος
        "combos": [[["scan", "image", "photo", "x-ray", "test result", "blood", "mole", "εξετασ", "εικον", "φωτογραφ",
                     "ακτινογραφ"],
                    ["detect", "flag", "diagnos", "identif", "predict", "spot", "ανιχν", "διαγνω", "διαγιγνωσκ",
                     "εντοπ", "προβλεπ"],
                    ["disease", "cancer", "tumour", "tumor", "tuberculosis", "illness", "patholog", "heart condition",
                     "ασθενει", "νοσ", "καρκιν", "ογκ", "παθησ", "καρδιακ"]]],
        "prototype": "Medical AI software that analyses scans, X-rays or patient data to diagnose disease, or AI "
                     "that is a safety component of a regulated product such as a vehicle, machinery or a toy.",
        "sectors": ["Healthcare"],
    },

    # ---------------- Περιορισμένου κινδύνου — Άρθρο 50 ----------------
    {
        "id": "chatbot", "tier": "limited", "ref": "Art. 50(1)",
        "label": {"en": "AI interacting directly with people (e.g. chatbots)",
                  "el": "ΤΝ που συνομιλεί απευθείας με ανθρώπους (π.χ. chatbots)"},
        "keywords": ["chatbot", "chat bot", "virtual assistant", "conversational", "voice assistant",
                     "customer service bot", "talk to customers", "ai assistant", "virtual agent", "voice agent",
                     "conversational agent", "virtual concierge", "answers the phone", "answers calls",
                     "εικονικος βοηθ", "ψηφιακος βοηθ", "εικονικο βοηθ", "ψηφιακο βοηθ", "ψηφιακου βοηθ",
                     "συνομιλι", "φωνητικος βοηθ", "φωνητικο βοηθ", "απαντα στο τηλεφωνο"],
        "prototype": "A chatbot or virtual assistant that talks directly with customers or users, who may "
                     "not realise they are interacting with an AI system.",
        "sectors": ["Customer Support", "Marketing & Sales", "Hospitality & Food Service"],
    },
    {
        "id": "synthetic_content", "tier": "limited", "ref": "Art. 50(2), 50(4)",
        "label": {"en": "AI-generated or manipulated content (incl. deepfakes)",
                  "el": "Περιεχόμενο που δημιουργείται ή αλλοιώνεται με ΤΝ (και deepfakes)"},
        "keywords": ["deepfake", "deep fake", "synthetic image", "synthetic video", "synthetic voice",
                     "ai-generated", "ai generated", "generate images", "generate articles", "generate text",
                     "voice clon", "generative ai", "write articles",
                     "συνθετικ", "παραγεται απο ΑΙ", "παραγωγη εικονων", "παραγωγη κειμεν", "κλωνοποιηση φωνης",
                     "γενετικη ΑΙ", "παραγωγικη ΑΙ",
                     # Γνωστά εργαλεία παραγωγής μουσικής, φωνής και εικόνας
                     "suno", "udio", "elevenlabs", "midjourney", "stable diffusion", "dall-e", "dall·e",
                     "ai music", "ai-generated music", "ai song", "ai art", "image generator", "music generator",
                     "γεννητρια εικονων", "γεννητρια μουσικης", "written by ai", "written entirely by ai",
                     "ai-written", "ai written", "γραμμενα απο τν", "γραμμενα απο ai", "κλωνοποι", "voice clone",
                     "cloned voice", "clone the voice", "synthetic voices", "ai avatar"],
        # Μουσική, ήχος, εικόνα ή βίντεο που δημιουργείται: λέξη μέσου + λέξη
        # δημιουργίας οπουδήποτε στο κείμενο (π.χ. «creating some music tracks»)
        "combos": [[["music", "song", "track", "audio", "vocals", "voice", "melody", "soundtrack", "podcast",
                     "artwork", "illustration", "image", "video", "photo",
                     "τραγουδ", "μουσικ", "κομματ", "ηχογραφ", "φωνη", "μελωδ", "εικονογραφ", "εικον", "βιντεο",
                     "φωτογραφ"],
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
# Ηθικές διαστάσεις που θέτει κάθε περίπτωση χρήσης. Όταν η βαθμίδα
# κινδύνου εντοπίσει μια περίπτωση, οι διαστάσεις της προστίθενται στην
# ανάκτηση της βάσης γνώσης, ακόμη κι αν ο ταξινομητής τις έχασε (π.χ. η
# δικαιοσύνη στην επιλογή βιογραφικών). Οι βαθμολογίες του ταξινομητή δεν
# αλλάζουν· η διεπαφή σημειώνει ποιες διαστάσεις προστέθηκαν έτσι.
# -----------------------------------------------------------------------
CASE_DIMENSIONS = {
    "social_scoring": ["fairness", "non_maleficence"],
    "manipulation": ["non_maleficence", "transparency"],
    "exploit_vulnerable": ["non_maleficence", "fairness"],
    "emotion_work_edu": ["privacy", "non_maleficence"],
    "biometric_categorisation": ["privacy", "fairness"],
    "face_scraping": ["privacy"],
    "realtime_rbi": ["privacy", "accountability"],
    "predictive_policing": ["fairness", "accountability"],
    "employment": ["fairness", "transparency"],
    "credit": ["fairness", "transparency"],
    "insurance": ["fairness", "privacy"],
    "public_benefits": ["fairness", "accountability"],
    "education": ["fairness", "transparency"],
    "biometric_id": ["privacy"],
    "critical_infra": ["non_maleficence", "accountability"],
    "law_enforcement": ["fairness", "accountability"],
    "migration": ["fairness", "accountability"],
    "justice_democracy": ["accountability", "transparency"],
    "safety_component": ["non_maleficence", "accountability"],
    "chatbot": ["transparency"],
    "synthetic_content": ["transparency"],
    "emotion_disclosure": ["privacy", "transparency"],
}

# -----------------------------------------------------------------------
# Τυπικές «καθημερινές» χρήσεις ΤΝ χωρίς ειδικές υποχρεώσεις. Χρησιμεύουν
# ως αντίπαλος στη σημασιολογική σύγκριση: μια ένδειξη υψηλού ή
# περιορισμένου κινδύνου χωρίς ρητή λέξη-κλειδί δίνεται μόνο αν το δίλημμα
# μοιάζει με την περίπτωση του AI Act σαφώς περισσότερο από ό,τι με
# οποιαδήποτε από αυτές (π.χ. οι «διαδρομές φορτηγών» μοιάζουν με
# «υποδομές ζωτικής σημασίας», αλλά ακόμη περισσότερο με την εφοδιαστική).
# -----------------------------------------------------------------------
EVERYDAY_PROTOTYPES = [
    "Optimising delivery routes, logistics, vehicle scheduling or supply-chain planning.",
    "Forecasting sales, demand or stock levels from historical business data.",
    "Predictive maintenance of factory machines, or detecting defective products in quality control.",
    "Writing, translating, summarising or grammar-checking internal documents, emails and meeting notes.",
    "AI coding assistants that help software developers write, review or test code.",
    "Recommending similar products or content to shoppers based on what they browsed or bought.",
    "Filtering spam and organising the company's internal email inbox.",
    "Optimising energy use, heating and lighting of an office building.",
    "Analysing sales data to find which products sell best.",
]
# Πόσο πρέπει η ομοιότητα με την περίπτωση να ξεπερνά την ισχυρότερη
# ομοιότητα με καθημερινή χρήση, ώστε να αρκεί χωρίς λέξη-κλειδί
EVERYDAY_GAP = 0.15

# -----------------------------------------------------------------------
# Πλησιέστεροι γείτονες: σύγκριση με επισημασμένα σενάρια (model/
# risk_tier_cases.py) αντί για μία μόνο περιγραφή ανά περίπτωση. Οι k
# πλησιέστεροι ψηφίζουν, σταθμισμένοι με την ομοιότητά τους· η ψήφος
# λαμβάνεται υπόψη μόνο όταν ο πλησιέστερος είναι αρκετά κοντά και η
# πλειοψηφία σαφής. Οι παράμετροι επιλέχθηκαν με leave-one-out.
# -----------------------------------------------------------------------
# Μετά την τελική μέτρηση στο TEST2_CASES, η μνήμη περιλαμβάνει και αυτό
KNN_SPLITS = ("DEV_CASES", "TEST_CASES", "TEST2_CASES")
KNN_K = 5
KNN_MIN_SIM = 0.45
KNN_MIN_SHARE = 0.6
# Ενεργό μόνο στην αξιολόγηση: αγνοεί το ίδιο το σενάριο (leave-one-out)
KNN_EXCLUDE_IDENTICAL = False
# Παραλλαγές συνδυασμού (επιλογή με leave-one-out, βλ. evaluate_risk_tier)
KNN_OVERRIDES_EXPLICIT = True
KNN_CAN_LOWER = True

# -----------------------------------------------------------------------
# Περιγραφή κάθε βαθμίδας και βασικές υποχρεώσεις, στις δύο γλώσσες
# -----------------------------------------------------------------------
TIER_INFO = {
    "unacceptable": {
        "verdict": {
            "en": "Possibly a PROHIBITED practice under the EU AI Act (unacceptable risk).",
            "el": "Πιθανότατα πρόκειται για ΑΠΑΓΟΡΕΥΜΕΝΗ πρακτική σύμφωνα με τον Κανονισμό της ΕΕ για την ΤΝ (μη αποδεκτός κίνδυνος).",
        },
        "obligations": {
            "en": [
                "Prohibited practices may not be placed on the EU market, put into service or used (Art. 5), in force since 2 February 2025.",
                "Fines can reach €35 million or 7% of worldwide annual turnover (Art. 99(3)).",
                "Check whether a narrow exception in Art. 5 applies (e.g. medical or safety reasons for emotion recognition).",
            ],
            "el": [
                "Οι απαγορευμένες πρακτικές δεν μπορούν να διατίθενται, να τίθενται σε λειτουργία ή να χρησιμοποιούνται στην ΕΕ (Άρθρο 5). Η απαγόρευση ισχύει από τις 2 Φεβρουαρίου 2025.",
                "Τα πρόστιμα φτάνουν τα 35 εκατ. € ή το 7% του παγκόσμιου ετήσιου κύκλου εργασιών (Άρθρο 99(3)).",
                "Ελέγξτε αν ισχύει κάποια από τις στενές εξαιρέσεις του Άρθρου 5 (π.χ. ιατρικοί λόγοι ή λόγοι ασφάλειας για την αναγνώριση συναισθημάτων).",
            ],
        },
    },
    "high": {
        "verdict": {
            "en": "Likely a HIGH-RISK AI system under the EU AI Act.",
            "el": "Πιθανότατα πρόκειται για σύστημα ΥΨΗΛΟΥ ΚΙΝΔΥΝΟΥ σύμφωνα με τον Κανονισμό της ΕΕ για την ΤΝ.",
        },
        "obligations": {
            "en": [
                "Providers: risk management system, data governance, technical documentation, logging, accuracy & robustness, conformity assessment, CE marking and EU database registration (Arts. 9–17, 43, 49).",
                "Deployers: use per instructions, human oversight by competent staff, monitor operation, keep logs, inform affected people and workers (Art. 26).",
                "Public bodies and some private deployers (e.g. credit, insurance) must run a fundamental-rights impact assessment (Art. 27).",
                "Affected people have a right to an explanation of decisions taken with the system's help (Art. 86).",
            ],
            "el": [
                "Πάροχοι: σύστημα διαχείρισης κινδύνων, διακυβέρνηση δεδομένων, τεχνικός φάκελος, αρχεία καταγραφής, ακρίβεια και στιβαρότητα, αξιολόγηση συμμόρφωσης, σήμανση CE και καταχώριση στη βάση δεδομένων της ΕΕ (Άρθρα 9–17, 43, 49).",
                "Φορείς εφαρμογής (όσοι χρησιμοποιούν το σύστημα): χρήση σύμφωνα με τις οδηγίες, ανθρώπινη εποπτεία από κατάλληλα καταρτισμένο προσωπικό, παρακολούθηση της λειτουργίας, τήρηση αρχείων, ενημέρωση των εργαζομένων και όσων επηρεάζονται (Άρθρο 26).",
                "Οι δημόσιοι φορείς και ορισμένοι ιδιώτες (π.χ. για πιστοληπτική αξιολόγηση ή ασφάλιση) πρέπει να κάνουν εκτίμηση επιπτώσεων στα θεμελιώδη δικαιώματα (Άρθρο 27).",
                "Όσοι επηρεάζονται από απόφαση που λήφθηκε με τη βοήθεια του συστήματος έχουν δικαίωμα να ζητήσουν εξήγηση (Άρθρο 86).",
            ],
        },
    },
    "limited": {
        "verdict": {
            "en": "Likely LIMITED risk: transparency obligations apply under the EU AI Act.",
            "el": "Πιθανότατα ΠΕΡΙΟΡΙΣΜΕΝΟΣ κίνδυνος: ισχύουν υποχρεώσεις ενημέρωσης σύμφωνα με τον Κανονισμό της ΕΕ για την ΤΝ.",
        },
        "obligations": {
            "en": [
                "Tell people they are interacting with an AI system, unless it is obvious (Art. 50(1)).",
                "Mark AI-generated audio, image, video or text in a machine-readable way (Art. 50(2)).",
                "Disclose deepfakes and AI-generated text published to inform the public (Art. 50(4)).",
            ],
            "el": [
                "Ενημερώστε τους χρήστες ότι μιλούν με σύστημα ΤΝ, εκτός αν αυτό είναι προφανές (Άρθρο 50(1)).",
                "Σημάνετε, σε μορφή αναγνώσιμη από μηχανή, τον ήχο, την εικόνα, το βίντεο ή το κείμενο που έχει δημιουργηθεί με ΤΝ (Άρθρο 50(2)).",
                "Δηλώστε ότι πρόκειται για deepfake ή για κείμενο από ΤΝ, όταν δημοσιεύεται για την ενημέρωση του κοινού (Άρθρο 50(4)).",
            ],
        },
    },
    "minimal": {
        "verdict": {
            "en": "Likely MINIMAL risk: no specific AI Act obligations beyond AI literacy.",
            "el": "Πιθανότατα ΕΛΑΧΙΣΤΟΣ κίνδυνος: ο Κανονισμός για την ΤΝ δεν επιβάλλει ειδικές υποχρεώσεις, πέρα από τον γραμματισμό στην ΤΝ.",
        },
        "obligations": {
            "en": [
                "Ensure sufficient AI literacy of staff who operate or use the system (Art. 4).",
                "Voluntary codes of conduct are encouraged (Art. 95); other laws (e.g. GDPR, consumer, anti-discrimination law) still apply.",
            ],
            "el": [
                "Φροντίστε ώστε όσοι χειρίζονται ή χρησιμοποιούν το σύστημα να έχουν επαρκή γνώση της ΤΝ (Άρθρο 4).",
                "Ενθαρρύνονται εθελοντικοί κώδικες δεοντολογίας (Άρθρο 95). Οι υπόλοιποι νόμοι (π.χ. ΓΚΠΔ, προστασία καταναλωτή, απαγόρευση διακρίσεων) εξακολουθούν να ισχύουν.",
            ],
        },
    },
}

TIER_NAMES = {
    "unacceptable": {"en": "unacceptable-risk (prohibited)", "el": "των απαγορευμένων πρακτικών"},
    "high": {"en": "high-risk", "el": "του υψηλού κινδύνου"},
    "limited": {"en": "limited-risk", "el": "του περιορισμένου κινδύνου"},
}

NOTES = {
    "also_high": {
        "en": "Even if an Art. 5 exception applied, the system would still be high-risk: {label} ({ref}).",
        "el": "Ακόμη κι αν ίσχυε κάποια εξαίρεση του Άρθρου 5, το σύστημα θα ήταν υψηλού κινδύνου: {label} ({ref}).",
    },
    "near_prohibited": {
        "en": "Your description closely resembles prohibited practices (Art. 5). Check carefully whether the system falls under a prohibition.",
        "el": "Η περιγραφή σας μοιάζει πολύ με απαγορευμένες πρακτικές (Άρθρο 5). Ελέγξτε προσεκτικά μήπως το σύστημα εμπίπτει σε απαγόρευση.",
    },
    "by_similarity": {
        "en": "This tier is based on similarity to labelled example scenarios rather than on specific terms in your description.",
        "el": "Η βαθμίδα προκύπτει από την ομοιότητα με παραδείγματα που έχουν ήδη αξιολογηθεί, όχι από συγκεκριμένους όρους της περιγραφής σας.",
    },
    "borderline": {
        "en": "Your description is close to the {tier} tier — check carefully whether that tier applies.",
        "el": "Η περιγραφή σας είναι κοντά και στη βαθμίδα {tier}· ελέγξτε προσεκτικά μήπως ισχύει εκείνη.",
    },
    "high_exception": {
        "en": "An Annex III system is not high-risk if it only performs a narrow procedural or preparatory task and does not materially influence decisions (Art. 6(3)) — but it is always high-risk if it profiles natural persons.",
        "el": "Ένα σύστημα του Παραρτήματος III δεν θεωρείται υψηλού κινδύνου όταν κάνει μόνο μια στενή, διαδικαστική ή προπαρασκευαστική εργασία και δεν επηρεάζει ουσιαστικά την απόφαση (Άρθρο 6(3)). Αν όμως δημιουργεί προφίλ φυσικών προσώπων, είναι πάντα υψηλού κινδύνου.",
    },
    "high_dates": {
        "en": "Under Art. 113, high-risk obligations apply from 2 August 2026 (Annex III) and 2 August 2027 (Annex I products); check whether later amendments have shifted these dates.",
        "el": "Σύμφωνα με το Άρθρο 113, οι υποχρεώσεις για τα συστήματα υψηλού κινδύνου ισχύουν από τις 2 Αυγούστου 2026 (Παράρτημα III) και τις 2 Αυγούστου 2027 (προϊόντα του Παραρτήματος I). Ελέγξτε αν μεταγενέστερες τροποποιήσεις έχουν αλλάξει αυτές τις ημερομηνίες.",
    },
    "non_eu": {
        "en": "You selected a jurisdiction outside the EU. The AI Act still applies if the system is placed on the EU market or its output is used in the EU (Art. 2).",
        "el": "Επιλέξατε χώρα εκτός ΕΕ. Ο Κανονισμός για την ΤΝ ισχύει παρ' όλα αυτά αν το σύστημα διατίθεται στην αγορά της ΕΕ ή αν τα αποτελέσματά του χρησιμοποιούνται στην ΕΕ (Άρθρο 2).",
    },
    "indicative": {
        "en": "This is an automated indication based on your description, not a legal classification.",
        "el": "Πρόκειται για αυτόματη εκτίμηση με βάση την περιγραφή σας, όχι για νομική κατάταξη.",
    },
}

# Καθυστερημένη φόρτωση των ενσωματώσεων των περιγραφών (μία φορά ανά διεργασία)
_prototype_embeddings = None
_everyday_embeddings = None
_knn_memory = None


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
COMBO_MIN_SEM = 0.15


def _combo_matches(case: dict, norm_text: str) -> int:
    if any(p.search(norm_text) for p in case["_combo_exclude"]):
        return 0
    return sum(
        1 for combo in case["_combos"]
        if all(any(p.search(norm_text) for p in group) for group in combo)
    )


# -----------------------------------------------------------------------
# Σημασιολογικές ομοιότητες με κάθε περιγραφή περίπτωσης χρήσης. Επιστρέφει
# (περιθώρια, διαφορές από την πλησιέστερη καθημερινή χρήση), ή (None, None)
# αν το μοντέλο ενσωμάτωσης δεν είναι διαθέσιμο (μόνο λέξεις-κλειδιά)
# -----------------------------------------------------------------------
def _unit(matrix):
    return matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-8)


def _semantic_similarities(dilemma: str):
    global _prototype_embeddings, _everyday_embeddings
    try:
        if _prototype_embeddings is None:
            _prototype_embeddings = _unit(embed_texts([c["prototype"] for c in USE_CASES]))
            _everyday_embeddings = _unit(embed_texts(EVERYDAY_PROTOTYPES))
        query = _unit(embed_texts([dilemma])[0])
        sims = _prototype_embeddings @ query
        everyday = float(np.max(_everyday_embeddings @ query))
        return sims - sims.mean(), sims - everyday, query
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[risk_tier] semantic matching unavailable, using keywords only: {exc}")
        return None, None, None


# -----------------------------------------------------------------------
# Ψήφος των k πλησιέστερων επισημασμένων σεναρίων: (βαθμίδα, μερίδιο
# ψήφων, ομοιότητα του πλησιέστερου), ή None αν δεν υπάρχει σαφής ένδειξη
# -----------------------------------------------------------------------
def _knn_vote(query):
    global _knn_memory
    if query is None:
        return None
    try:
        if _knn_memory is None:
            import model.risk_tier_cases as cases
            rows = [row for name in KNN_SPLITS for row in getattr(cases, name)]
            _knn_memory = (_unit(embed_texts([r[0] for r in rows])), [r[2] for r in rows])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[risk_tier] labelled scenarios unavailable: {exc}")
        return None
    matrix, tiers = _knn_memory
    sims = matrix @ query
    if KNN_EXCLUDE_IDENTICAL:
        sims = np.where(sims > 0.9999, -1.0, sims)
    top = np.argsort(-sims)[:KNN_K]
    votes = {}
    for i in top:
        votes[tiers[i]] = votes.get(tiers[i], 0.0) + max(float(sims[i]), 0.0)
    total = sum(votes.values()) or 1.0
    winner = max(votes, key=votes.get)
    share, nearest = votes[winner] / total, float(sims[top[0]])
    if nearest < KNN_MIN_SIM or share < KNN_MIN_SHARE:
        return None
    return winner, share, nearest


# -----------------------------------------------------------------------
# Βαθμολογία μίας περίπτωσης χρήσης από τα δύο σήματα
# -----------------------------------------------------------------------
def _score_case(case: dict, norm_text: str, margin, sector: str, everyday_gap=None) -> tuple:
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
    # Χωρίς ρητή λέξη-κλειδί ή συνδυασμό, η σημασιολογική ομοιότητα αρκεί
    # μόνο για υψηλό ή περιορισμένο κίνδυνο, και μόνο αν το δίλημμα μοιάζει
    # με την περίπτωση σαφώς περισσότερο από ό,τι με μια καθημερινή χρήση ΤΝ.
    # Αλλιώς η βαθμολογία μένει κάτω από το κατώφλι και αξιοποιείται μόνο
    # για τις οριακές σημειώσεις.
    semantic_ok = (case["tier"] in ("high", "limited") and not case.get("requires_keyword")
                   and everyday_gap is not None and everyday_gap >= EVERYDAY_GAP)
    if hits == 0 and not semantic_ok:
        score = min(score, TIER_THRESHOLDS[case["tier"]] - 0.05)
    # Δεύτερη τιμή: αν η ένδειξη στηρίζεται σε ρητή λέξη-κλειδί ή συνδυασμό
    return min(score, 1.0), hits > 0


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
    sims, gaps, query = _semantic_similarities(dilemma) if dilemma and dilemma.strip() else (None, None, None)

    scored = []
    explicit_tiers = set()
    for i, case in enumerate(USE_CASES):
        sim = None if sims is None else float(sims[i])
        gap = None if gaps is None else float(gaps[i])
        score, explicit = _score_case(case, norm_text, sim, sector, gap)
        scored.append((case, score))
        if explicit and score >= TIER_THRESHOLDS[case["tier"]]:
            explicit_tiers.add(case["tier"])

    # Η υψηλότερη βαθμίδα με αντιστοίχιση πάνω από το κατώφλι της κερδίζει.
    # Μια καθαρά σημασιολογική ένδειξη δεν υπερισχύει όμως ρητής ένδειξης
    # χαμηλότερης βαθμίδας (π.χ. «άρθρα γραμμένα από ΤΝ για την τοπική
    # πολιτική» είναι περιορισμένος κίνδυνος, όχι επηρεασμός εκλογών)· τότε
    # η υψηλότερη βαθμίδα εμφανίζεται μόνο ως οριακή σημείωση.
    tier, confidence = "minimal", 0.0
    for candidate in TIER_ORDER[:-1]:
        best = max((s for c, s in scored if c["tier"] == candidate), default=0.0)
        lower_explicit = any(TIER_ORDER.index(t) > TIER_ORDER.index(candidate) for t in explicit_tiers)
        if best >= TIER_THRESHOLDS[candidate] and (candidate in explicit_tiers or not lower_explicit):
            tier, confidence = candidate, best
            break

    # Ψήφος των πλησιέστερων επισημασμένων σεναρίων. Μπορεί να ανεβάσει τη
    # βαθμίδα (ή να ακυρώσει μια καθαρά σημασιολογική ένδειξη), αλλά δεν
    # δίνει ποτέ μόνη της «απαγορευμένη πρακτική»: τότε δίνει υψηλό κίνδυνο
    # με σημείωση ότι η περιγραφή μοιάζει με απαγορευμένη πρακτική.
    knn = _knn_vote(query)
    knn_used, near_prohibited = False, False
    if knn:
        knn_tier = knn[0]
        if knn_tier == "unacceptable" and tier != "unacceptable":
            knn_tier, near_prohibited = "high", True
        rule_explicit = tier in explicit_tiers
        if TIER_ORDER.index(knn_tier) < TIER_ORDER.index(tier) and (KNN_OVERRIDES_EXPLICIT or not rule_explicit):
            tier, confidence, knn_used = knn_tier, round(knn[1] * knn[2], 3), True
        elif (KNN_CAN_LOWER and knn_tier == "minimal" and tier in ("high", "limited") and not rule_explicit):
            tier, knn_used = "minimal", True

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
        if c["tier"] == tier and (s >= floor or knn_used)
    ][:1 if knn_used else 3]

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
        position = min(best_above / TIER_THRESHOLDS[above], 0.9)
        if best_above >= TIER_THRESHOLDS[above] - 0.15:
            borderline_with = above
    position = round(min(max(position, 0.1), 0.9), 3)

    notes = []
    if near_prohibited:
        notes.append(NOTES["near_prohibited"][lang])
    if knn_used:
        notes.append(NOTES["by_similarity"][lang])
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
        # Διαστάσεις των περιπτώσεων που εντοπίστηκαν (κενό για «ελάχιστο»)
        "dimensions": [] if tier == "minimal" else list(dict.fromkeys(
            d for m in matches for d in CASE_DIMENSIONS.get(m["id"], []))),
    }
