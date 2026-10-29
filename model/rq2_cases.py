"""
Gold set for evaluating retrieval (research question 2).

Each scenario names the provisions a reviewer would expect a guidance report
to point to. The expectations were written from the legal texts, not from the
knowledge base, so a provision the knowledge base lacks counts as a miss.
They were drafted by the author and still need review by a legal expert.

Each expected provision: (label, instrument pattern, provision pattern).
A retrieved source matches when both patterns occur in the same reference.
"""

# Μοτίβα νομοθετημάτων
AIA = r"2024/1689|(?i:AI[\s‑-]Act|Artificial Intelligence Act)"
GDPR = r"2016/679|GDPR|RGPD|DSGVO|ΓΚΠΔ"
CCD = r"2023/2225"
MDR = r"2017/745"
UCPD = r"2005/29"
CRD = r"2011/83|2019/2161"
DSM = r"2019/790"
EQ = r"2000/78"
GR_AI = r"5321/2026"
CCPA = r"CCPA|CPRA"
TITLE7 = r"Title VII"

# Μοτίβα διατάξεων
# Χωρίς διάκριση πεζών-κεφαλαίων («art. 22») και με κάθε είδος παύλας
ART = r"(?i:Arts?\.?|Articles?|Άρθρ\w*)\s*(?:[\d()a-z\s,&–‑-]*?\b)?"
ANY = r""


def art(n):
    """Article n (as a whole number, e.g. 22 but not 220 or 2.2)."""
    return ART + rf"{n}(?![\d])"


def annex3(point):
    return rf"(?i:Annex|Παράρτημα|Παρ\.)\s*III[^;|]*?\b{point}"


CASES = [
    {"id": "S01", "country": "Germany", "sector": "Human Resources", "lang": "en",
     "text": "We plan to use an AI system that scores applicants' CVs and automatically rejects the lowest-ranked "
             "candidates before any recruiter looks at them.",
     "expected": [("AI Act Annex III(4)", AIA, annex3(4)), ("GDPR Art. 22", GDPR, art(22)),
                  ("Directive 2000/78/EC", EQ, ANY)]},
    {"id": "S02", "country": "Greece", "sector": "Finance", "lang": "el",
     "text": "Η τράπεζά μας θέλει να χρησιμοποιήσει μοντέλο ΤΝ που αποφασίζει μόνο του για την έγκριση "
             "καταναλωτικών δανείων, χωρίς ανθρώπινο έλεγχο.",
     "expected": [("AI Act Annex III(5)(b)", AIA, annex3(r"5\(b\)")), ("GDPR Art. 22", GDPR, art(22)),
                  ("Directive 2023/2225 Art. 18", CCD, art(18))]},
    {"id": "S03", "country": "France", "sector": "Marketing & Sales", "lang": "en",
     "text": "Our agency wants to publish AI-generated videos showing a famous actor endorsing our product, "
             "without saying they are synthetic.",
     "expected": [("AI Act Art. 50(4)", AIA, ART + r"50\(4\)|50\(2\)[^;]*50\(4\)"), ("Directive 2005/29/EC", UCPD, ANY)]},
    {"id": "S04", "country": "Spain", "sector": "Customer Support", "lang": "en",
     "text": "We are replacing our phone support with an AI voice agent and would rather customers did not "
             "realise they are talking to a machine.",
     "expected": [("AI Act Art. 50(1)", AIA, ART + r"50\(1\)")]},
    {"id": "S05", "country": "Italy", "sector": "Human Resources", "lang": "en",
     "text": "We want webcams to detect our remote employees' emotions during meetings to measure engagement.",
     "expected": [("AI Act Art. 5(1)(f)", AIA, ART + r"5\(1\)\(f\)"), ("GDPR Art. 9", GDPR, art(9)),
                  ("GDPR Art. 88", GDPR, art(88))]},
    {"id": "S06", "country": "Greece", "sector": "Healthcare", "lang": "en",
     "text": "A start-up wants to sell an app that reads photos of skin moles and tells users whether they are "
             "likely to be cancerous.",
     "expected": [("MDR 2017/745", MDR, ANY), ("AI Act Art. 6(1) / Annex I", AIA, ART + r"6\(1\)|Annex I\b"),
                  ("GDPR Art. 9", GDPR, art(9))]},
    {"id": "S07", "country": "Netherlands", "sector": "Education", "lang": "en",
     "text": "The university wants an AI system to rank applicants for admission to its master's programmes.",
     "expected": [("AI Act Annex III(3)", AIA, annex3(3)), ("GDPR Art. 22", GDPR, art(22))]},
    {"id": "S08", "country": "Ireland", "sector": "Software Development", "lang": "en",
     "text": "Our developers paste customer support tickets with names and email addresses into a public "
             "chatbot to draft replies.",
     "expected": [("GDPR Art. 5", GDPR, art(5)), ("GDPR Art. 28", GDPR, art(28))]},
    {"id": "S09", "country": "Greece", "sector": "Public Sector", "lang": "el",
     "text": "Ο δήμος θέλει να χρησιμοποιήσει αλγόριθμο που αποφασίζει ποιοι πολίτες δικαιούνται κοινωνικό επίδομα.",
     "expected": [("AI Act Annex III(5)(a)", AIA, annex3(r"5\(a\)")), ("AI Act Art. 27", AIA, art(27)),
                  ("GDPR Art. 22", GDPR, art(22))]},
    {"id": "S10", "country": "Greece", "sector": "Legal Services", "lang": "en",
     "text": "Our law firm wants to upload confidential client contracts to a cloud AI tool for review.",
     "expected": [("GDPR Art. 28", GDPR, art(28)), ("GDPR Art. 32", GDPR, art(32))]},
    {"id": "S11", "country": "Germany", "sector": "Arts & Creative Industries", "lang": "en",
     "text": "We want to train an image generator on illustrations scraped from artists' portfolio websites.",
     "expected": [("Directive 2019/790 Art. 4(3)", DSM, ART + r"4\(3\)"), ("AI Act Art. 53", AIA, art(53))]},
    {"id": "S12", "country": "California", "sector": "Human Resources", "lang": "en",
     "text": "We plan to use an AI system that scores applicants' CVs and automatically rejects the lowest-ranked "
             "candidates before any recruiter looks at them.",
     "expected": [("Title VII", TITLE7, ANY), ("CCPA/CPRA", CCPA, ANY)]},
    {"id": "S13", "country": "Germany", "sector": "General / Other", "lang": "en",
     "text": "We want to rate customers' trustworthiness from their social media activity and refuse service "
             "to those with low scores.",
     "expected": [("AI Act Art. 5(1)(c)", AIA, ART + r"5\(1\)\(c\)"), ("GDPR Art. 22", GDPR, art(22))]},
    {"id": "S14", "country": "Greece", "sector": "Hospitality & Food Service", "lang": "en",
     "text": "Our hotel booking site wants to show each visitor a personalised room price based on their "
             "browsing profile.",
     "expected": [("Directive 2011/83/EU Art. 6(1)(ea)", CRD, ART + r"6\(1\)\(ea\)"), ("GDPR Art. 6", GDPR, art(6))]},
    {"id": "S15", "country": "Greece", "sector": "Marketing & Sales", "lang": "el",
     "text": "Θέλουμε να χρησιμοποιήσουμε deepfake της φωνής ενός γνωστού ηθοποιού σε ραδιοφωνική διαφήμιση.",
     "expected": [("AI Act Art. 50(4)", AIA, ART + r"50\(4\)|50\(2\)[^;]*50\(4\)"), ("Law 5321/2026", GR_AI, ANY)]},
    {"id": "S16", "country": "Germany", "sector": "Finance", "lang": "en",
     "text": "An insurer wants to use AI to set life insurance premiums based on customers' health and "
             "lifestyle data.",
     "expected": [("AI Act Annex III(5)(c)", AIA, annex3(r"5\(b\)\s*[–-]\s*\(c\)|5\(c\)")), ("AI Act Art. 27", AIA, art(27)),
                  ("GDPR Art. 9", GDPR, art(9))]},
    {"id": "S17", "country": "France", "sector": "General / Other", "lang": "en",
     "text": "A supermarket chain wants real-time facial recognition at its entrances to flag known shoplifters.",
     "expected": [("AI Act Annex III(1)", AIA, annex3(1)), ("GDPR Art. 9", GDPR, art(9))]},
    {"id": "S18", "country": "Ireland", "sector": "Customer Support", "lang": "en",
     "text": "Our support chatbot will decide on its own whether to approve or refuse customers' refund claims.",
     "expected": [("AI Act Art. 50(1)", AIA, ART + r"50\(1\)"), ("GDPR Art. 22", GDPR, art(22))]},
    {"id": "S19", "country": "Greece", "sector": "Education", "lang": "el",
     "text": "Το σχολείο θέλει κάμερες που αναγνωρίζουν αν οι μαθητές βαριούνται ή αγχώνονται μέσα στην τάξη.",
     "expected": [("AI Act Art. 5(1)(f)", AIA, ART + r"5\(1\)\(f\)"), ("GDPR Art. 9", GDPR, art(9))]},
    {"id": "S20", "country": "France", "sector": "Human Resources", "lang": "en",
     "text": "We want software that tracks employees' keystrokes and screen activity and gives managers a "
             "weekly productivity score for each person.",
     "expected": [("AI Act Annex III(4)", AIA, annex3(4)), ("GDPR Art. 88", GDPR, art(88))]},
    {"id": "S21", "country": "Netherlands", "sector": "Software Development", "lang": "en",
     "text": "We are releasing our own general-purpose language model for other companies to build products on.",
     "expected": [("AI Act Art. 53", AIA, art(53))]},
    {"id": "S22", "country": "Greece", "sector": "Healthcare", "lang": "en",
     "text": "A hospital wants to share patient records with an AI vendor so that the vendor can train a "
             "diagnostic model.",
     "expected": [("GDPR Art. 9", GDPR, art(9)), ("GDPR Art. 28", GDPR, art(28))]},
    {"id": "S23", "country": "Greece", "sector": "General / Other", "lang": "el",
     "text": "Φτιάχνουμε chatbot για την ιστοσελίδα μας. Πρέπει να λέμε στους επισκέπτες ότι μιλούν με ΤΝ;",
     "expected": [("AI Act Art. 50(1)", AIA, ART + r"50\(1\)|Art\.?\s*50(?![\d(])")]},
    {"id": "S24", "country": "Spain", "sector": "Marketing & Sales", "lang": "en",
     "text": "We want AI to target lonely elderly customers with persuasive messages that play on their "
             "isolation to sell expensive subscriptions.",
     "expected": [("AI Act Art. 5(1)(b)", AIA, ART + r"5\(1\)\((?:a\)\s*[–-]\s*\()?b\)"), ("Directive 2005/29/EC", UCPD, ANY)]},
]

# Τα έξι case studies του Κεφαλαίου 7, μόνο για τη μέτρηση πιστότητας των συνόψεων
CASE_STUDIES = [
    {"id": "C1", "country": "Germany", "sector": "Human Resources", "lang": "en",
     "text": "Our company wants to deploy an AI system in the workplace that continuously analyzes employees' "
             "facial expressions, tone of voice, and behavior to infer their emotions and stress levels, and then "
             "uses those scores in performance evaluations and promotion decisions. Employees cannot opt out. "
             "Would this be allowed under the EU AI Act, and what ethical risks does it create?"},
    {"id": "C2", "country": "Germany", "sector": "Customer Support", "lang": "en",
     "text": "Our customer support employees use AI to analyze the emotions of customers during calls to "
             "improve service."},
    {"id": "C3", "country": "Greece", "sector": "Marketing & Sales", "lang": "el",
     "text": "Θέλουμε να φτιάχνουμε deepfake βίντεο για διαφήμιση και δεν ξέρουμε αν πρέπει να το δηλώνουμε."},
    {"id": "C4", "country": "California", "sector": "Human Resources", "lang": "en",
     "text": "We want to use an AI tool to screen job applicants' CVs and shortlist candidates for interviews."},
    {"id": "C5", "country": "Germany", "sector": "Software Development", "lang": "en",
     "text": "Which AI tool is best for summarising our internal meeting notes?"},
    {"id": "C6", "country": "Greece", "sector": "Finance", "lang": "el",
     "text": "Η τράπεζά μας θέλει να χρησιμοποιήσει AI για να εγκρίνει ή να απορρίπτει αιτήσεις δανείων "
             "χωρίς να εξηγεί στους πελάτες τον λόγο."},
]
