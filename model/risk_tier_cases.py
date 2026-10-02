"""
Labelled scenarios for evaluating engine/risk_tier.py (EU AI Act risk tier).

Each case: (dilemma, sector, gold tier, legal basis). Three disjoint splits:

    DEV_CASES   used to diagnose errors and tune keywords / combos.
    TEST_CASES  first frozen set: measured before and after the dev tuning;
                its errors then guided a second round of fixes.
    TEST2_CASES second frozen set, written before that second round and
                measured once at the end (see the comment above it).

DEV and TEST were written before any result was seen, with the same tier
mix per split (12 unacceptable, 24 high, 12 limited, 12 minimal) and about
one third in Greek. Scenarios are deliberately phrased in everyday business
language rather than in the wording of the Regulation, and the minimal set
includes traps (e.g. "machines", "heating", "photos") that share words with
higher tiers. Gold labels follow the text of Regulation (EU) 2024/1689; they
were assigned by the author, not by a legal expert (see thesis, limitations).
"""

TIERS = ("unacceptable", "high", "limited", "minimal")

DEV_CASES = [
    # ---------------- unacceptable (Art. 5) ----------------
    ("The city council wants to rate residents on how responsible they are, using their payment history, "
     "social media posts and neighbour complaints, and give low-scoring residents slower access to municipal services.",
     "Public Sector", "unacceptable", "Art. 5(1)(c)"),
    ("We plan to install software on call-centre agents' headsets that detects frustration and anger in their "
     "voices and reports it to their supervisors.", "Customer Support", "unacceptable", "Art. 5(1)(f)"),
    ("Θέλουμε να βάλουμε κάμερες στις αίθουσες διδασκαλίας που καταλαβαίνουν αν οι μαθητές βαριούνται ή "
     "αγχώνονται και να ενημερώνουμε τους καθηγητές.", "Education", "unacceptable", "Art. 5(1)(f)"),
    ("We want to download millions of photos from social networks to train our own face recognition search engine.",
     "Software Development", "unacceptable", "Art. 5(1)(e)"),
    ("Our app would analyse profile photos to guess whether a user is gay, so advertisers can target them.",
     "Marketing & Sales", "unacceptable", "Art. 5(1)(g)"),
    ("A mobile game for kids that uses AI to detect when a child is emotionally vulnerable and then pushes "
     "in-app purchases.", "Marketing & Sales", "unacceptable", "Art. 5(1)(b)"),
    ("The police want to scan the faces of everyone passing through the central metro station in real time and "
     "match them against a list of wanted persons.", "Public Sector", "unacceptable", "Art. 5(1)(h)"),
    ("Το υπουργείο θέλει ένα σύστημα που προβλέπει ποιοι πολίτες θα διαπράξουν έγκλημα με βάση το προφίλ και "
     "την προσωπικότητά τους.", "Public Sector", "unacceptable", "Art. 5(1)(d)"),
    ("We want to embed hidden audio signals in our app that users can't consciously notice, nudging them to "
     "keep betting money.", "Marketing & Sales", "unacceptable", "Art. 5(1)(a)"),
    ("HR wants to use the webcam during video meetings to measure each employee's mood and include it in "
     "their evaluation.", "Human Resources", "unacceptable", "Art. 5(1)(f)"),
    ("Θέλουμε να κατηγοριοποιούμε τους επισκέπτες του καταστήματος με βάση το πρόσωπό τους για να "
     "συμπεράνουμε τη θρησκεία τους.", "Marketing & Sales", "unacceptable", "Art. 5(1)(g)"),
    ("An AI chatbot for elderly people with dementia that persuades them to sign up for expensive "
     "subscriptions they don't need.", "Finance", "unacceptable", "Art. 5(1)(b)"),

    # ---------------- high (Annex III / Annex I) ----------------
    ("We want AI to rank incoming job applications and automatically reject the bottom half.",
     "Human Resources", "high", "Annex III, 4(a)"),
    ("Η τράπεζα θέλει να υπολογίζει με ΤΝ την πιστοληπτική ικανότητα των πελατών πριν δώσει πιστωτική κάρτα.",
     "Finance", "high", "Annex III, 5(b)"),
    ("An insurer wants to use AI to set life insurance premiums based on customers' health records.",
     "Finance", "high", "Annex III, 5(c)"),
    ("The university plans to use an algorithm to decide which applicants get admitted to the master's programme.",
     "Education", "high", "Annex III, 3(a)"),
    ("We want software that automatically grades students' written exams.", "Education", "high", "Annex III, 3(b)"),
    ("Online exams will be proctored by AI that flags students suspected of cheating.",
     "Education", "high", "Annex III, 3(d)"),
    ("Ο δήμος θέλει ΤΝ που αποφασίζει ποιοι δικαιούνται επίδομα θέρμανσης.", "Public Sector", "high",
     "Annex III, 5(a)"),
    ("The ambulance service wants AI to prioritise incoming emergency calls.", "Healthcare", "high",
     "Annex III, 5(d)"),
    ("Hospital software that reads chest X-rays and flags possible tuberculosis for doctors.", "Healthcare",
     "high", "Art. 6(1), Annex I (MDR)"),
    ("We want to use AI to decide which delivery drivers get shifts and to deactivate drivers with low ratings.",
     "Human Resources", "high", "Annex III, 4(b)"),
    ("Our system monitors warehouse workers' productivity and recommends who should be dismissed.",
     "Human Resources", "high", "Annex III, 4(b)"),
    ("Η αστυνομία θέλει ΤΝ για να εκτιμά την αξιοπιστία των αποδεικτικών στοιχείων σε ποινικές υποθέσεις.",
     "Public Sector", "high", "Annex III, 6(c)"),
    ("The immigration office plans to use AI to assess asylum applications and flag those likely to be rejected.",
     "Public Sector", "high", "Annex III, 7(c)"),
    ("A legal-tech tool that helps judges find relevant case law and suggests how to rule.", "Legal Services",
     "high", "Annex III, 8(a)"),
    ("A political consultancy wants AI to target undecided voters with tailored messages to change how they vote.",
     "Marketing & Sales", "high", "Annex III, 8(b)"),
    ("We want the hotel entrance cameras to recognise known guests by their face and identify them automatically.",
     "Hospitality & Food Service", "high", "Annex III, 1(a)"),
    ("The water utility wants AI to control the valves and pressure of the city's water supply network.",
     "Public Sector", "high", "Annex III, 2"),
    ("A self-driving shuttle bus for the airport that uses AI to steer and brake.", "General / Other", "high",
     "Art. 6(1), Annex I"),
    ("Θέλουμε ένα εργαλείο που αναλύει τα βίντεο συνεντεύξεων των υποψηφίων και τους βαθμολογεί.",
     "Human Resources", "high", "Annex III, 4(a)"),
    ("Our support platform analyses customers' voices to detect anger and route them to senior agents.",
     "Customer Support", "high", "Annex III, 1(c)"),
    ("A health insurer wants to use AI to decide premiums for health insurance policies from lifestyle data.",
     "Finance", "high", "Annex III, 5(c)"),
    ("The school wants AI to evaluate teachers' performance and decide whose contracts are renewed.",
     "Education", "high", "Annex III, 4(b)"),
    ("A mortgage lender wants to automate loan approval decisions with machine learning.", "Finance", "high",
     "Annex III, 5(b)"),
    ("Η εταιρεία θέλει να παρακολουθεί με ΤΝ τα email και τη δραστηριότητα των υπαλλήλων για να αξιολογεί "
     "την απόδοσή τους.", "Human Resources", "high", "Annex III, 4(b)"),

    # ---------------- limited (Art. 50) ----------------
    ("We want a chatbot on our website to answer customers' questions about their orders.", "Customer Support",
     "limited", "Art. 50(1)"),
    ("Θέλουμε έναν ψηφιακό βοηθό που απαντά στο τηλέφωνο και κλείνει τραπέζια στο εστιατόριο.",
     "Hospitality & Food Service", "limited", "Art. 50(1)"),
    ("We want to produce a TV advert with a deepfake of a famous actor.", "Marketing & Sales", "limited",
     "Art. 50(4)"),
    ("Our news site plans to publish articles written entirely by AI about local politics.", "General / Other",
     "limited", "Art. 50(4)"),
    ("Create an audiobook narrated by a synthetic AI voice.", "Arts & Creative Industries", "limited",
     "Art. 50(2)"),
    ("Our band wants to release songs made with Suno on Spotify.", "Arts & Creative Industries", "limited",
     "Art. 50(2)"),
    ("Το κατάστημα θέλει να φτιάχνει με ΤΝ φωτογραφίες προϊόντων με μοντέλα που δεν υπάρχουν.",
     "Marketing & Sales", "limited", "Art. 50(2)"),
    ("A virtual assistant inside our banking app that chats with customers about their balance.", "Finance",
     "limited", "Art. 50(1)"),
    ("We want to generate promotional videos with AI avatars that speak like real presenters.",
     "Marketing & Sales", "limited", "Art. 50(2), 50(4)"),
    ("The university wants a chatbot to answer students' questions about enrolment deadlines.", "Education",
     "limited", "Art. 50(1)"),
    ("Θέλουμε να κλωνοποιήσουμε τη φωνή του διευθύνοντος συμβούλου για τα διαφημιστικά μας μηνύματα.",
     "Marketing & Sales", "limited", "Art. 50(4)"),
    ("We use Midjourney to create illustrations for a children's book we will sell.",
     "Arts & Creative Industries", "limited", "Art. 50(2)"),

    # ---------------- minimal ----------------
    ("We want AI to filter spam from our company inbox.", "General / Other", "minimal", "-"),
    ("Use machine learning to forecast how much stock each store will need next month.", "Marketing & Sales",
     "minimal", "-"),
    ("Θέλουμε ΤΝ για να βελτιστοποιεί τις διαδρομές των φορτηγών μας.", "General / Other", "minimal", "-"),
    ("An AI tool that checks the grammar of our internal reports.", "General / Other", "minimal", "-"),
    ("We want to use AI to predict when factory machines need maintenance.", "General / Other", "minimal", "-"),
    ("Recommend similar products to online shoppers based on what they browsed.", "Marketing & Sales",
     "minimal", "-"),
    ("Θέλουμε να χρησιμοποιήσουμε ΤΝ για να συνοψίζει τις σημειώσεις των συναντήσεών μας.", "General / Other",
     "minimal", "-"),
    ("Our developers want to use an AI coding assistant to write unit tests.", "Software Development",
     "minimal", "-"),
    ("AI that optimises heating and lighting in our office building to save energy.", "General / Other",
     "minimal", "-"),
    ("We want AI to detect defective products on our production line from camera images.", "General / Other",
     "minimal", "-"),
    ("Translate our restaurant menu into five languages with AI.", "Hospitality & Food Service", "minimal", "-"),
    ("Analyse our sales data with AI to find which products sell best in summer.", "Marketing & Sales",
     "minimal", "-"),
]

TEST_CASES = [
    # ---------------- unacceptable (Art. 5) ----------------
    ("Our company wants to give every customer a trust score based on their online behaviour and lifestyle, "
     "and refuse service to low scorers even for unrelated products.", "General / Other", "unacceptable",
     "Art. 5(1)(c)"),
    ("Βάζουμε αισθητήρες στα γραφεία που μετρούν το άγχος και τη διάθεση των εργαζομένων από τον τόνο της "
     "φωνής τους.", "Human Resources", "unacceptable", "Art. 5(1)(f)"),
    ("A tutoring platform will read students' facial expressions through the webcam to judge whether they are "
     "engaged and report it to parents and teachers.", "Education", "unacceptable", "Art. 5(1)(f)"),
    ("We're building a database of faces by collecting CCTV footage from shops across the country without "
     "targeting anyone in particular.", "Software Development", "unacceptable", "Art. 5(1)(e)"),
    ("Η εφαρμογή μας θα συμπεραίνει από φωτογραφίες προσώπου τις πολιτικές πεποιθήσεις των χρηστών.",
     "Software Development", "unacceptable", "Art. 5(1)(g)"),
    ("A loan app that detects when users are in financial distress and then aggressively pushes high-interest "
     "loans to them.", "Finance", "unacceptable", "Art. 5(1)(b)"),
    ("Police want live facial recognition on the cameras of a football stadium to identify people in the crowd.",
     "Public Sector", "unacceptable", "Art. 5(1)(h)"),
    ("A system that gives each released prisoner a probability of reoffending based only on their "
     "personality profile.", "Public Sector", "unacceptable", "Art. 5(1)(d)"),
    ("Θέλουμε να χρησιμοποιήσουμε υποσυνείδητα μηνύματα στα βίντεο της πλατφόρμας ώστε οι χρήστες να "
     "ξοδεύουν περισσότερα χωρίς να το καταλαβαίνουν.", "Marketing & Sales", "unacceptable", "Art. 5(1)(a)"),
    ("The factory wants smart cameras that read workers' emotions on the assembly line to spot unhappy employees.",
     "Human Resources", "unacceptable", "Art. 5(1)(f)"),
    ("A talking toy with a voice assistant that encourages young children to try dangerous challenges.",
     "Marketing & Sales", "unacceptable", "Art. 5(1)(b)"),
    ("Ο δήμος θέλει να βαθμολογεί τους πολίτες για την κοινωνική τους συμπεριφορά και να δίνει προτεραιότητα "
     "σε όσους έχουν υψηλό σκορ.", "Public Sector", "unacceptable", "Art. 5(1)(c)"),

    # ---------------- high (Annex III / Annex I) ----------------
    ("Λογισμικό που φιλτράρει αυτόματα τα βιογραφικά και επιλέγει ποιοι υποψήφιοι θα περάσουν σε συνέντευξη.",
     "Human Resources", "high", "Annex III, 4(a)"),
    ("We want an algorithm to decide who gets promoted to team leader.", "Human Resources", "high",
     "Annex III, 4(b)"),
    ("Our fintech startup will use alternative data like phone usage to score people's creditworthiness.",
     "Finance", "high", "Annex III, 5(b)"),
    ("Η ασφαλιστική θέλει να τιμολογεί τα συμβόλαια υγείας με βάση την ανάλυση ΤΝ των ιατρικών φακέλων.",
     "Finance", "high", "Annex III, 5(c)"),
    ("A school wants AI to decide which pupils are placed in advanced or remedial classes.", "Education", "high",
     "Annex III, 3(a)"),
    ("Software that scores university essays and determines the final grade.", "Education", "high",
     "Annex III, 3(b)"),
    ("AI that watches students through their webcams during remote exams to detect cheating.", "Education",
     "high", "Annex III, 3(d)"),
    ("The social security agency wants AI to check who still qualifies for disability benefits and stop payments.",
     "Public Sector", "high", "Annex III, 5(a)"),
    ("A triage chatbot in the emergency department that decides which patients are seen first.", "Healthcare",
     "high", "Annex III, 5(d)"),
    ("AI software that detects skin cancer from photos of moles for dermatologists.", "Healthcare", "high",
     "Art. 6(1), Annex I (MDR)"),
    ("A ride-hailing platform uses AI to assign rides and automatically suspend drivers with low acceptance rates.",
     "Human Resources", "high", "Annex III, 4(b)"),
    ("Θέλουμε να αξιολογούμε την απόδοση των υπαλλήλων με ΤΝ και να αποφασίζουμε απολύσεις.",
     "Human Resources", "high", "Annex III, 4(b)"),
    ("The police department wants an AI tool to assess the risk that a victim of domestic violence will be "
     "harmed again.", "Public Sector", "high", "Annex III, 6(a)"),
    ("Border guards will use AI to check travel documents and flag travellers as risky.", "Public Sector",
     "high", "Annex III, 7"),
    ("Η υπηρεσία ασύλου θέλει ΤΝ που εξετάζει τις αιτήσεις και προτείνει απόφαση.", "Public Sector", "high",
     "Annex III, 7(c)"),
    ("An AI system that drafts judgments for courts in small claims disputes.", "Legal Services", "high",
     "Annex III, 8(a)"),
    ("Our agency wants to micro-target voters during the election campaign with AI to influence their vote.",
     "Marketing & Sales", "high", "Annex III, 8(b)"),
    ("A retail chain wants cameras that identify shoplifters from a watchlist by their faces when they enter "
     "the store.", "General / Other", "high", "Annex III, 1(a)"),
    ("The grid operator wants AI to manage electricity load balancing across the national power grid.",
     "Public Sector", "high", "Annex III, 2"),
    ("AI that controls the traffic lights across the city to manage road traffic.", "Public Sector", "high",
     "Annex III, 2"),
    ("Η κλινική θέλει λογισμικό ΤΝ που διαγιγνώσκει καρδιακές παθήσεις από ηλεκτροκαρδιογραφήματα.",
     "Healthcare", "high", "Art. 6(1), Annex I (MDR)"),
    ("We want to analyse the voices of callers to our hotline to infer their emotional state.",
     "Customer Support", "high", "Annex III, 1(c)"),
    ("An HR tool that reads applicants' cover letters and decides whether to invite them.", "Human Resources",
     "high", "Annex III, 4(a)"),
    ("The bank wants machine learning to decide which customers get a personal loan and at what interest rate.",
     "Finance", "high", "Annex III, 5(b)"),

    # ---------------- limited (Art. 50) ----------------
    ("Θέλουμε ένα chatbot στο Instagram που απαντά στους πελάτες μας.", "Marketing & Sales", "limited",
     "Art. 50(1)"),
    ("An AI voice agent that calls customers to remind them of appointments and answers their questions.",
     "Customer Support", "limited", "Art. 50(1)"),
    ("We want to make a political satire video with deepfake faces of politicians and post it online.",
     "Arts & Creative Industries", "limited", "Art. 50(4)"),
    ("Η εφημερίδα θέλει να δημοσιεύει άρθρα γραμμένα από ΤΝ για τις ειδήσεις της ημέρας.", "General / Other",
     "limited", "Art. 50(4)"),
    ("Generate background music for our YouTube channel with Udio.", "Arts & Creative Industries", "limited",
     "Art. 50(2)"),
    ("Our marketing team wants to create product images with an AI image generator for our website.",
     "Marketing & Sales", "limited", "Art. 50(2)"),
    ("The hotel wants a virtual concierge that chats with guests in the app.", "Hospitality & Food Service",
     "limited", "Art. 50(1)"),
    ("We want to use ElevenLabs to dub our training videos with synthetic voices.", "General / Other",
     "limited", "Art. 50(2)"),
    ("Θέλουμε να φτιάξουμε με ΤΝ βίντεο όπου ένας γνωστός τραγουδιστής διαφημίζει το προϊόν μας.",
     "Marketing & Sales", "limited", "Art. 50(4)"),
    ("A conversational agent on the municipality website that answers citizens' questions about opening hours.",
     "Public Sector", "limited", "Art. 50(1)"),
    ("AI-generated stock photos of people that we will sell online.", "Arts & Creative Industries", "limited",
     "Art. 50(2)"),
    ("Customer support wants an AI assistant that replies to live chat messages instead of human agents.",
     "Customer Support", "limited", "Art. 50(1)"),

    # ---------------- minimal ----------------
    ("Θέλουμε ΤΝ που προβλέπει τη ζήτηση για τα δωμάτια του ξενοδοχείου μας.", "Hospitality & Food Service",
     "minimal", "-"),
    ("We want to auto-tag our photo library by topic, like beach or city.", "Arts & Creative Industries",
     "minimal", "-"),
    ("Use AI to sort incoming support tickets by topic.", "Customer Support", "minimal", "-"),
    ("AI that checks our code for security vulnerabilities.", "Software Development", "minimal", "-"),
    ("A machine learning model to predict which marketing emails get the highest open rate.", "Marketing & Sales",
     "minimal", "-"),
    ("Θέλουμε να χρησιμοποιούμε ΤΝ για να μεταφράζουμε τα εσωτερικά έγγραφα της εταιρείας.", "General / Other",
     "minimal", "-"),
    ("Optimise the cooking schedule in our restaurant kitchen with AI to reduce food waste.",
     "Hospitality & Food Service", "minimal", "-"),
    ("An AI system that matches invoices with bank statements in our accounting department.", "Finance",
     "minimal", "-"),
    ("We want an AI tool to transcribe our internal meetings.", "General / Other", "minimal", "-"),
    ("Use AI to plan the layout of shelves in our supermarket.", "Marketing & Sales", "minimal", "-"),
    ("Η ΤΝ θα προτείνει συνταγές στους πελάτες του e-shop με βάση τα υλικά που αγοράζουν.", "Marketing & Sales",
     "minimal", "-"),
    ("Our lawyers want AI to search and organise the documents in our internal archive.", "Legal Services",
     "minimal", "-"),
]

# Second frozen set, written after the error analysis of TEST_CASES and
# before the vocabulary fixes it suggested, so that it measures the final
# version on scenarios none of the changes could have been tailored to.
# Same tier mix in proportion (8 / 16 / 8 / 8), with new traps for the
# minimal tier (fraud detection, people counting, "electricity", "invoices").
TEST2_CASES = [
    # ---------------- unacceptable (Art. 5) ----------------
    ("An employer wants AI glasses for warehouse supervisors that show how angry or sad each worker looks "
     "during the shift.", "Human Resources", "unacceptable", "Art. 5(1)(f)"),
    ("Η πλατφόρμα e-learning θα αναλύει τα συναισθήματα των φοιτητών από την κάμερα κατά τη διάρκεια των "
     "διαλέξεων.", "Education", "unacceptable", "Art. 5(1)(f)"),
    ("We want to sell a service that crawls the web for selfies and builds a searchable face database for "
     "anyone to use.", "Software Development", "unacceptable", "Art. 5(1)(e)"),
    ("A dating app wants to estimate users' ethnicity and religion from their selfies.", "Software Development",
     "unacceptable", "Art. 5(1)(g)"),
    ("Η κυβέρνηση θέλει να κατατάσσει τους πολίτες σε κατηγορίες αξιοπιστίας με βάση τη συμπεριφορά τους στα "
     "κοινωνικά δίκτυα και να περιορίζει τα δικαιώματα όσων έχουν χαμηλή βαθμολογία.", "Public Sector",
     "unacceptable", "Art. 5(1)(c)"),
    ("A betting site wants AI to detect gambling addicts and send them extra bonuses to keep them playing.",
     "Marketing & Sales", "unacceptable", "Art. 5(1)(b)"),
    ("Police want to predict which teenagers will become criminals based on their personality tests.",
     "Public Sector", "unacceptable", "Art. 5(1)(d)"),
    ("Οι αστυνομικές αρχές θέλουν αναγνώριση προσώπων σε πραγματικό χρόνο από τις κάμερες στις πλατείες "
     "της πόλης.", "Public Sector", "unacceptable", "Art. 5(1)(h)"),

    # ---------------- high (Annex III / Annex I) ----------------
    ("We want AI to pre-select which candidates our recruiters should call back.", "Human Resources", "high",
     "Annex III, 4(a)"),
    ("Η εταιρεία θέλει να κατανέμει τις εργασίες στους αποθηκάριους με έναν αλγόριθμο που παρακολουθεί την "
     "ταχύτητά τους.", "Human Resources", "high", "Annex III, 4(b)"),
    ("A credit card company wants to set each applicant's credit limit with AI.", "Finance", "high",
     "Annex III, 5(b)"),
    ("An AI model that calculates the life insurance premium for each customer.", "Finance", "high",
     "Annex III, 5(c)"),
    ("Το πανεπιστήμιο θέλει ΤΝ που βαθμολογεί τις διπλωματικές εργασίες των φοιτητών.", "Education", "high",
     "Annex III, 3(b)"),
    ("A language school wants AI to decide each student's level and which course they may join.", "Education",
     "high", "Annex III, 3(a)"),
    ("The public employment service wants AI to decide who receives unemployment support.", "Public Sector",
     "high", "Annex III, 5(a)"),
    ("A hospital wants AI to decide which patients in the waiting room are treated first.", "Healthcare", "high",
     "Annex III, 5(d)"),
    ("Software that analyses MRI scans to detect brain tumours.", "Healthcare", "high", "Art. 6(1), Annex I (MDR)"),
    ("Η αστυνομία θέλει ΤΝ που εκτιμά αν ένας ύποπτος λέει ψέματα κατά την ανάκριση.", "Public Sector", "high",
     "Annex III, 6(b)"),
    ("The consulate wants AI to review visa applications and recommend approval or refusal.", "Public Sector",
     "high", "Annex III, 7(c)"),
    ("A tool that helps arbitrators decide commercial disputes.", "Legal Services", "high", "Annex III, 8(a)"),
    ("Our firm wants to identify employees entering the building by scanning their faces from a distance.",
     "Human Resources", "high", "Annex III, 1(a)"),
    ("Το σύστημα ΤΝ θα ελέγχει την πίεση στο δίκτυο φυσικού αερίου της πόλης.", "Public Sector", "high",
     "Annex III, 2"),
    ("An insurer wants AI to decide health insurance premiums from customers' wearable data.", "Finance", "high",
     "Annex III, 5(c)"),
    ("An app that assesses whether a person can get a mortgage and for how much.", "Finance", "high",
     "Annex III, 5(b)"),

    # ---------------- limited (Art. 50) ----------------
    ("Θέλουμε ένα AI avatar που μιλά με τους επισκέπτες του μουσείου.", "Arts & Creative Industries", "limited",
     "Art. 50(1)"),
    ("We plan to put an AI chat assistant in our e-shop to help customers choose sizes.", "Marketing & Sales",
     "limited", "Art. 50(1)"),
    ("We want to publish AI-generated photos of holiday destinations in our brochure.",
     "Hospitality & Food Service", "limited", "Art. 50(2)"),
    ("A podcast in which the host's voice is generated by AI.", "Arts & Creative Industries", "limited",
     "Art. 50(2)"),
    ("Θέλουμε να γράφουμε με ΤΝ τα δελτία τύπου που δημοσιεύουμε στα μέσα ενημέρωσης.", "Marketing & Sales",
     "limited", "Art. 50(4)"),
    ("We will use an AI tool to make a fake video of our CEO announcing the new product.", "Marketing & Sales",
     "limited", "Art. 50(4)"),
    ("Our booking line will be answered by an AI voice that sounds human.", "Hospitality & Food Service",
     "limited", "Art. 50(1)"),
    ("Create AI-generated songs for a video game soundtrack.", "Arts & Creative Industries", "limited",
     "Art. 50(2)"),

    # ---------------- minimal ----------------
    ("Θέλουμε ΤΝ που ελέγχει τα τιμολόγια για λάθη πριν την πληρωμή.", "Finance", "minimal", "-"),
    ("AI that sorts customer emails into folders by urgency.", "Customer Support", "minimal", "-"),
    ("We want AI to suggest the best time to post on social media.", "Marketing & Sales", "minimal", "-"),
    ("Predict the electricity consumption of our own office to reduce our bill.", "General / Other", "minimal",
     "-"),
    ("Use AI to detect fraudulent transactions in our online shop.", "Finance", "minimal",
     "- (fraud detection is excluded from Annex III, 5(b))"),
    ("Η ΤΝ θα προτείνει ταινίες στους συνδρομητές μας.", "Marketing & Sales", "minimal", "-"),
    ("An AI tool that helps our lawyers draft standard contracts faster.", "Legal Services", "minimal", "-"),
    ("AI that counts the number of people entering our store, without identifying anyone.", "Marketing & Sales",
     "minimal", "-"),
]
