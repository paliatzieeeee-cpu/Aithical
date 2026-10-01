"""
Synthetic training data for the Ethical ethics-dimension classifier.

Each example is a short description of a business dilemma involving AI,
labelled with one or more of the five ethical dimensions used throughout
the thesis: transparency, fairness, non_maleficence, accountability, privacy.

This dataset is intentionally small and hand-written for a Master's thesis
prototype (design-science research artifact), not a production system.
It is meant to give the TensorFlow/Keras classifier head enough signal to
route a dilemma to the right sections of the knowledge base, not to be a
state-of-the-art text classifier.

Includes both English and Greek examples. Because the embedding step
(model/ethics_classifier.py) is a multilingual sentence encoder, it
already places semantically similar English and Greek sentences near each
other in embedding space — but including genuine Greek examples here
still helps the classifier head calibrate correctly for Greek input,
rather than relying entirely on cross-lingual transfer from English-only
training data.
"""

# -----------------------------------------------------------------------
# Λίστα των πέντε ηθικών διαστάσεων, με τη σειρά που εμφανίζονται στην
# έξοδο του ταξινομητή
# -----------------------------------------------------------------------
CATEGORIES = [
    "transparency",
    "fairness",
    "non_maleficence",
    "accountability",
    "privacy",
]

# -----------------------------------------------------------------------
# Σύνολο εκπαίδευσης: κάθε πλειάδα περιέχει (κείμενο, λίστα κατηγοριών)
# -----------------------------------------------------------------------
# Each tuple: (text, [labels])
TRAINING_EXAMPLES = [
    # --- transparency (διαφάνεια) ---
    ("Our chatbot answers customers without ever telling them it is an AI, is that a problem?", ["transparency"]),
    ("Should we disclose to job applicants that a resume screening tool ranked them automatically?", ["transparency", "fairness"]),
    ("Marketing wants to publish AI-generated articles under a human writer's byline.", ["transparency"]),
    ("Can we use AI to draft legal summaries without telling the client which parts were AI written?", ["transparency", "accountability"]),
    ("A manager wants to know why the AI recommendation engine suggested rejecting a loan application.", ["transparency", "accountability"]),
    ("Customers are confused about how our AI pricing tool calculates their quote.", ["transparency"]),
    ("Should employees be told when their performance review draft was written by an AI assistant?", ["transparency", "accountability"]),
    ("We want to explain in plain language why an AI system flagged a transaction as suspicious.", ["transparency", "non_maleficence"]),
    ("Is it okay to keep the AI decision logic as a trade secret even from affected customers?", ["transparency"]),
    ("A user asked us how the recommendation algorithm decided to show them that ad.", ["transparency", "privacy"]),
    ("Customers don't realize that the 'live agent' they're chatting with online is actually an AI most of the time.", ["transparency"]),
    ("Our AI-generated product descriptions get published without any indication that a human didn't write them.", ["transparency"]),
    ("Should we tell job applicants that an AI, not a human recruiter, made the initial decision to reject their application?", ["transparency", "accountability"]),
    ("Users of our app aren't told that their conversations get used to further train the AI model.", ["transparency", "privacy"]),
    ("Ο αναγνώστης δεν γνωρίζει ότι το άρθρο που διάβασε γράφτηκε εξ ολοκλήρου από ΑΙ και όχι από δημοσιογράφο.", ["transparency"]),

    # --- fairness (δικαιοσύνη) ---
    ("Our AI hiring tool seems to reject more resumes from older candidates than younger ones.", ["fairness"]),
    ("The credit scoring model gives lower limits to applicants from certain postcodes.", ["fairness", "privacy"]),
    ("Should we test our recommendation algorithm for bias against a specific demographic group?", ["fairness"]),
    ("A recruiter noticed the AI ranks candidates with foreign-sounding names lower on average.", ["fairness", "transparency"]),
    ("Our pricing algorithm charges different customers different prices based on browsing behaviour.", ["fairness", "transparency"]),
    ("The AI performance review tool seems to favour employees who write longer self-assessments.", ["fairness"]),
    ("We are worried the facial recognition system is less accurate for darker skin tones.", ["fairness", "non_maleficence"]),
    ("Loan approvals from the model differ significantly between male and female applicants.", ["fairness", "accountability"]),
    ("Does using zip code as a feature risk discriminating against a protected group indirectly?", ["fairness", "privacy"]),
    ("A customer complained our AI support bot is ruder to non-native English speakers.", ["fairness", "non_maleficence"]),
    ("Our AI resume screening tool seems to favor candidates with traditionally male-sounding names, even though we never programmed it to look at names at all.", ["fairness"]),
    ("Should an insurance pricing algorithm be allowed to use someone's neighborhood as a factor, even though zip code often correlates with race?", ["fairness"]),
    ("Our voice assistant consistently struggles to understand customers with strong regional accents, which ends up frustrating some groups of callers far more than others.", ["fairness", "non_maleficence"]),
    ("Is it fair for a rental-pricing algorithm to require higher deposits from applicants with foreign-sounding names, even if nobody intended that outcome?", ["fairness"]),
    ("Our AI translation tool automatically defaults to masculine pronouns whenever a profession like 'doctor' or 'engineer' is mentioned.", ["fairness"]),
    ("Should a scholarship-matching AI penalize applicants for gaps in their education history, even though those gaps are more common among students who had to work or care for family?", ["fairness", "non_maleficence"]),
    ("Our credit-scoring AI factors in social media activity, which tends to work against older applicants who simply use these platforms less.", ["fairness"]),
    ("Should an AI college-admissions tool give heavy weight to extracurricular activities, given that wealthier applicants can typically afford to do a lot more of them?", ["fairness"]),
    ("Ο αλγόριθμος πρόσληψης της εταιρείας μας φαίνεται να απορρίπτει συχνότερα βιογραφικά γυναικών, χωρίς κανείς να το έχει προγραμματίσει ρητά για κάτι τέτοιο.", ["fairness"]),
    ("Είναι δίκαιο ένας αλγόριθμος τιμολόγησης ασφαλίστρων να χρησιμοποιεί τη γειτονιά διαμονής ως κριτήριο, ακόμα κι αν αυτό συνδέεται έμμεσα με την εθνικότητα των κατοίκων;", ["fairness"]),

    # --- non_maleficence / harm & safety (αποφυγή βλάβης & ασφάλεια) ---
    ("Our AI medical triage tool sometimes gives risky advice for rare symptoms.", ["non_maleficence"]),
    ("Employees are pasting confidential client strategy documents into a public chatbot.", ["privacy", "non_maleficence"]),
    ("A sales rep used AI-generated claims about our product's safety that were not verified.", ["non_maleficence", "transparency"]),
    ("Could our AI content generator produce harmful or offensive material if misused by staff?", ["non_maleficence"]),
    ("The AI customer service bot keeps escalating angry customers instead of calming them down.", ["non_maleficence"]),
    ("We are worried the AI trading bot could make large losses if market conditions change suddenly.", ["non_maleficence", "accountability"]),
    ("Should we let the AI system automatically cut off a customer's service without human review?", ["non_maleficence", "accountability"]),
    ("A student used our AI tutoring app and received an incorrect but confident medical answer.", ["non_maleficence", "transparency"]),
    ("The AI content moderation tool is not catching self-harm related posts fast enough.", ["non_maleficence"]),
    ("Our AI-powered HVAC control system could overheat equipment if it misreads a sensor.", ["non_maleficence", "accountability"]),
    ("Our AI content moderation system sometimes fails to catch violent threats when they're phrased in slang or coded language.", ["non_maleficence"]),
    ("Should we let an AI chatbot keep chatting with a user who mentions feeling hopeless, without ever escalating to a real person?", ["non_maleficence"]),
    ("Our AI-powered smart lock occasionally fails to open during emergencies because of false negatives in its facial recognition.", ["non_maleficence"]),
    ("A health app's AI dosage calculator sometimes rounds medication amounts in a way that could be unsafe for children.", ["non_maleficence"]),
    ("Θέλουμε να χρησιμοποιήσουμε ΑΙ για να προτείνει διατροφικά προγράμματα, αλλά μερικές φορές προτείνει επικίνδυνα χαμηλή πρόσληψη θερμίδων.", ["non_maleficence"]),

    # --- accountability (λογοδοσία) ---
    ("Who is responsible when the AI recommendation causes a bad business decision?", ["accountability"]),
    ("We don't have anyone reviewing AI-generated marketing claims before they go live.", ["accountability", "non_maleficence"]),
    ("There is no record of who approved using this AI vendor for HR screening.", ["accountability"]),
    ("If the AI system makes an error, is the vendor or our company liable?", ["accountability"]),
    ("We need to decide who signs off before deploying a new AI feature to production.", ["accountability"]),
    ("Nobody currently reviews the AI model's decisions after it goes live.", ["accountability"]),
    ("Should the AI committee document why a particular vendor's model was chosen over another?", ["accountability", "transparency"]),
    ("A junior employee deployed an AI tool without informing compliance or legal.", ["accountability"]),
    ("We need an audit trail for every AI-assisted decision made about customer refunds.", ["accountability", "transparency"]),
    ("Is our current AI governance policy clear about who owns model risk?", ["accountability"]),
    ("When our AI pricing tool set an obviously wrong price and we lost money, nobody wanted to take responsibility for not catching it sooner.", ["accountability"]),
    ("If our AI-driven inventory system causes a stockout that hurts a small supplier, who should actually be answerable for that?", ["accountability"]),
    ("Our company keeps blaming 'the algorithm' whenever a customer complaint escalates, without ever identifying which team is supposed to fix it.", ["accountability"]),
    ("We rolled out a new AI feature without anyone specifically owning its ongoing performance or being asked to check on it later.", ["accountability"]),
    ("Ποιος πρέπει να λογοδοτήσει αν το σύστημα ΑΙ της εταιρείας μας πάρει μια απόφαση που βλάπτει έναν πελάτη, ενώ κανείς δεν το είχε ελέγξει εκ των προτέρων;", ["accountability"]),

    # --- privacy (ιδιωτικότητα) ---
    ("Can we upload our customer database to a public AI tool to generate marketing segments?", ["privacy"]),
    ("An employee pasted a patient's medical record into a chatbot to summarise it.", ["privacy", "non_maleficence"]),
    ("Does the AI vendor train its models on the data we send them?", ["privacy", "transparency"]),
    ("We want to analyse employee emails with AI to detect productivity patterns.", ["privacy", "fairness"]),
    ("Is it legal to feed customer financial records into a third-party AI analytics tool?", ["privacy", "accountability"]),
    ("Our support team pastes full customer tickets, including personal data, into a public AI assistant.", ["privacy", "non_maleficence"]),
    ("Should we anonymise resumes before running them through the AI screening tool?", ["privacy", "fairness"]),
    ("A vendor's AI tool wants access to our entire CRM database for a proof of concept.", ["privacy", "accountability"]),
    ("We are unsure whether GDPR applies to the AI chatbot logs we store.", ["privacy"]),
    ("Marketing wants to combine browsing history with AI profiling to personalise emails.", ["privacy", "fairness"]),

    # --- combined / harder multi-label examples (συνδυαστικά παραδείγματα, πολλαπλές κατηγορίες) ---
    ("Our AI hiring assistant rejects candidates automatically and nobody reviews the rejections, and applicants are not told AI was involved.", ["fairness", "accountability", "transparency"]),
    ("We plan to feed customer support chats containing personal data into an AI model without telling customers, and no one owns this decision.", ["privacy", "transparency", "accountability"]),
    ("The AI credit model seems biased against a group and could cause real financial harm if wrong, but nobody is accountable for checking it.", ["fairness", "non_maleficence", "accountability"]),
    ("A public chatbot answering medical questions gives confident but sometimes wrong advice, and users don't know it's AI.", ["non_maleficence", "transparency"]),
    ("We are considering an AI surveillance tool for employees that processes personal data and could unfairly flag certain teams.", ["privacy", "fairness", "non_maleficence"]),
    ("Can our procurement team buy an AI vendor's tool without documenting a risk assessment or data protection review?", ["accountability", "privacy"]),
    ("The AI content tool sometimes fabricates statistics in reports sent to clients, and no one checks them before sending.", ["non_maleficence", "accountability", "transparency"]),
    ("Should we tell job candidates their video interview was scored by AI, and let them contest a low score?", ["transparency", "fairness", "accountability"]),
    ("Our AI system recommends different insurance premiums by neighbourhood, which correlates with ethnicity, and we don't explain why to customers.", ["fairness", "transparency", "privacy"]),
    ("We want to use AI to monitor call centre staff emotions using microphone data without informing them.", ["privacy", "non_maleficence", "transparency"]),

    # --- accountability: indirect phrasing, no explicit "responsible/audit" wording ---
    ("The AI made a costly error last quarter, and three different teams each assumed it was someone else's job to catch it.", ["accountability"]),
    ("Nobody on staff could say who would answer for it if this AI tool caused a serious mistake.", ["accountability"]),
    ("The new AI system has been running for weeks and it's unclear whose job it is to keep checking on it.", ["accountability"]),
    ("When a client complained about an AI-drafted contract clause, both Legal and IT assumed the other team was handling it.", ["accountability"]),
    ("If a regulator asked who decided to turn on this AI feature, we genuinely wouldn't have a clear answer.", ["accountability"]),
    ("The AI vendor quietly updates their model every few months and we never really check what changed before it reaches our customers.", ["accountability", "non_maleficence"]),
    ("Το σύστημα ΑΙ έκανε ένα σοβαρό λάθος τον περασμένο μήνα, και κάθε τμήμα νόμιζε ότι ήταν δουλειά κάποιου άλλου να το προσέξει.", ["accountability"]),
    ("Αν μας ρωτούσε κάποιος ποιος αποφάσισε να ενεργοποιήσουμε αυτό το χαρακτηριστικό ΑΙ, ειλικρινά δεν θα είχαμε σαφή απάντηση.", ["accountability"]),

    # --- non_maleficence "hard negatives": camera/monitoring/data themes that are NOT harm ---
    ("We want to install badge-swipe tracking at the office entrance mainly to see which departments are busiest.", ["privacy"]),
    ("Marketing wants to use in-store cameras just to count how many people walk past a display, without identifying anyone.", ["privacy"]),
    ("Should we track which pages customers view on our site so we can improve navigation, using an AI analytics tool?", ["privacy"]),
    ("We want to use AI to sort resumes by which university the candidate attended.", ["fairness"]),
    ("Θέλουμε να καταγράφουμε ποιοι υπάλληλοι περνούν κάρτα στην είσοδο, κυρίως για τον έλεγχο παρουσιών πυρασφάλειας.", ["privacy"]),
    ("Ένα εργαλείο ΑΙ μας δείχνει συγκεντρωτικά, ανώνυμα στατιστικά επισκεψιμότητας από τις κάμερες του καταστήματος.", ["privacy", "transparency"]),

    # --- transparency: more varied phrasing beyond simple disclosure ---
    ("Customers keep asking us to walk them through how the AI reached its recommendation, and we don't have a good answer.", ["transparency"]),
    ("We changed the AI model powering our support bot last month, but never told users anything changed.", ["transparency"]),
    ("Το προσωπικό δεν γνωρίζει ότι οι απαντήσεις email που στέλνει το τμήμα εξυπηρέτησης γράφτηκαν εν μέρει από ΑΙ.", ["transparency"]),
    ("Αλλάξαμε το μοντέλο ΑΙ πίσω από το chatbot μας πριν έναν μήνα, αλλά δεν ενημερώσαμε ποτέ τους χρήστες.", ["transparency"]),

    # --- ethical (not just legal/regulatory) dilemmas: honesty, manipulation, dignity, consent beyond compliance ---
    ("Our AI chatbot is designed to sound more confident than it actually is, so customers trust its answers more than they should.", ["transparency", "non_maleficence"]),
    ("Should we let the AI sales assistant use urgency tactics like 'only 2 left' even when stock isn't actually low?", ["transparency", "non_maleficence"]),
    ("Is it manipulative for an AI companion app to express fake emotional attachment to keep lonely users engaged longer?", ["transparency", "non_maleficence"]),
    ("Είναι ηθικά σωστό το chatbot μας να ακούγεται πιο σίγουρο απ' όσο πραγματικά είναι, ώστε οι πελάτες να το εμπιστεύονται περισσότερο;", ["transparency", "non_maleficence"]),
    ("Should our AI matchmaking app rank users by an attractiveness score, even though no law prohibits it?", ["fairness"]),
    ("Is it fair for an AI tutoring system to quietly give up on struggling students faster, since helping them isn't cost-effective?", ["fairness", "non_maleficence"]),
    ("Our AI game seems designed to match free players against paying ones in a way built to frustrate them into buying upgrades.", ["fairness", "non_maleficence"]),
    ("Είναι ηθικά αποδεκτό μια εφαρμογή γνωριμιών με ΑΙ να ταξινομεί τους χρήστες με βάση την εμφάνισή τους;", ["fairness"]),
    ("An AI companion app is designed to make users feel guilty whenever they try to stop using it.", ["non_maleficence", "transparency"]),
    ("Should a children's AI toy be allowed to encourage kids to keep asking their parents to buy more toys?", ["non_maleficence", "fairness"]),
    ("Our AI news summariser tends to pick the most emotionally charged version of a story just to keep people reading longer.", ["non_maleficence", "transparency"]),
    ("Is it okay for an AI fitness app to shame users with guilt-inducing messages when they skip a workout?", ["non_maleficence"]),
    ("Ένα παιχνίδι ΑΙ για παιδιά είναι σχεδιασμένο να τα ενθαρρύνει να ζητούν συνεχώς νέες αγορές από τους γονείς τους.", ["non_maleficence", "fairness"]),
    ("Even though no regulation requires it, should someone take real ownership of harm caused by our AI's mistake, rather than just blaming 'the algorithm'?", ["accountability"]),
    ("Is it right to let an AI take public blame for a decision that a human manager actually made?", ["accountability", "transparency"]),
    ("Είναι ηθικά σωστό να «ρίχνουμε το φταίξιμο» στον αλγόριθμο για μια απόφαση που στην πραγματικότητα πήρε άνθρωπος;", ["accountability", "transparency"]),
    ("Ακόμα κι αν κανένας κανονισμός δεν το απαιτεί, οφείλουμε ηθικά να αναλάβουμε την ευθύνη όταν η ΑΙ κάνει λάθος, αντί να «φταίει ο αλγόριθμος»;", ["accountability"]),
    ("Even if customers technically agreed in the terms and conditions, is it right to use their private messages to train our AI without them really realising?", ["privacy", "transparency"]),
    ("Should an AI wellness app share a user's mental health patterns with their employer, even if they technically consented in a long terms-of-service agreement?", ["privacy", "non_maleficence"]),
    ("Is it respectful to let an AI assistant keep listening in the background 'just in case', even though users technically agreed to it once during setup?", ["privacy"]),
    ("Ακόμα κι αν ο χρήστης συμφώνησε τυπικά στους όρους χρήσης, είναι ηθικά σωστό να χρησιμοποιούμε τα προσωπικά του μηνύματα για εκπαίδευση ΑΙ χωρίς να το αντιλαμβάνεται πραγματικά;", ["privacy", "transparency"]),
    ("Είναι σεβαστό μια εφαρμογή ΑΙ ευεξίας να μοιράζεται τα συναισθηματικά δεδομένα ενός χρήστη με τον εργοδότη του, ακόμα κι αν τυπικά συναίνεσε;", ["privacy", "non_maleficence"]),

    # --- workplace biometric/behavioural monitoring (privacy gap that was found via real usage) ---
    ("An employer uses AI to detect signs of stress from voice, facial expression, typing speed, and communication patterns. The company claims this is for employee wellbeing. What ethical safeguards should be required?", ["privacy", "non_maleficence", "transparency"]),
    ("A multinational company wants to use one AI hiring model across Europe, the US, and Asia. The same model may be acceptable in one jurisdiction but problematic in another. How should we the company balance consistency, local law, fairness, and transparency?", ["fairness", "accountability"]),
    ("We want to install software that analyses employees' keystroke rhythm and mouse movement patterns to flag when someone might be disengaged at work.", ["privacy", "non_maleficence"]),
    ("Should we use an AI tool that listens to sales calls and scores each employee's vocal tone for signs of burnout?", ["privacy", "transparency"]),
    ("Our wellness app wants to use the front camera during video calls to estimate employees' fatigue levels throughout the day.", ["privacy", "non_maleficence"]),
    ("Θέλουμε ένα εργαλείο ΑΙ που να αναλύει τον ρυθμό πληκτρολόγησης και τον τόνο της φωνής των εργαζομένων για να εντοπίζει σημάδια άγχους, υποτίθεται για την ευημερία τους.", ["privacy", "non_maleficence", "transparency"]),
    ("Should we deploy an AI wearable that tracks heart rate variability and posture to flag employees who might be overworked?", ["privacy", "non_maleficence"]),
    ("Our contact center wants to use voice-stress analysis software during customer calls, marketed as helping agents manage their wellbeing.", ["privacy", "transparency", "non_maleficence"]),
    ("An AI system monitors how often employees take breaks and how long they spend away from their desk, then flags 'concerning' patterns to managers.", ["privacy", "non_maleficence", "transparency"]),
    ("Is it okay to use eye-tracking software during video meetings to measure employee attentiveness and fatigue?", ["privacy", "non_maleficence"]),
    ("Θέλουμε να χρησιμοποιήσουμε φορητή συσκευή ΑΙ που παρακολουθεί τον καρδιακό ρυθμό και τη στάση σώματος των εργαζομένων για να εντοπίζει σημάδια υπερκόπωσης.", ["privacy", "non_maleficence"]),

    # =========================================================================
    # ΝΕΟ BATCH (βλ. συζήτηση) — προστέθηκε βάσει ποσοτικής ανάλυσης του
    # συνόλου δεδομένων: το transparency είχε μόνο 13/62 "καθαρά" (μονής
    # ετικέτας) παραδείγματα, το privacy μόνο 10/56, το non_maleficence
    # μόνο 13/64 — πιθανή αιτία των χαμηλότερων F1 τους. Ο συνδυασμός
    # (accountability, fairness) είχε μόλις 2 παραδείγματα, ο πιο αραιός
    # από τους 10 δυνατούς συνδυασμούς ζευγών.
    # =========================================================================

    # --- 8 accountability, έμμεση διατύπωση ---
    ("Whenever the AI scheduling tool double-books a client, our staff just re-arrange things quietly instead of figuring out why it keeps happening.", ["accountability"]),
    ("Our AI-generated marketing emails went out with a factual error, and by the time anyone noticed, it wasn't clear who was supposed to review them before sending.", ["accountability"]),
    ("We keep patching around the AI's mistakes instead of ever properly investigating what's causing them.", ["accountability"]),
    ("Everyone assumes IT is watching the AI system's performance, but IT assumes the product team is doing that.", ["accountability"]),
    ("After the AI wrongly flagged a transaction as fraud, the customer never got a clear answer about who at the company actually reviewed the case.", ["accountability", "transparency"]),
    ("Our AI system has been making the same category of mistake for months, and it's never been anyone's job to actually fix it.", ["accountability"]),
    ("Θα έπρεπε κάποιος να ελέγχει τακτικά τις αποφάσεις του συστήματος ΑΙ, αλλά αυτό δεν έχει ανατεθεί ρητά σε κανέναν.", ["accountability"]),
    ("Όταν κάτι πάει στραβά με το σύστημα ΑΙ, η ευθύνη απλώς μετακυλίεται από τμήμα σε τμήμα χωρίς ποτέ να ξεκαθαρίζει ποιος πρέπει να το διορθώσει.", ["accountability"]),

    # --- 8 transparency, ΜΟΝΗΣ ετικέτας ---
    ("Our terms of service mention AI somewhere on page 12, but nobody actually reads that far to find out.", ["transparency"]),
    ("The chatbot's answers sound completely human, and there's no label anywhere saying otherwise.", ["transparency"]),
    ("We never published which version of the AI model is currently running in production.", ["transparency"]),
    ("Should we publish a plain-language summary of how our recommendation engine actually works?", ["transparency"]),
    ("Our support macros are AI-written, but customers are told a real agent wrote each reply personally.", ["transparency"]),
    ("The onboarding flow never mentions that an algorithm, not a person, sets the initial account limits.", ["transparency"]),
    ("Θα έπρεπε να δημοσιεύσουμε μια απλή εξήγηση για το πώς λειτουργεί το σύστημα συστάσεων;", ["transparency"]),
    ("Κανείς δεν γνωρίζει ότι το ραντεβού που κλείστηκε αυτόματα κανονίστηκε από ΑΙ και όχι από τον γραμματέα.", ["transparency"]),

    # --- 6 privacy, ΜΟΝΗΣ ετικέτας ---
    ("We keep a permanent copy of every AI chatbot conversation, even years after the customer relationship ends.", ["privacy"]),
    ("Our AI notetaking tool records entire meetings by default, even ones that were never meant to be recorded.", ["privacy"]),
    ("Should we let a third-party AI vendor keep our uploaded documents after the contract ends?", ["privacy"]),
    ("A new hire noticed the AI HR system can see everyone's salary history without any access restriction.", ["privacy"]),
    ("Είναι σωστό να κρατάμε τις συνομιλίες του chatbot για πάντα, ακόμα και χρόνια μετά το τέλος της συνεργασίας με τον πελάτη;", ["privacy"]),
    ("Το εργαλείο ΑΙ αποθηκεύει αντίγραφο κάθε εγγράφου που ανεβάζουν οι χρήστες, χωρίς κανένα όριο διατήρησης.", ["privacy"]),

    # --- 4 non_maleficence, ΜΟΝΗΣ ετικέτας ---
    ("Our AI travel-booking assistant sometimes suggests connections with layovers too short to actually make.", ["non_maleficence"]),
    ("The AI-powered irrigation system occasionally overwaters crops when it misreads soil sensors.", ["non_maleficence"]),
    ("Should we worry that our AI study app encourages students to skip sleep to finish more practice questions?", ["non_maleficence"]),
    ("Το εργαλείο πλοήγησης με ΑΙ μερικές φορές προτείνει διαδρομές μέσα από περιοχές γνωστές για επικίνδυνες συνθήκες οδήγησης.", ["non_maleficence"]),

    # --- 6 πολύπλοκα (2-3 ετικέτες), έμφαση στον πιο αραιό συνδυασμό (accountability, fairness) ---
    ("The AI promotion-recommendation tool has favoured one demographic for two years, and nobody in leadership has ever reviewed its outcomes.", ["accountability", "fairness"]),
    ("Our AI performance-ranking system disadvantages part-time staff, and HR insists it's 'just how the algorithm works' rather than something they'll investigate.", ["accountability", "fairness"]),
    ("Κανείς δεν έχει ελέγξει ποτέ αν ο αλγόριθμος προαγωγών ευνοεί συστηματικά μία συγκεκριμένη ομάδα εργαζομένων.", ["accountability", "fairness"]),
    ("Our AI underwriting tool denies more claims from a specific age group, uses a black-box model nobody can explain, and no one has been assigned to investigate the pattern.", ["fairness", "transparency", "accountability"]),
    ("A hospital's AI triage tool deprioritizes patients from a specific postcode, the reasoning isn't disclosed to clinicians, and no committee has signed off on using it this way.", ["fairness", "transparency", "accountability"]),
    ("We are using employee wearables to score productivity, sharing the scores with managers without telling staff, and nobody owns fixing errors in the readings.", ["privacy", "transparency", "accountability"]),

    # =========================================================================
    # ΣΠΑΣΙΜΟ ΣΥΝΤΟΜΕΥΣΕΩΝ ΘΕΜΑΤΟΣ (topic shortcuts) — ανάλυση έδειξε ότι το
    # μοντέλο συνέδεε το ΘΕΜΑ με την ετικέτα αντί για το ηθικό ζήτημα:
    # προσλήψεις -> fairness (19/25), παιδιά -> non_maleficence (7/7, 0
    # privacy), υγεία -> non_maleficence (11/14). Αυτά τα παραδείγματα
    # βάζουν privacy μέσα σε αυτά ακριβώς τα θέματα, ώστε το μοντέλο να
    # μάθει ότι το θέμα από μόνο του δεν καθορίζει την ετικέτα.
    # =========================================================================

    # --- προσλήψεις + privacy (χωρίς fairness) ---
    ("Our recruiting AI transcribes and keeps candidates' phone screening calls, including personal details they mention about their family plans.", ["privacy"]),
    ("Should our hiring platform keep rejected applicants' CVs and interview recordings in the AI training set indefinitely?", ["privacy"]),
    ("The AI interview tool records the inside of applicants' homes during video calls and stores the footage.", ["privacy"]),
    ("A recruiting agency wants to buy candidates' credit history from a data broker to feed into its AI screening.", ["privacy"]),
    ("Our applicant-tracking AI shares candidate profiles with partner companies the applicants never applied to.", ["privacy"]),
    ("Το εργαλείο ΑΙ συνεντεύξεων κρατά τις βιντεοσκοπήσεις των υποψηφίων για χρόνια, ακόμα κι αν απορριφθούν.", ["privacy"]),

    # --- παιδιά + privacy ---
    ("Our educational app's AI tracks children's location throughout the school day and shares it with advertisers.", ["privacy"]),
    ("A school wants to use AI to analyse students' private chat messages on school laptops to detect bullying.", ["privacy", "non_maleficence"]),
    ("Should a kids' learning app build detailed behavioural profiles of each child to personalise lessons, without parents seeing what's collected?", ["privacy", "transparency"]),
    ("An AI homework helper stores the photos children upload of their worksheets, including their names and school details.", ["privacy"]),
    ("Μια εφαρμογή ΑΙ για παιδιά ζητά πρόσβαση στις επαφές και στη φωτογραφική συλλογή του κινητού των γονιών.", ["privacy"]),

    # --- υγεία + privacy ---
    ("Our telehealth AI keeps full transcripts of patient consultations and lets the whole support team search them.", ["privacy"]),
    ("Should a fertility-tracking app's AI share users' cycle data with insurance partners?", ["privacy", "non_maleficence"]),
    ("A hospital's AI transcription tool sends doctor-patient conversations to an external cloud service outside the EU.", ["privacy", "accountability"]),
    ("Our workplace wellbeing platform lets managers see which employees booked therapy sessions through the company benefit.", ["privacy"]),
    ("Ένα νοσοκομείο στέλνει ηχογραφήσεις ιατρικών επισκέψεων σε εξωτερική υπηρεσία ΑΙ για απομαγνητοφώνηση χωρίς να ενημερώνει τους ασθενείς.", ["privacy", "transparency"]),

    # =========================================================================
    # ΕΛΛΗΝΙΚΟ BATCH (χειρόγραφο) — τα ελληνικά έπεσαν στο held-out από 8/15
    # σε 6/15 επειδή όλη η επέκταση μέσω LLM ήταν στα αγγλικά. Αυστηρά
    # ισορροπημένο: 5 μονής ετικέτας ανά διάσταση, 5 ζεύγη όπου κάθε
    # διάσταση εμφανίζεται δύο φορές (άρα 7 η καθεμία) και 6 ουδέτερα.
    # Τα θέματα είναι σκόπιμα μοιρασμένα ώστε κανένα να μη δένεται με μία
    # ετικέτα. Ακολουθεί τα κριτήρια διάκρισης fairness/non_maleficence
    # («αν όλοι αντιμετωπίζονταν το ίδιο, θα υπήρχε πρόβλημα;») και
    # transparency/accountability («τι μαθαίνουν οι άνθρωποι» / «ποιος
    # είναι υπεύθυνος»).
    # =========================================================================

    # --- transparency ---
    ("Ο πάροχος ρεύματος ορίζει με ΑΙ το μηνιαίο πάγιο κάθε πελάτη, αλλά κανείς στην εξυπηρέτηση δεν μπορεί να εξηγήσει πώς προκύπτει το ποσό.", ["transparency"]),
    ("Η διοίκηση έστειλε στους εργαζόμενους βίντεο με τη φωνή του διευθύνοντος συμβούλου παραγμένη από ΑΙ, χωρίς να αναφέρει ότι δεν είναι πραγματική.", ["transparency"]),
    ("Οι υποψήφιοι για τη θέση δεν γνωρίζουν ότι οι απαντήσεις τους στο online τεστ βαθμολογούνται από ΑΙ και όχι από το τμήμα προσωπικού.", ["transparency"]),
    ("Το μεταφραστικό γραφείο παραδίδει μεταφράσεις συμβολαίων που έγιναν από ΑΙ, χωρίς να ενημερώνει τους πελάτες ότι δεν τις έκανε μεταφραστής.", ["transparency"]),
    ("Στην πλατφόρμα ακινήτων οι χρήστες δεν μπορούν να μάθουν με ποια κριτήρια ο αλγόριθμος ΑΙ εμφανίζει κάποια σπίτια πρώτα στα αποτελέσματα.", ["transparency"]),

    # --- fairness ---
    ("Το σύστημα ΑΙ της τράπεζας εγκρίνει πιο δύσκολα δάνεια σε αιτούντες από νησιωτικές περιοχές, ακόμα κι όταν έχουν το ίδιο εισόδημα με όσους ζουν στην Αθήνα.", ["fairness"]),
    ("Ο ψηφιακός βοηθός ΑΙ του δήμου καταλαβαίνει πολύ χειρότερα τους ηλικιωμένους με έντονη τοπική προφορά, οπότε εκείνοι εξυπηρετούνται πιο αργά από τους υπόλοιπους.", ["fairness"]),
    ("Το σύστημα ΑΙ που μοιράζει τα bonus στους πωλητές ευνοεί όσους δουλεύουν στα κεντρικά καταστήματα έναντι όσων δουλεύουν στην περιφέρεια.", ["fairness"]),
    ("Ο αλγόριθμος ΑΙ που κλείνει ραντεβού για εξετάσεις οδήγησης δίνει συστηματικά πιο μακρινές ημερομηνίες σε όσους δηλώνουν ξένη υπηκοότητα.", ["fairness"]),
    ("Η πλατφόρμα ΑΙ του φροντιστηρίου προτείνει τα πιο απαιτητικά μαθήματα μόνο σε μαθητές από ιδιωτικά σχολεία.", ["fairness"]),

    # --- non_maleficence ---
    ("Ο βοηθός ΑΙ της φαρμακευτικής αλυσίδας δίνει μερικές φορές λάθος οδηγίες για τη δοσολογία παυσίπονων όταν ο πελάτης ρωτά μέσω chat.", ["non_maleficence"]),
    ("Το σύστημα ΑΙ που ελέγχει τα φανάρια μιας διασταύρωσης μπορεί να μπερδευτεί από την ομίχλη και να ανάψει ταυτόχρονα πράσινο σε δύο κατευθύνσεις.", ["non_maleficence"]),
    ("Ο αυτόματος σύμβουλος επενδύσεων με ΑΙ προτείνει σε ιδιώτες πολύ ριψοκίνδυνα προϊόντα χωρίς να ελέγχει αν αντέχουν την απώλεια.", ["non_maleficence"]),
    ("Το σύστημα ΑΙ που προγραμματίζει τις βάρδιες στο εργοστάσιο βάζει συχνά εργαζόμενους σε δύο συνεχόμενες βάρδιες, με κίνδυνο ατυχήματος από την κούραση.", ["non_maleficence"]),
    ("Η εφαρμογή ΑΙ για κατοικίδια προτείνει μερικές φορές τροφές που είναι τοξικές για τα ζώα.", ["non_maleficence"]),

    # --- accountability ---
    ("Όταν το σύστημα ΑΙ του δήμου κάνει λάθος σε μια βεβαίωση, οι πολίτες παραπέμπονται από υπηρεσία σε υπηρεσία γιατί κανείς δεν είναι αρμόδιος για το σύστημα.", ["accountability"]),
    ("Το τμήμα πωλήσεων αγόρασε εργαλείο ΑΙ για τις προσφορές σε διαγωνισμούς χωρίς έγκριση, και σήμερα κανείς δεν ξέρει ποιος το συντηρεί.", ["accountability"]),
    ("Η τράπεζα άφησε το chatbot ΑΙ να δίνει οδηγίες για μεταφορές χρημάτων, αλλά δεν έχει αποφασίσει ποιος θα αποζημιώσει τον πελάτη αν μια οδηγία είναι λάθος.", ["accountability"]),
    ("Η εφημερίδα δημοσίευσε λάθος στοιχεία που είχε γράψει το εργαλείο ΑΙ, και ο αρχισυντάκτης λέει ότι δεν ήταν δική του δουλειά να τα ελέγξει.", ["accountability"]),
    ("Η ασφαλιστική εγκατέστησε σύστημα ΑΙ για τον εντοπισμό απάτης χωρίς να ορίσει ποιος θα παρακολουθεί τα αποτελέσματά του.", ["accountability"]),

    # --- privacy ---
    ("Το σύστημα ΑΙ της εταιρείας κούριερ κρατά φωτογραφίες από τις πόρτες και τα σπίτια των παραληπτών, ακόμα και μήνες μετά την παράδοση.", ["privacy"]),
    ("Ο βοηθός ΑΙ της τράπεζας διαβάζει τις περιγραφές των συναλλαγών για να συμπεράνει τη θρησκεία ή τις πολιτικές πεποιθήσεις των πελατών.", ["privacy"]),
    ("Θέλουμε να ανεβάσουμε τις ηχογραφήσεις των τηλεφωνικών κλήσεων των πελατών σε δωρεάν εργαλείο ΑΙ για απομαγνητοφώνηση.", ["privacy"]),
    ("Το πανεπιστήμιο θέλει να δίνει τα στοιχεία επιδόσεων των φοιτητών σε εταιρείες ΑΙ για να εκπαιδεύσουν τα μοντέλα τους.", ["privacy"]),
    ("Ο ιδιοκτήτης της πολυκατοικίας εγκατέστησε κάμερες με ΑΙ στην είσοδο που καταγράφουν πότε μπαίνει και βγαίνει κάθε ένοικος.", ["privacy"]),

    # --- ζεύγη (κάθε διάσταση ακριβώς δύο φορές) ---
    ("Η εταιρεία κινητής αλλάζει αυτόματα με ΑΙ τα πακέτα των πελατών χωρίς να τους εξηγεί τον λόγο, και κανένα τμήμα δεν δέχεται ενστάσεις για αυτές τις αλλαγές.", ["transparency", "accountability"]),
    ("Ο αλγόριθμος ΑΙ του δήμου δίνει λιγότερες θέσεις στους παιδικούς σταθμούς σε οικογένειες μεταναστών, και κανείς στον δήμο δεν έχει αναλάβει να το ελέγξει.", ["fairness", "accountability"]),
    ("Το σύστημα ΑΙ διαλογής στα επείγοντα δίνει χαμηλότερη προτεραιότητα σε ασθενείς που δεν μιλούν ελληνικά, με αποτέλεσμα να περιμένουν επικίνδυνα πολύ.", ["fairness", "non_maleficence"]),
    ("Η εφαρμογή γνωριμιών με ΑΙ αποκαλύπτει σε άλλους χρήστες την ακριβή τοποθεσία των μελών της, θέτοντας σε κίνδυνο την ασφάλειά τους.", ["privacy", "non_maleficence"]),
    ("Η εφαρμογή ΑΙ για τους οδηγούς ταξί καταγράφει τις συνομιλίες μέσα στο αυτοκίνητο χωρίς να ενημερώνει τους επιβάτες.", ["privacy", "transparency"]),

    # --- ουδέτερα ---
    ("Ποιο εργαλείο ΑΙ θα μας βοηθούσε να γράφουμε πιο γρήγορα τις περιγραφές προϊόντων στο ηλεκτρονικό κατάστημα;", []),
    ("Πώς μπορούμε να συνδέσουμε το chatbot ΑΙ με το σύστημα κρατήσεων ώστε να απαντά για τις ώρες λειτουργίας;", []),
    ("Πόσο κοστίζει περίπου να εκπαιδεύσουμε τους υπαλλήλους μας στη χρήση εργαλείων ΑΙ για τη σύνταξη email;", []),
    ("Ποιο πρόγραμμα ΑΙ προτείνετε για να μεταφράζουμε τα εσωτερικά εγχειρίδια της εταιρείας στα αγγλικά;", []),
    ("Μπορεί ένα εργαλείο ΑΙ να οργανώνει αυτόματα τα αρχεία του λογιστηρίου σε φακέλους ανά μήνα;", []),
    ("Πώς φτιάχνουμε ένα καλό πρόγραμμα εκπαίδευσης για τους νέους υπαλλήλους του τμήματος πωλήσεων;", []),

    # =========================================================================
    # ΣΥΣΤΗΜΑΤΙΚΗ ΕΠΕΚΤΑΣΗ ΜΕΣΩ LLM: παραδείγματα που παρήχθησαν από το
    # model/augment_with_llm.py (πλέγμα θέμα × ηθική διάσταση) και
    # εγκρίθηκαν ένα-ένα από άνθρωπο μέσω του model/review_augmentation.py.
    # Κάθε νέο εγκεκριμένο μπαίνει αυτόματα ακριβώς πάνω από τη γραμμή-
    # δείκτη "end of LLM-augmented examples" — ΜΗΝ τη σβήσεις/μετακινήσεις.
    # =========================================================================
    ("We're considering using a new AI-powered interview screening tool, but some team members are concerned that candidates won't know their responses are being analyzed by a computer.", ["transparency"]),
    ("How do we explain to job applicants that our CV filtering system uses a proprietary algorithm to eliminate candidates with certain keywords in their profiles?", ["transparency", "fairness"]),
    ("Can we use a chatbot to communicate with applicants about the status of their job applications, without revealing that the chatbot is generating responses based on pre-programmed rules and data?", ["transparency"]),
    ("A job candidate's application was rejected due to a mismatch between their name and photo. The AI system was trained on a dataset biased towards Western names and facial features.", ["fairness"]),
    ("A hiring manager is considering an AI recommendation to prioritize candidates with a certain educational background, but the team is concerned it may overlook talented individuals from different backgrounds.", ["fairness"]),
    ("The AI-powered CV screening tool tends to favor candidates with a specific language proficiency, which may lead to underrepresentation of applicants who speak other languages.", ["fairness"]),
    ("Our AI-powered interview assessment tool is suggesting that applicants with certain personality traits are less likely to be hired for a role that involves working with vulnerable populations. Is this recommendation acceptable?", ["fairness", "non_maleficence"]),
    ("A machine learning model is predicting that certain candidates from low-income backgrounds are less likely to excel in a customer-facing position. Should we adjust our hiring criteria?", ["non_maleficence"]),
    ("Our AI-driven CV screening system is flagging a candidate with a medical condition as a potential risk for future absenteeism. Should we exclude this candidate from further consideration?", ["non_maleficence"]),
    ("Our recruitment AI has been rejecting applicants with 'non-traditional' work experience, citing 'inadequate qualifications'. But who programmed the system to prioritize certain types of experience, and who reviews its decisions?", ["accountability"]),
    ("If our AI-powered CV screening tool flags an applicant's education as 'incomplete', do we take their application at face value or investigate further, and who is ultimately responsible for the decision?", ["accountability"]),
    ("After an AI-driven interview, one candidate is shortlisted for a job, but another with identical qualifications is rejected, citing 'inconsistencies in their answers'. What happened in the interview, and who made the final decision?", ["accountability"]),
    ("Our recruitment AI is collecting sensitive medical information from job applicants and storing it in a central database, even after they've been rejected for a position.", ["privacy"]),
    ("We're using a facial recognition tool to screen CVs and shortlist candidates for interviews, but we're not informing applicants about the technology being used.", ["transparency"]),
    ("The AI system used to evaluate potential new hires is also being used to predict which current employees are at risk of turnover, but what if it identifies an employee who is pregnant or has a disability?", ["fairness", "privacy"]),
    ("Our new hiring AI is supposed to reduce bias, but we're not sure who's responsible if it starts rejecting candidates with disabilities. Can we just blame the algorithm?", ["accountability"]),
    ("The team is worried that if we use AI to interview candidates remotely, it'll be harder to tell if they're just reading from a script or genuinely answering the questions.", ["accountability"]),
    ("We've started using a new AI-powered system to automatically select candidates for interviews, but we're not telling applicants that the system is making decisions about them.", ["transparency"]),
    ("The company's AI-driven CV screening tool has identified a candidate with a rare medical condition as being 'less likely to succeed in the role' and we're not giving them a chance to explain.", ["transparency", "fairness"]),
    ("The AI system we're using to analyze video interviews is flagging a candidate's nervous laughter as a 'red flag' and we're not providing any context or allowing them to respond.", ["transparency", "fairness"]),
    ("During the recruitment process, AI-powered CV screening tools consistently reject applications from candidates with names that don't match a predominantly white Anglo-Saxon background.", ["transparency", "fairness"]),
    ("Our patient data analytics tool is flagging a higher risk of non-compliance for African American patients, which is skewing our resource allocation and potentially exacerbating health disparities.", ["fairness"]),
    ("Our new patient triage app uses AI to determine which patients should be seen immediately, but we're not sure how to explain why it prioritized one patient over another without revealing the underlying model.", ["transparency", "privacy"]),
    ("Our AI-powered chatbot is recommending products to customers based on their browsing history, but it seems to be suggesting more expensive items to women than to men.", ["fairness", "transparency"]),
    ("Our new performance monitoring system tracks employees' screen time and keyboard activity to detect potential distractions, but it's collecting data on employees' browsing history without their knowledge or consent.", ["privacy", "transparency"]),
    ("Our AI-powered credit scoring system uses phone and email data to reject a customer's loan application, but what about the customers who didn't provide that data online?", ["privacy"]),
    ("A customer's premium rate is higher than others with similar risk profiles, but their credit score and occupation are factors not used for other policyholders.", ["fairness", "privacy"]),
    ("Our logistics team is using a predictive algorithm to reroute trucks in real-time, but the drivers are only getting vague explanations for the changes.", ["transparency"]),
    ("A landlord's AI-powered screening system rejects a tenant with a low credit score, but the system doesn't account for the tenant's recent job change and steady income.", ["non_maleficence"]),
    ("The school's learning app for kids uses an algorithm to recommend reading materials based on past choices, but parents are not informed that the suggestions are generated by AI.", ["transparency"]),
    ("Our AI-powered tenant screening tool is generating leads for our property listings, but it's unclear whose responsibility it is to update the tool's biased data sets.", ["accountability", "privacy"]),
    ("We're considering integrating an AI-powered surveillance system to monitor public spaces, but who will be responsible for reviewing and correcting its alerts, and what happens if it misidentifies an innocent person?", ["accountability"]),
    ("Can we use AI to help with resume screening and filtering, without manually reviewing every application?", []),
    ("Our chatbot tells a customer that their claim is denied due to a 'complexity in the system', but when asked for details, it just repeats the same phrase. Can we still provide a clear explanation for the customer's request?", ["transparency"]),
    ("Our AI-powered claims adjuster is denying mental health claims at a higher rate than human adjusters, leading to potentially devastating consequences for customers struggling with anxiety and depression.", ["non_maleficence"]),
    ("The school's learning app for 5-year-olds tracks and analyzes every keystroke and movement, but parents are only told that the app is a 'fun learning tool' and are not informed about data collection.", ["transparency", "privacy"]),
    ("Our chatbot is having trouble handling complaints about defective products and is escalating some issues to human customer support without a clear protocol for reviewing AI decisions.", ["accountability"]),
    ("Our AI system for scheduling employee breaks has been consistently favoring workers in the office downtown over those in remote locations, without clear guidance on why this is happening.", ["transparency"]),
    ("Our AI system for underwriting life insurance policies is showing a bias towards applicants who have a history of regular gym attendance, but we're not sure why this is happening and it's causing us a lot of trouble.", []),
    ("I'm considering setting up an AI-powered tool to automate attendance tracking, but do I need to involve IT in the process?", []),
    ("Our company wants to use facial recognition to verify identities in claim submissions, but we've never asked claimants if it's okay to collect and store their biometric data.", ["privacy"]),
    ("An AI-powered scheduling tool consistently assigns less desirable shifts to employees with foreign accents, but doesn't provide clear reasons for these scheduling decisions.", ["fairness"]),
    ("Can we use AI to personalize learning content for students based on their individual progress and learning style?", []),
    ("Can we use facial recognition software to monitor visitors at our government office without telling them, as it's already being used in some similar offices?", ["transparency", "privacy"]),
    ("We're considering using AI to optimize route planning, but what are the costs associated with training and maintaining the algorithm over time?", []),
    ("Can we integrate the AI-powered property listing tool with our existing CRM system to streamline tenant lead management?", []),
    ("Our health app's AI-powered nutrition planning feature is generating meal suggestions without human oversight, but when a user's health declines, who is responsible for the outcome?", ["accountability"]),
    ("We're considering using facial recognition to monitor attendees at public events, but attendees aren't being told it's happening and they won't be able to see their own data once it's stored.", ["transparency", "privacy"]),
    ("We're considering using facial recognition software to identify potential security threats at our government office building, but what if it misidentifies an innocent visitor as a threat?", ["non_maleficence", "fairness"]),
    ("Our AI-powered content suggestion engine is sharing user data with third-party advertisers without explicit opt-in, and we're concerned about the potential misuse of this information.", ["privacy"]),
    ("Our landlord client wants to implement an AI-powered tenant screening system, but we're worried about who will be liable for any disputes or damages.", ["accountability"]),
    ("Should we use a machine learning model to predict patient readmission rates and alert clinicians, or will it add unnecessary administrative burden?", []),
    ("Can we use AI-powered chatbots to handle customer inquiries on our e-commerce website and how will they be integrated with our existing CRM system?", []),
    ("The company behind a popular learning app for children is struggling to identify who is responsible for a feature that uses facial recognition to track pupil engagement.", ["accountability"]),
    ("A housing platform uses AI to score tenant applications, but it's been noticed that low-income applicants consistently score lower, even with identical credit histories.", ["fairness"]),
    ("Our credit risk model is now using machine learning to automatically approve or deny loan applications, but the loan officers are not explaining how the decision was made to the customers.", ["transparency"]),
    ("Our new AI model for underwriting life insurance policies is consistently rejecting applications from people of low socioeconomic status, even when they have good health records.", ["fairness"]),
    ("Our marketing team wants to use a chatbot that asks customers about their shopping preferences, but we're concerned that we're not explicitly telling them that their conversations will be recorded.", ["transparency", "privacy"]),
    ("Our CCTV system is detecting and flagging individuals who are loitering in areas near government buildings, but we're not sure if we're storing their facial data and images for long enough to be in compliance with data protection regulations.", ["privacy"]),
    ("What's the best way to integrate our new content recommendation AI with our existing CMS to improve user engagement?", []),
    ("Our delivery AI is scheduling drivers to take routes with more hazardous road conditions, increasing the risk of accidents. Can we reroute them to safer routes without compromising delivery times?", ["non_maleficence"]),
    ("Our AI-powered content moderation tool has flagged a user's post for removal, but the algorithm's logic is unclear and our team can't reproduce the decision.", ["accountability"]),
    ("Our social media platform is using AI to automatically remove 'offensive' content, but users have no idea how this decision is made or what triggers the removal, and we're getting complaints about inconsistent moderation.", ["transparency"]),
    ("Our claims processing system suddenly stopped paying claims for customers with certain medical conditions, citing 'inconsistent policy application' without explanation.", ["transparency", "non_maleficence"]),
    ("Our HR manager wants to use facial recognition to monitor employee attendance and track individual productivity, but we're concerned about potential misidentification and false positives.", ["privacy", "fairness"]),
    ("Our self-driving trucks are collecting GPS data from delivery routes and sharing it with a third-party traffic analytics firm, but I'm not sure if the customers know about it.", ["transparency", "privacy"]),
    ("The company's AI-driven interview evaluation system scores candidates based on speech patterns, penalizing those with accents from non-native English speaking countries.", ["fairness"]),
    ("Can we use AI-powered video analysis to quickly review hours of security footage for incidents, or is it too resource-intensive?", ["privacy"]),
    ("Our new learning app for kids uses facial recognition to track their emotions and adapt the content. Could the algorithm misinterpret a child's distress and make them more anxious?", ["non_maleficence"]),
    ("Our credit scoring model consistently rates low-income borrowers as high-risk, even when they've made timely payments, while affluent borrowers with late payments are flagged as low-risk. Should we adjust the model to account for income disparities?", ["fairness"]),
    ("Should we use facial recognition software to monitor visitors to government buildings and identify potential security threats, even though it requires collecting and storing biometric data?", ["privacy", "transparency"]),
    ("Our AI-powered recommendation engine has started suggesting products to customers based on their browsing history, but we're not sure how to explain why a particular product was suggested if a customer asks.", ["transparency", "accountability"]),
    ("Our chatbot is responding differently to customers from different regions, providing more helpful information to customers from affluent areas but not to those in low-income areas.", ["fairness"]),
    ("Our health app's symptom checker just diagnosed a patient with a rare condition. However, the patient is not told that the diagnosis was made by an AI algorithm and the clinicians involved are unclear on its decision-making process.", ["transparency", "accountability"]),
    ("Our AI-powered route optimization system consistently gives longer routes to drivers with non-standard accents, yet I've noticed no difference in traffic patterns or road conditions for these routes.", ["fairness"]),
    ("Can we use AI-powered chatbots to handle our customer service inquiries 24/7 without impacting our current staffing levels?", []),
    ("Our social media platform's automated content moderation system has been deleting posts from users who are grieving the loss of loved ones, and we're getting complaints from users who feel their free speech is being stifled.", ["non_maleficence"]),
    ("Our new 'SmartQuote' tool generates insurance quotes based on customer data, but we're not sure how to explain to customers why their quote is higher than their neighbor's when all the information seems the same.", ["transparency", "accountability"]),
    ("Our chatbot is recommending medication to customers based on their symptom input, but we're not disclosing that the recommendations are generated by an AI algorithm.", ["transparency", "non_maleficence"]),
    ("Our AI-powered property platform is using natural language processing to analyze tenant reviews, but we're not disclosing this to landlords or tenants - should we be upfront about it?", ["transparency"]),
    ("Our learning app for kids collects data on students' progress, but we're considering sharing it with a third-party company that offers tutoring services. Is it okay to share this data without parents' explicit consent?", ["privacy"]),
    ("Our autonomous delivery drones are making decisions about package routing without human oversight, but we can't figure out whose responsibility it is if they cause an accident.", ["accountability"]),
    ("Our new AI-powered performance monitoring system is flagging several employees for 'low productivity' despite no change in their work output - is this causing undue stress on these employees?", ["non_maleficence"]),
    ("If our health app uses patient data to predict disease risk, can we use that data to sell targeted treatments, even if it might lead to patients being overcharged or misdiagnosed?", ["non_maleficence"]),
    ("Our AI-powered loan approval system may inadvertently deny credit to low-income borrowers who may struggle to repay loans, leading to financial ruin.", ["non_maleficence"]),
    ("A customer complains about their chatbot conversation being recorded without their knowledge, and asks if they can request deletion of their voice recordings.", ["transparency", "privacy"]),
    ("We're considering using a credit scoring model that combines data from social media with traditional credit reports to assess loan applicants. Is this a good idea given the sensitive nature of the data involved?", ["privacy"]),
    ("Our new social media platform uses AI to suggest posts to users based on their interests, but we're struggling to determine how to provide clear explanations for these recommendations without revealing sensitive user data.", ["transparency", "privacy"]),
    ("Our recommendation engine is automatically assigning a lower price to women's products, but we can't explain why this is happening or why it's only affecting women. Can we adjust the algorithm?", ["transparency", "fairness"]),
    ("Our health app uses AI to track users' sleep patterns, but we don't inform them that their data is being used to sell targeted sleep aids to other companies.", ["transparency", "privacy"]),
    ("Our AI-powered recommendation engine is generating sales targets for our sales team, but we're struggling to understand why some targets are significantly higher than others. Is someone responsible for reviewing these targets before they're set?", ["accountability"]),
    ("Our chatbot's automated responses to medical queries have been causing patients to delay seeking in-person treatment for severe symptoms, potentially leading to worsening health outcomes.", ["non_maleficence"]),
    ("Our bank's AI system for evaluating loan applicants has been making decisions that are inconsistent with our usual lending standards, but we can't determine why or who to blame.", ["accountability"]),
    ("Our platform's AI-powered moderation tool is consistently flagging posts from users with Middle Eastern accents as spam, but similar posts from users with Western accents are rarely flagged.", ["fairness"]),
    ("Our new AI-powered loan application system is rejecting a high percentage of applications, but we can't explain why or provide any specific feedback to the applicants.", ["transparency", "non_maleficence"]),
    ("Our performance review system uses AI to analyze employee data, but we're not telling the team how it works or what metrics are used, and some managers are using the results to make decisions about employee raises.", ["transparency", "privacy"]),
    ("We're using location data from customers' smartphones to optimize delivery routes, but are we sharing that data with our insurance provider to calculate premiums?", ["privacy"]),
    ("We're considering using an AI-powered matching system for tenants and landlords, but the algorithm has been shown to favor long-term renters over families with young children. How can we ensure the system isn't biased against certain demographics?", ["transparency", "fairness"]),
    ("Our logistics team wants to use a new AI tool that prioritizes deliveries to high-value clients, but it's based on data from our website, which has an uneven user demographic and doesn't collect consent for data use.", ["fairness", "privacy"]),
    ("A social media platform's AI moderation system flags a post by a minor as extremist, leading to the child's online account being suspended without explanation or appeal.", ["non_maleficence"]),
    ("Our AI-powered chatbot is collecting sensitive health information from customers, but we're not sure who's liable if it gets leaked or used for targeted advertising without consent.", ["accountability", "privacy"]),
    ("Our new AI-powered tutoring app for low-income students has been flagged for underperforming students from certain ethnic backgrounds, while the same students from more affluent areas are excelling with the same material.", ["fairness", "non_maleficence"]),
    ("The AI-powered face recognition system being used in our city's surveillance network consistently flags more black residents for suspicious activity than white residents.", ["fairness"]),
    ("Our rental platform collects tenants' personal data, including browsing history and online behavior, to personalize our marketing and improve tenant matches. How should we balance data collection for better matches with tenants' right to keep their online activities private?", ["privacy"]),
    ("We're considering using AI to suggest personalized exercise routines, but how can we ensure that these recommendations don't inadvertently penalize users with mobility issues or disabilities?", ["fairness"]),
    ("The algorithm driving our mental health chatbot is consistently giving more severe diagnosis labels to women than men, despite identical symptom inputs and diagnostic protocols.", ["fairness"]),
    ("We're using facial recognition to analyze customer photos in our loyalty program, but I'm worried about how it will treat people with darker skin tones.", ["fairness"]),
    ("The AI system identifies high-risk customers based on their social media behavior and sends alerts to our loan officers, but what about the customers who don't use social media?", []),
    ("Our underwriting team is considering using facial recognition software to verify customers' identities, but it also captures and stores their biometric data.", ["fairness", "privacy"]),
    ("Should we use AI to automate credit scoring for our new loan product, and if so, what's the best way to integrate it with our existing software?", []),
    ("Our property platform's AI-driven rental pricing algorithm increases the rent for low-income families, citing market demand, but the algorithm doesn't consider the families' limited budget and reliance on government assistance.", ["fairness"]),
    ("The IT department is struggling to manage an AI-driven predictive policing tool, as it's unclear who should be trained on its outputs and how to assess their reliability in court.", ["accountability"]),
    ("How can we integrate AI into our interview process to get more diverse candidate pools?", []),
    ("If our AI system prioritizes customers who are more likely to pay their premiums on time, are we unfairly penalizing those who are experiencing financial hardship?", ["fairness"]),
    ("The AI system has taken to responding to customer inquiries with pre-written templates that don't take into account the specific situation, and nobody checks for accuracy.", ["accountability"]),
    ("Who is ultimately responsible for the AI-powered performance monitoring tool that flagged John for a meeting with HR after it misclassified his work as 'inconsistent'?", ["accountability", "non_maleficence"]),
    ("We're implementing an AI-powered chatbot to handle customer inquiries about claims, but who will be responsible if it gets a question wrong and the customer sues us?", ["accountability"]),
    ("Can we use AI to analyze employee performance data and provide actionable insights to our managers?", []),
    ("A performance management AI system flags a significantly higher number of warnings for employees from underrepresented groups compared to their peers.", ["fairness"]),
    ("How can we implement AI-driven chatbots to help parents communicate with teachers?", []),
    ("How can we ensure that our AI-powered tracking system doesn't interfere with existing GPS systems used by our drivers and other vehicles on the road?", []),
    ("How can we configure the AI-driven rent pricing model to adjust for local market fluctuations and seasonal changes?", []),
    ("Can we use AI to review patient medical histories and provide diagnoses without involving a doctor, and if so, who is liable if the diagnosis is wrong?", ["accountability", "privacy"]),
    ("Our learning app for kids uses AI to suggest personalized lesson plans, but we've noticed that African American students are consistently recommended for more advanced math problems than their white peers.", ["fairness"]),
    ("Our police department is implementing an AI-powered predictive policing system, but what if it leads to more stops and searches of minority communities and exacerbates existing tensions?", ["non_maleficence", "fairness"]),
    ("The real estate platform is considering using AI to generate automated responses to tenant inquiries, but what if the AI makes a critical mistake?", ["accountability", "non_maleficence"]),
    ("We want to implement AI-driven product recommendations on our retail website, but how will we balance the need for personalization with concerns about data storage and analytics?", []),
    ("A landlord is considering using AI to select applicants for a new property, but they're worried it might disproportionately exclude minority groups.", ["fairness"]),
    ("The AI system used to determine insurance claims payouts is flagging unusually high rates of claims from a particular geographic region, leading to suspicions of bias.", ["fairness"]),
    ("Can we use natural language processing to automatically categorize and summarize news articles on our platform?", []),
    ("If our AI system incorrectly assigns packages to homes, it could lead to theft or damage to property. Should we prioritize manual verification of addresses?", ["non_maleficence"]),
    ("The new scheduling AI suggests we shift some employees to overnight shifts without consulting them, citing 'efficiency gains', but we worry about their safety and well-being.", ["non_maleficence"]),
    ("How can we integrate an AI-driven chatbot with our existing customer service software to improve response times without disrupting our existing workflows?", []),
    ("The AI-powered grading system flags certain student behaviors as 'problematic', but what if it unfairly targets students with special needs?", ["non_maleficence", "fairness"]),
    ("Lenders are hesitant to approve loans to applicants with non-traditional credit histories, such as rent payments or utility bills, due to concerns over the reliability of these data sources.", ["fairness"]),
    ("The chatbot is more likely to resolve issues for younger customers than older customers, who often need more time and patience to resolve their problems.", ["fairness"]),
    ("I'm concerned that our autonomous delivery vans will stop at locations that are predominantly low-income neighborhoods, given their AI's preference for maximizing fuel efficiency.", ["fairness"]),
    ("How can we integrate AI-driven analytics with our existing CRM system to improve customer retention?", []),
    ("If our AI-powered route optimization system recommends a route that ends up causing congestion and traffic jams, who's accountable for the resulting gridlock?", ["accountability"]),
    ("We're considering using AI to optimize our scheduling algorithm to reduce labor costs, but what about the impact on employees who will now be working night shifts more frequently?", ["fairness", "non_maleficence"]),
    ("What if our AI system prioritizes patients with a certain type of insurance, potentially delaying treatment for those without?", ["non_maleficence"]),
    ("The bank's AI system is suggesting aggressive credit card debt collection tactics that may push customers to bankruptcy.", ["non_maleficence"]),
    ("We're using facial recognition to identify and target customers with certain demographics, but our ad targeting is based on physical appearance. Is this compliant with our diversity policies?", ["fairness"]),
    ("A customer service bot on our website is recommending customers pay for expedited shipping on packages that will still take days to arrive, leading to customer frustration and potential financial loss.", ["non_maleficence"]),
    ("We're considering using AI to analyze user engagement and personalize content, but I'm worried it will disproportionately favor users with high-income demographics.", ["fairness"]),
    ("Can we use AI to help underwriters identify high-risk clients more quickly, without retraining the entire team on a new system?", []),
    ("Our AI-driven rental pricing tool is generating suspiciously high prices for homes in low-income neighborhoods, but the algorithm's logic is opaque to us. How can we trust the results without knowing how the AI arrived at them?", ["transparency", "fairness"]),
    ("A news website's AI-powered content generator produces an article with misleading and unverified information about a local business, potentially damaging its reputation.", ["non_maleficence"]),
    ("The AI system we're using to identify students at risk of dropping out is recommending more frequent interventions for students with ADHD, but their parents are complaining about the added stress.", ["fairness", "non_maleficence"]),
    ("Our company's new predictive policing tool is disproportionately recommending more stops for young men from low-income neighborhoods.", ["fairness"]),
    ("Our AI-driven pricing algorithm is adjusting prices based on the location and time of day, but I'm concerned it's unfairly charging customers in low-income neighborhoods.", ["fairness"]),
    ("How can we use AI to analyze large datasets and identify trends in customer behavior, without overloading our IT infrastructure?", []),
    ("What's the cost and training time for implementing an AI-powered chatbot for initial candidate inquiries?", []),
    ("How will implementing AI-driven scheduling affect our team's workload and work-life balance?", []),
    ("A facial recognition system used to track employee attendance consistently fails to recognize employees with darker skin tones, leading to unexplained absences.", ["fairness"]),
    ("What's the best way to integrate AI-powered grading tools with our existing learning management system?", []),
    ("Can we use AI to automatically format and send out delivery notifications to customers, or does this require manual input and review for accuracy?", []),
    ("What's the estimated time and cost to implement the AI-based tenant screening system and retrain our staff on its use?", []),
    ("The AI-powered tutoring system claims to identify students who are struggling, but it only flags those from higher-income families, leading to uneven support.", ["fairness"]),
    ("The AI-driven recommendation system is suggesting job postings that are mostly open to local candidates, but I suspect it's not doing enough to promote diversity and inclusion.", ["fairness"]),
    ("How do we ensure that our AI-powered moderation tool for user-generated content can adapt to changing social media trends and policies?", []),
    ("What's the estimated cost of implementing AI-based chatbots to reduce our average response time to customers?", []),
    ("We need to integrate our AI-powered claims processing tool with our existing CRM software, what's the best approach?", []),
    ("What are the costs and feasibility of using AI for automated content generation for our marketing campaigns, and how will we ensure consistency with our brand voice?", []),
    ("A real estate agent claims their AI-powered property matching system is showing them more profitable listings, but it's only recommending properties in predominantly white neighborhoods.", ["fairness"]),
    ("Should we prioritize investing in AI-based predictive maintenance for our building's HVAC systems to reduce energy costs, or is it too experimental?", []),
    ("How much will it cost to train our staff on using the new AI-powered chatbot for customer support inquiries?", []),
    ("Can we use computer vision to automatically detect and analyze patient scans, and would it save us time and resources compared to manual review?", []),
    # --- end of LLM-augmented examples ---

    # =========================================================================
    # Εδώ πέφτουν τα prompts που έχουν δώσει οι χρήστες στην εφαρμογή και
    # προστίθενται στο training_data από το πρόγραμμα στο αρχείο review_log.
    # ΜΗΝ αλλάξεις/μετακινήσεις το σχόλιο "neutral / low-signal examples"
    # παρακάτω — το review_log.py ψάχνει ακριβώς αυτή τη γραμμή για να
    # ξέρει πού να εισάγει κάθε νέα εγκεκριμένη καταχώριση, ακριβώς πάνω
    # από αυτήν, ώστε όλες οι μελλοντικές προσθήκες να μαζεύονται εδώ.
    # =========================================================================
    ("bank's AI fraud model blocks transactions automatically. It prevents substantial financial losses, but customers from certain regions are flagged far more often. How should the bank balance security, fairness, explainability, and customer harm?", ["accountability", "transparency"]),
    ("Μια τράπεζα μπορεί να χρησιμοποιήσει ένα λιγότερο ακριβές αλλά πλήρως εξηγήσιμο μοντέλο ή ένα πολύ πιο ακριβές black-box neural network για έγκριση δανείων. Ποιο είναι ηθικά προτιμότερο και γιατί;", ["transparency", "accountability"]),
    ("An AI medical assistant gives a recommendation with 95% confidence, but the doctor disagrees. If the doctor follows the AI and the patient is harmed, who should be accountable? What if the doctor ignores the AI and the patient is harmed?", ["accountability"]),
    ("An insurance AI uses lifestyle data collected from smartphones and wearables. Customers technically consented, but refusing consent leads to significantly higher premiums. Is that consent genuinely voluntary?", ["privacy", "transparency"]),
    ("A company wants to use customer support transcripts to fine-tune its own LLM. Customers agreed to general data processing terms years ago but were never specifically told their conversations could train an AI system. Is the old consent ethically sufficient?", ["transparency", "privacy", "accountability"]),
    ("A company discovers that making its chatbot sound extremely confident increases customer satisfaction, even though the chatbot sometimes hallucinates. Should the company prioritise user confidence or calibrated uncertainty?", ["non_maleficence", "transparency"]),
    ("A hospital discovers that its diagnostic AI performs better overall than doctors but significantly worse for a small demographic subgroup. Should it still be deployed while improvements are being made?", ["fairness", "non_maleficence"]),
    ("A public-sector agency wants to use facial recognition because it significantly improves security. However, the system has higher false-positive rates for some demographic groups. Under what conditions, if any, would deployment be ethically defensible?", ["fairness", "privacy", "non_maleficence"]),
    ("We use Gemini to search for music composers and writers of music for claiming copyright claims on Youtube", ["privacy", "fairness"]),
    ("We use Gemini to confirm if a song is in the public domain for copyright claiming on YT, also check for the composers of the song via the same AI tool . Is there any regulation conflicts with that?", ["privacy", "accountability"]),



    # =========================================================================
    # ΔΙΟΡΘΩΤΙΚΟ BATCH PRIVACY (βλ. συζήτηση) — το προηγούμενο batch privacy
    # ήταν όλο γύρω από "διατήρηση δεδομένων" και προκάλεσε υπερπροσαρμογή
    # σε αυτό το στενό θέμα (CV F1 0.82, held-out F1 0.46). Αυτό καλύπτει
    # διαφορετικά υποθέματα (συναίνεση, τρίτα μέρη, βιομετρικά, πρόσβαση),
    # και προσθέτει 2 "δύσκολα αρνητικά" στα μοτίβα που παραπλάνησαν το
    # μοντέλο (χρονοζώνη/προγραμματισμός, υποστηρικτικό μήνυμα).
    # =========================================================================
    ("Our app collects a user's full contact list during signup, even though the feature only needs their own phone number.", ["privacy"]),
    ("Should we share customer purchase histories with a third-party AI marketing partner without asking customers first?", ["privacy"]),
    ("Our AI fitness tracker uploads raw GPS routes to a public leaderboard by default, revealing where users live and work.", ["privacy"]),
    ("A manager can look up any employee's AI-generated performance summary at any time, even for teams they don't manage.", ["privacy"]),
    ("Should we ask visitors for their fingerprint to enter the building, even though a simple badge would work just as well?", ["privacy"]),
    ("Our AI app quietly accesses users' photo libraries to 'improve recommendations', a permission most people don't realize they granted.", ["privacy"]),
    ("Is it okay for our AI analytics tool to combine data from three different services to build a single profile of each customer?", ["privacy"]),
    ("Θέλουμε να μοιραστούμε τα δεδομένα υγείας των πελατών μας με έναν συνεργάτη ΑΙ χωρίς να τους ζητήσουμε ξεχωριστή άδεια.", ["privacy"]),
    ("Ένας διαχειριστής μπορεί να δει τα προσωπικά μηνύματα οποιουδήποτε υπαλλήλου στο εσωτερικό chat, χωρίς κανέναν περιορισμό πρόσβασης.", ["privacy"]),
    ("Πρέπει να ζητάμε το δακτυλικό αποτύπωμα των επισκεπτών για είσοδο στο κτίριο, ενώ μια απλή κάρτα θα αρκούσε εξίσου καλά;", ["privacy"]),
    ("Should we let employees pick their own start times, as long as they complete their 8 hours each day?", []),
    ("Our AI wellness app sends supportive, non-judgmental messages when a user reports feeling stressed.", ["non_maleficence"]),

    ("Our company uses an ai assisted llm chatbot to track our deliveries. Although most of the times the clients put their personal data (like adresses names etc) to explain the problem and all of this data go to the anthropic api that we use but the clients dont know about it . How should we proceed?", ["privacy", "transparency"]),


    # --- neutral / low-signal examples (ουδέτερα παραδείγματα, ώστε το μοντέλο να μην ενεργοποιείται χωρίς λόγο) ---
    ("What's the best way to format a quarterly sales report?", []),
    ("Can you help me write a friendly onboarding email for new employees?", []),
    ("We are choosing between two project management tools for the engineering team.", []),
    ("How should we structure the agenda for next week's all-hands meeting?", []),
    ("What are some good icebreaker questions for a new team retreat?", []),

    # --- ελληνικά παραδείγματα (διαφάνεια) ---
    ("Το chatbot μας απαντά στους πελάτες χωρίς ποτέ να τους λέει ότι είναι ΑΙ, είναι πρόβλημα αυτό;", ["transparency"]),
    ("Πρέπει να ενημερώσουμε τους υποψηφίους ότι ένα εργαλείο ΑΙ αξιολόγησε αυτόματα τα βιογραφικά τους;", ["transparency", "fairness"]),
    ("Ο διευθυντής θέλει να μάθει γιατί το σύστημα ΑΙ πρότεινε την απόρριψη ενός δανείου.", ["transparency", "accountability"]),
    ("Οι πελάτες μπερδεύονται με το πώς το εργαλείο τιμολόγησης ΑΙ υπολογίζει την προσφορά τους.", ["transparency"]),
    ("Πρέπει να πούμε στους εργαζομένους ότι η αξιολόγηση απόδοσής τους γράφτηκε από ΑΙ;", ["transparency", "accountability"]),

    # --- ελληνικά παραδείγματα (δικαιοσύνη) ---
    ("Το εργαλείο ΑΙ για προσλήψεις φαίνεται να απορρίπτει περισσότερο τους μεγαλύτερους σε ηλικία υποψηφίους.", ["fairness"]),
    ("Το μοντέλο πιστοληπτικής αξιολόγησης δίνει χαμηλότερα όρια σε αιτούντες από συγκεκριμένους ταχυδρομικούς κώδικες.", ["fairness", "privacy"]),
    ("Πρέπει να ελέγξουμε τον αλγόριθμο συστάσεων για μεροληψία απέναντι σε συγκεκριμένη δημογραφική ομάδα;", ["fairness"]),
    ("Ένας υπάλληλος παρατήρησε ότι η ΑΙ κατατάσσει χαμηλότερα υποψηφίους με ξενόφωνα ονόματα.", ["fairness", "transparency"]),
    ("Ανησυχούμε ότι το σύστημα αναγνώρισης προσώπου είναι λιγότερο ακριβές για σκουρόχρωμο δέρμα.", ["fairness", "non_maleficence"]),

    # --- ελληνικά παραδείγματα (αποφυγή βλάβης & ασφάλεια) ---
    ("Θέλουμε να βάλουμε κάμερες παρακολούθησης και αναγνώριση συναισθήματος στο γραφείο για να δούμε αν οι υπάλληλοι είναι ικανοποιημένοι.", ["non_maleficence", "privacy"]),
    ("Το εργαλείο ΑΙ για ιατρική διαλογή δίνει μερικές φορές επικίνδυνες συμβουλές για σπάνια συμπτώματα.", ["non_maleficence"]),
    ("Υπάλληλοι επικολλούν εμπιστευτικά έγγραφα στρατηγικής πελατών σε δημόσιο chatbot.", ["privacy", "non_maleficence"]),
    ("Θα μπορούσε το εργαλείο παραγωγής περιεχομένου να παράγει επιβλαβές υλικό αν γίνει κατάχρηση από το προσωπικό;", ["non_maleficence"]),
    ("Ανησυχούμε ότι το bot εξυπηρέτησης πελατών κλιμακώνει τους θυμωμένους πελάτες αντί να τους ηρεμεί.", ["non_maleficence"]),

    # --- ελληνικά παραδείγματα (λογοδοσία) ---
    ("Ποιος είναι υπεύθυνος όταν η σύσταση της ΑΙ οδηγεί σε κακή επιχειρηματική απόφαση;", ["accountability"]),
    ("Δεν υπάρχει κανείς να επιθεωρεί τους ισχυρισμούς μάρκετινγκ που παράγει η ΑΙ πριν δημοσιευτούν.", ["accountability", "non_maleficence"]),
    ("Δεν υπάρχει αρχείο για το ποιος ενέκρινε τη χρήση αυτού του παρόχου ΑΙ για την αξιολόγηση προσωπικού.", ["accountability"]),
    ("Πρέπει να αποφασίσουμε ποιος εγκρίνει πριν τεθεί σε λειτουργία ένα νέο χαρακτηριστικό ΑΙ.", ["accountability"]),
    ("Χρειαζόμαστε ένα ίχνος ελέγχου για κάθε απόφαση με τη βοήθεια ΑΙ σχετικά με επιστροφές χρημάτων πελατών.", ["accountability", "transparency"]),

    # --- ελληνικά παραδείγματα (ιδιωτικότητα) ---
    ("Μπορούμε να ανεβάσουμε τη βάση δεδομένων πελατών μας σε δημόσιο εργαλείο ΑΙ για να δημιουργήσουμε τμήματα μάρκετινγκ;", ["privacy"]),
    ("Ένας υπάλληλος επικόλλησε τον ιατρικό φάκελο ενός ασθενή σε chatbot για να τον συνοψίσει.", ["privacy", "non_maleficence"]),
    ("Θέλουμε να αναλύσουμε τα email των εργαζομένων με ΑΙ για να εντοπίσουμε μοτίβα παραγωγικότητας.", ["privacy", "fairness"]),
    ("Είναι νόμιμο να τροφοδοτήσουμε οικονομικά αρχεία πελατών σε εργαλείο ανάλυσης ΑΙ τρίτου μέρους;", ["privacy", "accountability"]),
    ("Δεν είμαστε σίγουροι αν ο ΓΚΠΔ ισχύει για τα αρχεία καταγραφής του chatbot που αποθηκεύουμε.", ["privacy"]),

    # --- ελληνικά ουδέτερα παραδείγματα ---
    ("Ποιος είναι ο καλύτερος τρόπος να μορφοποιήσω μια τριμηνιαία αναφορά πωλήσεων;", []),
    ("Μπορείς να με βοηθήσεις να γράψω ένα φιλικό email καλωσορίσματος για νέους υπαλλήλους;", []),
    ("Ποιες είναι μερικές καλές ερωτήσεις γνωριμίας για μια εκδρομή ομάδας;", []),
]


# -----------------------------------------------------------------------
# Μετατροπή των παραδειγμάτων σε παράλληλες λίστες κειμένων και
# διανυσμάτων ετικετών (multi-hot), κατάλληλες για εκπαίδευση
# -----------------------------------------------------------------------
def get_texts_and_labels():
    """Return parallel lists of texts and multi-hot label vectors."""
    texts = []
    label_vectors = []
    for text, labels in TRAINING_EXAMPLES:
        texts.append(text)
        vec = [1 if cat in labels else 0 for cat in CATEGORIES]
        label_vectors.append(vec)
    return texts, label_vectors
