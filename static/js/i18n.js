// -----------------------------------------------------------------------
// Λεξικό μεταφράσεων: όλο το στατικό κείμενο του περιβάλλοντος χρήστη,
// σε Αγγλικά και Ελληνικά
// -----------------------------------------------------------------------
const TRANSLATIONS = {
  // --- Αγγλικές μεταφράσεις ---
  en: {
    pageTitle: "AI-thical — AI ethics & compliance advisory",
    wordmarkTitle: "AI-thical",
    wordmarkSub: "AI ethics & compliance advisory",
    mastheadNote: "Prototype research tool · not legal advice",
    h1: "Describe the situation",
    lede: "Tell us what your team is trying to do with AI, and where the uncertainty lies. Add the country and business function so the advisory can narrow its sources.",
    dilemmaLabel: "Business dilemma",
    dilemmaPlaceholder: "e.g. We want to use an AI tool to screen job applicants, but we're not sure whether we need to tell candidates, or check the tool for bias.",
    countryLabel: "Country",
    regionLabel: "Region",
    stateLabel: "State",
    sectorLabel: "Business function",
    narrativeCheckboxLabel: "Also write a natural-language summary using an AI model.",
    narrativeLocalNote: "Running locally via Ollama — nothing leaves this machine.",
    narrativeExternalNote: "This sends your dilemma text to an external API. Off by default.",
    narrativeNoneNote: "The AI-drafted summary isn't available at the moment.",
    liveSourcesCheckboxLabel: "Also search the web for recent, unverified sources.",
    liveSourcesCheckboxNote: "Adds recent web pages to your report. They are not reviewed, so the option is off by default.",
    sectionLive: "Recent sources (live, unverified)",
    liveSourcesNote: "These come from a live web search and are not reviewed — verify before relying on them.",
    liveSourcesEmpty: "No live results found for this combination.",
    submitBtn: "Get advisory report",
    submitBtnLoading: "Analyzing…",
    loadingText: "Analyzing your dilemma…",
    loadingTextWithSearch: "Analyzing your dilemma and searching the web — this can take a little longer…",
    historyBtnLabel: "History",
    historyEmpty: "No analyses yet this session.",
    copyBtnLabel: "Copy",
    copiedLabel: "Copied ✓",
    copyFailed: "Couldn't copy — try selecting the text manually.",
    printBtnLabel: "Print / Save as PDF",
    reportEmpty1: "Your advisory report will appear here once you submit a dilemma.",
    reportEmpty2: "It will name the ethical dimensions detected in your description, place it on the EU AI Act risk pyramid, then list matching principles, regulatory considerations and recommended actions.",
    narrativeLabel: "AI-drafted summary",
    narrativeLabelNote: "generated from the sources and matched literature below — verify against them",
    llmAssistNote: "Assisted by a local LLM classification pass, since the local model wasn't confident from keywords/patterns alone.",
    addedByRisk: "from the risk level",
    addedByRiskTitle: "The classifier did not detect this issue, but the AI Act use case recognised in your description raises it.",
    section1: "Ethical dimensions detected",
    section2: "Relevant principles",
    sectionRisk: "EU AI Act risk level",
    riskMatchesTitle: "Why this level",
    riskObligationsTitle: "Key obligations at this level",
    riskHere: "Your case",
    riskTiers: {
      unacceptable: ["Unacceptable risk", "Prohibited practices — Art. 5"],
      high: ["High risk", "Strict requirements — Art. 6, Annex I & III"],
      limited: ["Limited risk", "Transparency duties — Art. 50"],
      minimal: ["Minimal risk", "No specific obligations"],
    },
    sectionBiblio: "Further literature (semantic match)",
    biblioNote: "Matched by meaning across a wider bibliography, not filtered by country or sector like the principles above.",
    section3: "Regulatory considerations",
    section4: "Recommended next steps",
    section5: "Limits of this report",
    noDimension: "No strong ethical signal was detected in this description. Try adding more detail.",
    noPrinciple: "No specific principle matched this combination yet.",
    noRegulation: "No jurisdiction-specific regulation matched this combination yet.",
    noAction: "No sector-specific action matched this combination yet.",
    sourceLabel: "Source",
    footerText: 'Ethical prototype — built for the MSc thesis "Aithical: The AI Consulting LLM for Ethical and Legal issues about AI".',
    formErrorEmpty: "Please describe a business dilemma before submitting.",
    formErrorUnreachable: "Something went wrong and we couldn't generate your report. Please try again in a few minutes.",
    formErrorGeneric: "Something went wrong with your request. Please try again.",
    formErrorRateLimit: "You've sent several requests in a short time. Please wait a minute and try again.",
    infoBtnLabel: "About this section",
    themeToDark: "Switch to dark mode",
    themeToLight: "Switch to light mode",
    tips: {
      dimensions: "The ethical issues found in your description, out of five: transparency, fairness, harm prevention, accountability and privacy. The percentage shows how confident the classifier is, not how serious the issue is. Issues marked “from the risk level” were added because the AI Act use case recognised in your description raises them.",
      risk: "Where your use case would likely sit on the four risk levels of the EU AI Act (Regulation 2024/1689), based on the prohibited practices of Art. 5, the high-risk areas of Annex III and Annex I, and the transparency duties of Art. 50. The marker sits higher in its band the closer the case is to the level above. It is an automated indication, not a legal classification.",
      principles: "Established AI ethics principles that apply to the issues found, each with its source. They explain why the issue matters, beyond what the law requires.",
      biblio: "Academic papers closest in meaning to your description, found by semantic search. Unlike the principles above, they are not filtered by country or sector.",
      regulations: "Laws and regulations that may apply in the country or state you selected, with the specific article cited. They show you which rules to check; they are not legal advice.",
      actions: "Practical steps for your business function that address the issues found. Start with these before you deploy or change the AI system.",
      live: "Recent web pages found by a live search at the moment of your request. They have not been reviewed, so check each one before relying on it.",
      limits: "What this report can and cannot tell you. It offers general guidance from curated sources and does not replace advice from a lawyer or an ethics expert.",
    },
    categoryLabels: {
      transparency: "Transparency",
      fairness: "Fairness",
      non_maleficence: "Non-maleficence",
      accountability: "Accountability",
      privacy: "Privacy",
    },
  },
  // --- Ελληνικές μεταφράσεις ---
  el: {
    pageTitle: "AI-thical — Ηθική και συμμόρφωση στην τεχνητή νοημοσύνη",
    wordmarkTitle: "AI-thical",
    wordmarkSub: "Οδηγός ηθικής και συμμόρφωσης για την ΤΝ",
    mastheadNote: "Ερευνητικό πρωτότυπο · δεν αποτελεί νομική συμβουλή",
    h1: "Περιγράψτε την περίπτωσή σας",
    lede: "Πείτε μας τι θέλει να κάνει η ομάδα σας με την τεχνητή νοημοσύνη και τι σας προβληματίζει. Επιλέξτε χώρα και τομέα δραστηριότητας, ώστε οι πηγές να αφορούν τη δική σας περίπτωση.",
    dilemmaLabel: "Η περίπτωσή σας",
    dilemmaPlaceholder: "π.χ. Θέλουμε ένα εργαλείο ΤΝ που θα ξεχωρίζει βιογραφικά, αλλά δεν ξέρουμε αν πρέπει να ενημερώσουμε τους υποψηφίους ή να ελέγξουμε το εργαλείο για μεροληψία.",
    countryLabel: "Χώρα",
    regionLabel: "Περιοχή",
    stateLabel: "Πολιτεία",
    sectorLabel: "Τομέας",
    narrativeCheckboxLabel: "Θέλω και σύντομη περίληψη σε απλά λόγια, γραμμένη από μοντέλο ΤΝ.",
    narrativeLocalNote: "Τρέχει τοπικά μέσω Ollama· το κείμενό σας δεν φεύγει από τον υπολογιστή.",
    narrativeExternalNote: "Για την περίληψη, το κείμενό σας στέλνεται σε εξωτερική υπηρεσία. Γι' αυτό η επιλογή είναι κλειστή από προεπιλογή.",
    narrativeNoneNote: "Η περίληψη δεν είναι διαθέσιμη αυτή τη στιγμή.",
    liveSourcesCheckboxLabel: "Θέλω και πρόσφατες πηγές από το διαδίκτυο (χωρίς έλεγχο).",
    liveSourcesCheckboxNote: "Προσθέτει στην αναφορά πρόσφατα άρθρα από τον ιστό. Επειδή δεν έχουν ελεγχθεί, η επιλογή είναι κλειστή από προεπιλογή.",
    sectionLive: "Πρόσφατες πηγές από το διαδίκτυο (χωρίς έλεγχο)",
    liveSourcesNote: "Βρέθηκαν με αναζήτηση τη στιγμή του αιτήματος και δεν έχουν ελεγχθεί. Επιβεβαιώστε τες πριν βασιστείτε σε αυτές.",
    liveSourcesEmpty: "Δεν βρέθηκαν πρόσφατες πηγές για αυτόν τον συνδυασμό.",
    submitBtn: "Δημιουργία αναφοράς",
    submitBtnLoading: "Γίνεται ανάλυση…",
    loadingText: "Αναλύουμε την περίπτωσή σας…",
    loadingTextWithSearch: "Αναλύουμε την περίπτωσή σας και ψάχνουμε στο διαδίκτυο· ίσως χρειαστεί λίγο περισσότερος χρόνος…",
    historyBtnLabel: "Ιστορικό",
    historyEmpty: "Δεν έχετε κάνει ακόμη καμία ανάλυση.",
    copyBtnLabel: "Αντιγραφή",
    copiedLabel: "Αντιγράφηκε ✓",
    copyFailed: "Η αντιγραφή δεν έγινε. Επιλέξτε το κείμενο και αντιγράψτε το χειροκίνητα.",
    printBtnLabel: "Εκτύπωση / PDF",
    reportEmpty1: "Η αναφορά θα εμφανιστεί εδώ μόλις περιγράψετε την περίπτωσή σας.",
    reportEmpty2: "Θα δείτε ποια ηθικά ζητήματα προκύπτουν, σε ποια βαθμίδα κινδύνου του Κανονισμού για την ΤΝ ανήκει πιθανότατα η περίπτωση, και ποιες αρχές, νόμοι και πρακτικά βήματα σας αφορούν.",
    narrativeLabel: "Περίληψη από ΤΝ",
    narrativeLabelNote: "βασίζεται στις πηγές που ακολουθούν — ελέγξτε τη με βάση αυτές",
    llmAssistNote: "Επειδή ο ταξινομητής δεν ήταν βέβαιος, ζητήθηκε και η γνώμη ενός γλωσσικού μοντέλου.",
    addedByRisk: "από τη βαθμίδα κινδύνου",
    addedByRiskTitle: "Ο ταξινομητής δεν εντόπισε αυτό το ζήτημα, αλλά το θέτει η περίπτωση του Κανονισμού για την ΤΝ που αναγνωρίστηκε.",
    section1: "Ηθικά ζητήματα που εντοπίστηκαν",
    section2: "Σχετικές ηθικές αρχές",
    sectionRisk: "Βαθμίδα κινδύνου κατά τον Κανονισμό για την ΤΝ",
    riskMatchesTitle: "Γιατί αυτή η βαθμίδα",
    riskObligationsTitle: "Τι προβλέπεται για αυτή τη βαθμίδα",
    riskHere: "Η περίπτωσή σας",
    riskTiers: {
      unacceptable: ["Μη αποδεκτός κίνδυνος", "Απαγορεύεται — Άρθρο 5"],
      high: ["Υψηλός κίνδυνος", "Αυστηρές απαιτήσεις — Άρθρο 6, Παραρτήματα I και III"],
      limited: ["Περιορισμένος κίνδυνος", "Υποχρέωση ενημέρωσης — Άρθρο 50"],
      minimal: ["Ελάχιστος κίνδυνος", "Καμία ειδική υποχρέωση"],
    },
    sectionBiblio: "Σχετική βιβλιογραφία",
    biblioNote: "Επιλέγεται με βάση το νόημα της περιγραφής σας από ευρύτερη βιβλιογραφία, χωρίς φίλτρο χώρας ή τομέα.",
    section3: "Νομοθεσία που πρέπει να εξετάσετε",
    section4: "Προτεινόμενα επόμενα βήματα",
    section5: "Τι δεν καλύπτει αυτή η αναφορά",
    noDimension: "Η περιγραφή δεν δείχνει κάποιο σαφές ηθικό ζήτημα. Δοκιμάστε να δώσετε περισσότερες λεπτομέρειες.",
    noPrinciple: "Δεν βρέθηκε σχετική αρχή για αυτόν τον συνδυασμό.",
    noRegulation: "Δεν βρέθηκε νομοθεσία ειδικά για αυτή τη χώρα και αυτόν τον συνδυασμό.",
    noAction: "Δεν βρέθηκαν βήματα ειδικά για αυτόν τον τομέα.",
    sourceLabel: "Πηγή",
    footerText: 'Ερευνητικό πρωτότυπο της μεταπτυχιακής διπλωματικής εργασίας «Aithical: The AI Consulting LLM for Ethical and Legal issues about AI».',
    formErrorEmpty: "Περιγράψτε πρώτα την περίπτωσή σας.",
    formErrorUnreachable: "Κάτι πήγε στραβά και η αναφορά δεν δημιουργήθηκε. Δοκιμάστε ξανά σε λίγα λεπτά.",
    formErrorGeneric: "Κάτι πήγε στραβά. Δοκιμάστε ξανά.",
    formErrorRateLimit: "Στείλατε πολλά αιτήματα μέσα σε λίγο χρόνο. Περιμένετε ένα λεπτό και δοκιμάστε ξανά.",
    infoBtnLabel: "Τι δείχνει αυτή η ενότητα",
    themeToDark: "Σκούρο θέμα",
    themeToLight: "Φωτεινό θέμα",
    tips: {
      dimensions: "Ποια από τα πέντε βασικά ηθικά ζητήματα (διαφάνεια, δικαιοσύνη, αποφυγή βλάβης, λογοδοσία, ιδιωτικότητα) αφορούν την περίπτωσή σας. Το ποσοστό δείχνει πόσο βέβαιος είναι ο ταξινομητής, όχι πόσο σοβαρό είναι το ζήτημα. Όσα έχουν την ένδειξη «από τη βαθμίδα κινδύνου» προστέθηκαν επειδή τα θέτει η περίπτωση του Κανονισμού που αναγνωρίστηκε.",
      risk: "Πού τοποθετείται πιθανότατα η περίπτωσή σας στις τέσσερις βαθμίδες κινδύνου του Κανονισμού (ΕΕ) 2024/1689 για την ΤΝ: απαγορευμένες πρακτικές (Άρθρο 5), υψηλός κίνδυνος (Παραρτήματα III και I), υποχρέωση ενημέρωσης (Άρθρο 50). Όσο πιο ψηλά στη ζώνη του βρίσκεται ο δείκτης, τόσο πιο κοντά είναι η περίπτωση στην επόμενη βαθμίδα. Είναι αυτόματη εκτίμηση, όχι νομική κατάταξη.",
      principles: "Βασικές αρχές ηθικής της ΤΝ που αφορούν τα ζητήματα που εντοπίστηκαν, με την πηγή τους. Εξηγούν γιατί το ζήτημα έχει σημασία, πέρα από όσα ορίζει ο νόμος.",
      biblio: "Επιστημονικά άρθρα με περιεχόμενο κοντά στην περιγραφή σας, που βρέθηκαν με σημασιολογική αναζήτηση. Αντίθετα με τις αρχές, δεν φιλτράρονται ανά χώρα ή τομέα.",
      regulations: "Νόμοι και κανονισμοί που μπορεί να ισχύουν στη χώρα ή την πολιτεία που επιλέξατε, με το συγκεκριμένο άρθρο. Σας δείχνουν τι πρέπει να ελέγξετε· δεν αποτελούν νομική συμβουλή.",
      actions: "Πρακτικά βήματα για τον τομέα σας, που απαντούν στα ζητήματα που εντοπίστηκαν. Καλό είναι να τα δείτε πριν βάλετε σε λειτουργία ή αλλάξετε το σύστημα.",
      live: "Πρόσφατες σελίδες που βρέθηκαν με αναζήτηση τη στιγμή του αιτήματος. Δεν έχουν ελεγχθεί, οπότε επιβεβαιώστε καθεμία πριν βασιστείτε σε αυτήν.",
      limits: "Τι μπορεί και τι δεν μπορεί να σας πει η αναφορά. Δίνει γενική καθοδήγηση από επιλεγμένες πηγές και δεν αντικαθιστά τη γνώμη δικηγόρου ή ειδικού σε θέματα ηθικής.",
    },
    categoryLabels: {
      transparency: "Διαφάνεια",
      fairness: "Δικαιοσύνη",
      non_maleficence: "Αποφυγή βλάβης",
      accountability: "Λογοδοσία",
      privacy: "Ιδιωτικότητα",
    },
  },
};

// -----------------------------------------------------------------------
// Αποθήκευση και ανάκτηση της επιλεγμένης γλώσσας στο localStorage,
// ώστε να διατηρείται μεταξύ επισκέψεων
// -----------------------------------------------------------------------
const LANG_STORAGE_KEY = "ethical_lang";

function getStoredLang() {
  const stored = localStorage.getItem(LANG_STORAGE_KEY);
  return stored === "el" ? "el" : "en";
}

function setStoredLang(lang) {
  localStorage.setItem(LANG_STORAGE_KEY, lang);
}

let currentLang = getStoredLang();

// -----------------------------------------------------------------------
// Βοηθητική συνάρτηση μετάφρασης: επιστρέφει την τιμή για το δοσμένο
// κλειδί στην τρέχουσα γλώσσα, με εφεδρική επιλογή τα Αγγλικά
// -----------------------------------------------------------------------
function t(key) {
  return TRANSLATIONS[currentLang][key] ?? TRANSLATIONS.en[key] ?? key;
}

// -----------------------------------------------------------------------
// Ανάγνωση των ρυθμίσεων backend (πάροχος LLM κ.λπ.) που είναι
// ενσωματωμένες στη σελίδα από τον Flask κατά την απόδοσή της
// -----------------------------------------------------------------------
function getAppConfig() {
  try {
    return JSON.parse(document.getElementById("app-config").textContent);
  } catch (e) {
    return { usingFallbackClassifier: true, llmProviderIsLocal: false, llmProviderConfigured: false };
  }
}

// -----------------------------------------------------------------------
// Επιλογή του κατάλληλου επεξηγηματικού κειμένου για το πλαίσιο ελέγχου
// της σύνοψης LLM, ανάλογα με το αν ο πάροχος είναι τοπικός, εξωτερικός
// ή μη ρυθμισμένος
// -----------------------------------------------------------------------
function narrativeNoteText(config) {
  if (config.llmProviderIsLocal) return t("narrativeLocalNote");
  if (config.llmProviderConfigured) return t("narrativeExternalNote");
  return t("narrativeNoneNote");
}

// -----------------------------------------------------------------------
// Κύρια συνάρτηση: εφαρμόζει την επιλεγμένη γλώσσα σε ολόκληρη τη σελίδα
// -----------------------------------------------------------------------
// options.silent: αλλαγή γλώσσας χωρίς επανυποβολή της αναφοράς (π.χ. όταν
// η σελίδα απλώς συγχρονίζεται με τη γλώσσα μιας απάντησης ή του ιστορικού)
function applyLanguage(lang, options = {}) {
  const previousLang = currentLang;
  currentLang = lang === "el" ? "el" : "en";
  setStoredLang(currentLang);

  // Ενημέρωση κεφαλίδας, λογοτύπου και βασικών στοιχείων της φόρμας
  document.getElementById("html-root").setAttribute("lang", currentLang);
  document.getElementById("page-title").textContent = t("pageTitle");
  document.getElementById("wordmark-title").textContent = t("wordmarkTitle");
  document.getElementById("wordmark-sub").textContent = t("wordmarkSub");
  document.getElementById("masthead-note").textContent = t("mastheadNote");
  document.getElementById("panel-h1").textContent = t("h1");
  document.getElementById("panel-lede").textContent = t("lede");
  document.getElementById("dilemma-label").textContent = t("dilemmaLabel");
  document.getElementById("dilemma").setAttribute("placeholder", t("dilemmaPlaceholder"));
  document.getElementById("region-label").textContent = t("regionLabel");
  document.getElementById("eu-country-label").textContent = t("countryLabel");
  document.getElementById("us-state-label").textContent = t("stateLabel");
  document.getElementById("sector-label").textContent = t("sectorLabel");
  document.getElementById("narrative-checkbox-label").textContent = t("narrativeCheckboxLabel");

  // Επεξηγηματικά κείμενα που εξαρτώνται και από τις ρυθμίσεις backend
  const config = getAppConfig();
  document.getElementById("narrative-checkbox-note").textContent = narrativeNoteText(config);
  document.getElementById("live-sources-checkbox-label").textContent = t("liveSourcesCheckboxLabel");
  document.getElementById("live-sources-checkbox-note").textContent = t("liveSourcesCheckboxNote");
  document.getElementById("section-live-title").textContent = t("sectionLive");
  document.getElementById("live-sources-note").textContent = t("liveSourcesNote");

  // Ενότητες της αναφοράς και υποσέλιδο
  document.getElementById("report-empty-1").textContent = t("reportEmpty1");
  document.getElementById("report-empty-2").textContent = t("reportEmpty2");
  document.getElementById("narrative-label-text").textContent = t("narrativeLabel");
  document.getElementById("narrative-label-note").textContent = t("narrativeLabelNote");
  document.getElementById("llm-assist-note").textContent = t("llmAssistNote");
  document.getElementById("section-1-title").textContent = t("section1");
  document.getElementById("section-2-title").textContent = t("section2");
  document.getElementById("section-risk-title").textContent = t("sectionRisk");
  document.getElementById("risk-matches-title").textContent = t("riskMatchesTitle");
  document.getElementById("risk-obligations-title").textContent = t("riskObligationsTitle");
  document.querySelectorAll("#risk-legend li").forEach((li) => {
    const [name, desc] = t("riskTiers")[li.dataset.tier];
    li.querySelector(".risk-legend-name").textContent = name;
    li.querySelector(".risk-legend-desc").textContent = desc;
    li.querySelector(".risk-here").textContent = t("riskHere");
  });
  document.getElementById("section-biblio-title").textContent = t("sectionBiblio");
  document.getElementById("biblio-note").textContent = t("biblioNote");
  document.getElementById("section-3-title").textContent = t("section3");
  document.getElementById("section-4-title").textContent = t("section4");
  document.getElementById("section-5-title").textContent = t("section5");
  document.getElementById("site-footer-text").textContent = t("footerText");

  const submitBtn = document.getElementById("submit-btn");
  if (!submitBtn.disabled) {
    submitBtn.textContent = t("submitBtn");
  }

  // Εξηγήσεις ενοτήτων (tooltips) και προσβάσιμες ετικέτες των εικονιδίων
  document.querySelectorAll(".info-btn").forEach((btn) => {
    const key = btn.getAttribute("data-tip-key");
    const tip = document.getElementById(`tip-${key}`);
    if (tip) tip.textContent = TRANSLATIONS[currentLang].tips[key] || "";
    btn.setAttribute("aria-label", t("infoBtnLabel"));
  });
  if (typeof updateThemeButton === "function") {
    updateThemeButton();
  }

  document.getElementById("history-btn-label").textContent = t("historyBtnLabel");
  document.getElementById("copy-btn-label").textContent = t("copyBtnLabel");
  document.getElementById("print-btn-label").textContent = t("printBtnLabel");
  // Το ίδιο το ιστορικό (κείμενα διλημμάτων, ώρες) ξαναποδίδεται αν
  // υπάρχει ήδη, ώστε να ενημερωθούν και οι μεταφρασμένες ετικέτες χώρας
  if (typeof renderHistoryList === "function") {
    renderHistoryList();
  }

  // Ενημέρωση των ετικετών χώρας/κλάδου στα αναδυόμενα μενού, χωρίς να
  // αλλάζει η εσωτερική (αγγλική) τιμή τους
  document.querySelectorAll("#region option, #eu-country option, #us-state option, #sector option").forEach((opt) => {
    const label = opt.getAttribute(`data-${currentLang}`);
    if (label) opt.textContent = label;
  });

  // Ενημέρωση της οπτικής κατάστασης των κουμπιών γλώσσας
  document.getElementById("lang-btn-en").classList.toggle("active", currentLang === "en");
  document.getElementById("lang-btn-el").classList.toggle("active", currentLang === "el");

  // Ειδοποίηση του main.js ώστε να ξαναποδώσει τυχόν ήδη ορατή αναφορά
  // στη νέα γλώσσα
  // Μόνο όταν ο χρήστης άλλαξε πραγματικά γλώσσα (όχι σε σιωπηλό
  // συγχρονισμό, ούτε όταν πατήθηκε η ήδη ενεργή γλώσσα)
  if (!options.silent && currentLang !== previousLang) {
    document.dispatchEvent(new CustomEvent("languagechange", { detail: { lang: currentLang } }));
  }
}

// -----------------------------------------------------------------------
// Αρχικοποίηση: σύνδεση των κουμπιών γλώσσας και εφαρμογή της
// αποθηκευμένης γλώσσας κατά τη φόρτωση της σελίδας
// -----------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("lang-btn-en").addEventListener("click", () => applyLanguage("en"));
  document.getElementById("lang-btn-el").addEventListener("click", () => applyLanguage("el"));
  applyLanguage(currentLang);
});
