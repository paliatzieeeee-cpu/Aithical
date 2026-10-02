// -----------------------------------------------------------------------
// Αναφορές στα στοιχεία της φόρμας εισαγωγής και των πλαισίων ελέγχου
// -----------------------------------------------------------------------
const form = document.getElementById("dilemma-form");
const dilemmaInput = document.getElementById("dilemma");
const charCount = document.getElementById("char-count");
const regionSelect = document.getElementById("region");
const euCountrySelect = document.getElementById("eu-country");
const usStateSelect = document.getElementById("us-state");
const euCountryField = document.getElementById("eu-country-field");
const usStateField = document.getElementById("us-state-field");
const sectorSelect = document.getElementById("sector");
const submitBtn = document.getElementById("submit-btn");
const formError = document.getElementById("form-error");
const generateNarrativeCheckbox = document.getElementById("generate-narrative");
const narrativeBlock = document.getElementById("narrative-block");
const narrativeText = document.getElementById("narrative-text");
const narrativeError = document.getElementById("narrative-error");
const llmAssistNote = document.getElementById("llm-assist-note");
const liveSourcesCheckbox = document.getElementById("live-sources");
const liveSourcesSection = document.getElementById("section-live-sources");
const liveSourcesList = document.getElementById("live-sources-list");
const liveSourcesErrorEl = document.getElementById("live-sources-error");
const loadingIndicator = document.getElementById("loading-indicator");
const loadingText = document.getElementById("loading-text");
const historyBtn = document.getElementById("history-btn");
const historyPanel = document.getElementById("history-panel");
const historyList = document.getElementById("history-list");
const historyCount = document.getElementById("history-count");
const copyBtn = document.getElementById("copy-btn");
const printBtn = document.getElementById("print-btn");

// -----------------------------------------------------------------------
// Αναφορές στα στοιχεία της περιοχής απεικόνισης της αναφοράς
// -----------------------------------------------------------------------
const reportEmpty = document.getElementById("report-empty");
const report = document.getElementById("report");
const reportContext = document.getElementById("report-context");
const reportDilemma = document.getElementById("report-dilemma");
const dimensionList = document.getElementById("dimension-list");
const principlesList = document.getElementById("principles-list");
const regulationsList = document.getElementById("regulations-list");
const actionsList = document.getElementById("actions-list");
const bibliographySection = document.getElementById("section-bibliography");
const bibliographyList = document.getElementById("bibliography-list");
const reportDisclaimer = document.getElementById("report-disclaimer");
const riskSection = document.getElementById("section-risk");
const riskVerdict = document.getElementById("risk-verdict");
const riskPyramid = document.getElementById("risk-pyramid");
const riskMarker = document.getElementById("risk-marker");
const riskLegend = document.getElementById("risk-legend");
const riskMatchesTitle = document.getElementById("risk-matches-title");
const riskMatchesList = document.getElementById("risk-matches");
const riskObligationsList = document.getElementById("risk-obligations");
const riskNotesList = document.getElementById("risk-notes");

let lastResultVisible = false;

// -----------------------------------------------------------------------
// Ιστορικό αναλύσεων μέσα στην ίδια συνεδρία (session) — ζει μόνο στη
// μνήμη της σελίδας, δεν αποθηκεύεται πουθενά. Κάθε στοιχείο κρατάει τα
// πλήρη δεδομένα απόκρισης, ώστε να μπορεί να ξαναποδοθεί ακριβώς.
// -----------------------------------------------------------------------
let analysisHistory = [];

// -----------------------------------------------------------------------
// Ενημέρωση μετρητή χαρακτήρων κατά την πληκτρολόγηση
// -----------------------------------------------------------------------
dilemmaInput.addEventListener("input", () => {
  charCount.textContent = dilemmaInput.value.length;
});

// -----------------------------------------------------------------------
// Εμφάνιση/απόκρυψη της υποδοχής επιλογής χώρας (ΕΕ) ή πολιτείας (ΗΠΑ),
// ανάλογα με την επιλεγμένη περιοχή. Η επιλογή "General / Other" δεν
// χρειάζεται καμία υπο-επιλογή.
// -----------------------------------------------------------------------
function updateRegionFields() {
  const region = regionSelect.value;
  euCountryField.hidden = region !== "European Union";
  usStateField.hidden = region !== "United States";
}

regionSelect.addEventListener("change", updateRegionFields);
updateRegionFields();

// -----------------------------------------------------------------------
// Η πραγματική τιμή "χώρα" που στέλνεται στο backend: η συγκεκριμένη
// επιλογή (χώρα ΕΕ ή πολιτεία ΗΠΑ) όταν υπάρχει, αλλιώς η ίδια η περιοχή
// (π.χ. "General / Other")
// -----------------------------------------------------------------------
function effectiveCountry() {
  if (regionSelect.value === "European Union") return euCountrySelect.value;
  if (regionSelect.value === "United States") return usStateSelect.value;
  return regionSelect.value;
}

// -----------------------------------------------------------------------
// Υποβολή της φόρμας
// -----------------------------------------------------------------------
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitDilemma();
});

// -----------------------------------------------------------------------
// Αν ο χρήστης αλλάξει γλώσσα ενώ ήδη εμφανίζεται μια αναφορά, γίνεται
// αυτόματη επανυποβολή στη νέα γλώσσα, ώστε να μην μένει στην οθόνη
// αναφορά σε μεικτή γλώσσα
// -----------------------------------------------------------------------
// Κρατάμε το αίτημα της αναφοράς που φαίνεται, ώστε η αλλαγή γλώσσας να
// ξαναζητά ΑΥΤΗ την αναφορά (όχι ό,τι έχει τυχόν αλλάξει στη φόρμα στο
// μεταξύ). Η γλώσσα στέλνεται ως ρητή επιλογή, ώστε ο server να μην την
// παρακάμψει με αυτόματη ανίχνευση.
let lastRequest = null;
let requestInFlight = false;

document.addEventListener("languagechange", () => {
  if (lastResultVisible && lastRequest) {
    submitDilemma({ payload: lastRequest, explicitLang: true });
  }
});

// -----------------------------------------------------------------------
// Κύρια συνάρτηση: αποστολή του διλήμματος στο backend και χειρισμός
// της απόκρισης (επιτυχία, σφάλμα, κατάσταση φόρτωσης)
// -----------------------------------------------------------------------
async function submitDilemma(options = {}) {
  if (requestInFlight) return;  // ποτέ δύο ταυτόχρονα αιτήματα
  hideError();

  let request;
  if (options.payload) {
    request = { ...options.payload };
  } else {
    const dilemma = dilemmaInput.value.trim();
    if (!dilemma) {
      showError(t("formErrorEmpty"));
      return;
    }
    request = {
      dilemma,
      country: effectiveCountry(),
      sector: sectorSelect.value,
      generate_narrative: generateNarrativeCheckbox.checked,
      include_live_sources: liveSourcesCheckbox.checked,
    };
  }

  requestInFlight = true;
  setLoading(true, request.include_live_sources);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...request,
        lang: currentLang,
        lang_explicit: !!options.explicitLang,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      // Φιλικά, δίγλωσσα μηνύματα για τον χρήστη· τα τεχνικά στοιχεία
      // μένουν στο log του server και δεν εμφανίζονται στη σελίδα.
      if (response.status === 429) {
        showError(t("formErrorRateLimit"));
      } else if (response.status >= 500) {
        showError(t("formErrorUnreachable"));
      } else {
        showError(data.error || t("formErrorGeneric"));
      }
      return;
    }

    lastRequest = request;
    renderReport(data);
    addToHistory(data, request);
    lastResultVisible = true;
  } catch (err) {
    showError(t("formErrorUnreachable"));
  } finally {
    requestInFlight = false;
    setLoading(false);
  }
}

// -----------------------------------------------------------------------
// Ενεργοποίηση/απενεργοποίηση κατάστασης φόρτωσης: κρύβει την άδεια
// κατάσταση/αναφορά και δείχνει το spinner, με διαφορετικό μήνυμα αν
// είναι ενεργή η (πιο αργή) ζωντανή αναζήτηση
// -----------------------------------------------------------------------
function setLoading(isLoading, includeLive = false) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? t("submitBtnLoading") : t("submitBtn");

  if (isLoading) {
    reportEmpty.hidden = true;
    report.hidden = true;
    loadingText.textContent = includeLive ? t("loadingTextWithSearch") : t("loadingText");
    loadingIndicator.hidden = false;
  } else {
    loadingIndicator.hidden = true;
    // Επαναφορά στην κατάσταση πριν το αίτημα, αν αυτό απέτυχε (renderReport
    // δεν έτρεξε ξανά ώστε να ξαναβάλει τη σωστή ορατότητα μόνο του)
    report.hidden = !lastResultVisible;
    reportEmpty.hidden = lastResultVisible;
  }
}

// -----------------------------------------------------------------------
// Εμφάνιση / απόκρυψη μηνύματος σφάλματος στη φόρμα
// -----------------------------------------------------------------------
function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function hideError() {
  formError.hidden = true;
  formError.textContent = "";
}

// -----------------------------------------------------------------------
// Απόδοση ολόκληρης της αναφοράς με βάση την απόκριση του backend
// -----------------------------------------------------------------------
function renderReport(data) {
  // Συγχρονισμός του κουμπιού γλώσσας με τη γλώσσα που εντόπισε ο
  // διακομιστής στο ίδιο το δίλημμα (βλ. detect_lang στο advisor.py)
  if (data.lang && data.lang !== currentLang) {
    applyLanguage(data.lang, { silent: true });
  }

  reportEmpty.hidden = true;
  report.hidden = false;

  reportContext.textContent = `${localizedCountryLabel(data.country)} · ${localizedOptionLabel(sectorSelect, data.sector)}`;
  reportDilemma.textContent = data.dilemma;

  renderNarrative(data.narrative, data.narrative_error);
  llmAssistNote.hidden = !data.llm_assisted_detection;
  renderDimensions(data.detected_dimensions);
  renderRiskTier(data.ai_act_risk);
  renderSourceList(principlesList, data.principles, t("noPrinciple"));
  renderBibliography(data.further_reading);
  renderSourceList(regulationsList, data.regulations, t("noRegulation"));
  renderSourceList(actionsList, data.actions, t("noAction"));
  renderLiveSources(data.live_sources, data.live_sources_error);

  reportDisclaimer.textContent = data.disclaimer;

  report.scrollIntoView({ behavior: "smooth", block: "start" });
}

// -----------------------------------------------------------------------
// Εύρεση της μεταφρασμένης ετικέτας μιας χώρας/πολιτείας/περιοχής,
// ψάχνοντας στα τρία σχετικά dropdowns (περιοχή, χώρα ΕΕ, πολιτεία ΗΠΑ)
// -----------------------------------------------------------------------
function localizedCountryLabel(value) {
  for (const selectEl of [regionSelect, euCountrySelect, usStateSelect]) {
    const opt = Array.from(selectEl.options).find((o) => o.value === value);
    if (opt) return opt.getAttribute(`data-${currentLang}`) || value;
  }
  return value;
}

// -----------------------------------------------------------------------
// Εύρεση της μεταφρασμένης ετικέτας μιας επιλογής (χώρα/κλάδος) βάσει
// της τρέχουσας γλώσσας, χωρίς να αλλάζει η εσωτερική (αγγλική) τιμή της
// -----------------------------------------------------------------------
function localizedOptionLabel(selectEl, value) {
  const opt = Array.from(selectEl.options).find((o) => o.value === value);
  if (!opt) return value;
  return opt.getAttribute(`data-${currentLang}`) || value;
}

// -----------------------------------------------------------------------
// Απόδοση της ενότητας βαθμίδας κινδύνου του Κανονισμού ΤΝ: φωτίζεται η
// ζώνη της πυραμίδας που αντιστοιχεί στην περίπτωση και τοποθετείται ο
// δείκτης μέσα σε αυτήν (πιο ψηλά = πιο κοντά στην αμέσως υψηλότερη
// βαθμίδα), μαζί με την αιτιολόγηση, τις υποχρεώσεις και τις σημειώσεις
// -----------------------------------------------------------------------
// Κατακόρυφα όρια κάθε ζώνης στο viewBox του SVG (0 = κορυφή, 240 = βάση)
const RISK_BANDS = {
  unacceptable: { top: 0, bottom: 60 },
  high: { top: 60, bottom: 120 },
  limited: { top: 120, bottom: 180 },
  minimal: { top: 180, bottom: 240 },
};

function renderRiskTier(risk) {
  if (!risk) {
    riskSection.hidden = true;
    return;
  }
  riskSection.hidden = false;
  riskSection.dataset.tier = risk.tier;

  riskVerdict.textContent = risk.verdict;

  riskPyramid.querySelectorAll(".risk-band").forEach((band) => {
    band.classList.toggle("active", band.dataset.tier === risk.tier);
  });
  riskLegend.querySelectorAll("li").forEach((li) => {
    li.classList.toggle("active", li.dataset.tier === risk.tier);
  });

  // Η κορυφαία ζώνη είναι τρίγωνο, οπότε ο δείκτης δεν ανεβαίνει πολύ
  // κοντά στην κορυφή του, όπου δεν θα χωρούσε
  const band = RISK_BANDS[risk.tier];
  const padTop = risk.tier === "unacceptable" ? 34 : 12;
  const span = band.bottom - 12 - (band.top + padTop);
  const y = band.bottom - 12 - risk.position * span;
  riskMarker.setAttribute("transform", `translate(100 ${y.toFixed(1)})`);

  riskMatchesList.innerHTML = "";
  (risk.matches || []).forEach((m) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = m.label;
    const ref = document.createElement("span");
    ref.className = "tag";
    ref.textContent = m.ref;
    li.appendChild(ref);
    li.appendChild(label);
    riskMatchesList.appendChild(li);
  });
  riskMatchesTitle.hidden = !(risk.matches && risk.matches.length);

  riskObligationsList.innerHTML = "";
  (risk.obligations || []).forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    riskObligationsList.appendChild(li);
  });

  riskNotesList.innerHTML = "";
  (risk.notes || []).forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    riskNotesList.appendChild(li);
  });
}

// -----------------------------------------------------------------------
// Απόδοση της ενότητας ζωντανών (μη επαληθευμένων) πηγών
// -----------------------------------------------------------------------
function renderLiveSources(sources, errorMessage) {
  const requested = liveSourcesCheckbox.checked;

  if (!requested) {
    liveSourcesSection.hidden = true;
    return;
  }

  liveSourcesSection.hidden = false;
  liveSourcesList.innerHTML = "";

  if (errorMessage) {
    liveSourcesErrorEl.textContent = errorMessage;
    liveSourcesErrorEl.hidden = false;
  } else {
    liveSourcesErrorEl.hidden = true;
    liveSourcesErrorEl.textContent = "";
  }

  if (!sources || sources.length === 0) {
    if (!errorMessage) {
      const li = document.createElement("li");
      li.className = "empty-note";
      li.textContent = t("liveSourcesEmpty");
      liveSourcesList.appendChild(li);
    }
    return;
  }

  sources.forEach((src) => {
    const li = document.createElement("li");
    li.className = "source-item live-source-item";

    const text = document.createElement("p");
    text.className = "source-text";
    text.textContent = src.title;

    const meta = document.createElement("p");
    meta.className = "source-meta";
    if (src.snippet) {
      meta.appendChild(document.createTextNode(src.snippet.slice(0, 160)));
    }

    const link = document.createElement("a");
    link.href = src.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.className = "source-link";
    link.textContent = currentLang === "el" ? "Άνοιγμα ↗" : "Open ↗";
    meta.appendChild(link);

    li.appendChild(text);
    li.appendChild(meta);
    liveSourcesList.appendChild(li);
  });
}

// -----------------------------------------------------------------------
// Απόδοση της ενότητας σημασιολογικής βιβλιογραφίας (RAG). Η ενότητα
// παραμένει κρυφή όταν δεν υπάρχει κανένα αποτέλεσμα πάνω από το
// κατώφλι ομοιότητας, αντί να εμφανίζει κενή κατάσταση
// -----------------------------------------------------------------------
function renderBibliography(entries) {
  if (!entries || entries.length === 0) {
    bibliographySection.hidden = true;
    return;
  }

  bibliographySection.hidden = false;
  bibliographyList.innerHTML = "";

  entries.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "source-item";

    const text = document.createElement("p");
    text.className = "source-text";
    text.textContent = entry.summary;

    const meta = document.createElement("p");
    meta.className = "source-meta";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = `${Math.round(entry.similarity * 100)}%`;
    meta.appendChild(tag);
    meta.appendChild(document.createTextNode(entry.citation));

    if (entry.url) {
      const link = document.createElement("a");
      link.href = entry.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "source-link";
      link.textContent = currentLang === "el" ? "Άνοιγμα ↗" : "Open ↗";
      meta.appendChild(link);
    }

    li.appendChild(text);
    li.appendChild(meta);
    bibliographyList.appendChild(li);
  });
}

// -----------------------------------------------------------------------
// Απόδοση της σύνοψης που παράχθηκε από LLM (ή του σχετικού σφάλματος)
// -----------------------------------------------------------------------
function renderNarrative(narrative, narrativeErrorMessage) {
  if (narrative) {
    narrativeText.textContent = narrative;
    narrativeBlock.hidden = false;
  } else {
    narrativeBlock.hidden = true;
    narrativeText.textContent = "";
  }

  if (narrativeErrorMessage) {
    narrativeError.textContent = `${narrativeErrorMessage}`;
    narrativeError.hidden = false;
  } else {
    narrativeError.hidden = true;
    narrativeError.textContent = "";
  }
}

// -----------------------------------------------------------------------
// Απόδοση της λίστας εντοπισμένων ηθικών διαστάσεων με τη βαθμολογία τους
// -----------------------------------------------------------------------
function renderDimensions(dimensions) {
  dimensionList.innerHTML = "";

  if (!dimensions || dimensions.length === 0) {
    const li = document.createElement("li");
    li.className = "empty-note";
    li.textContent = t("noDimension");
    dimensionList.appendChild(li);
    return;
  }

  dimensions.forEach((dim) => {
    const li = document.createElement("li");
    li.className = "dimension-item";

    const nameWrap = document.createElement("span");
    const name = document.createElement("span");
    name.className = "dimension-name";
    name.textContent = dim.label;
    const desc = document.createElement("span");
    desc.className = "dimension-desc";
    desc.textContent = dim.description;
    nameWrap.appendChild(name);
    // Διάσταση που προστέθηκε από τη βαθμίδα κινδύνου (όχι από τον ταξινομητή)
    if (dim.added_by_risk) {
      const tag = document.createElement("span");
      tag.className = "dimension-tag";
      tag.textContent = t("addedByRisk");
      tag.title = t("addedByRiskTitle");
      nameWrap.appendChild(tag);
    }
    nameWrap.appendChild(desc);

    const score = document.createElement("span");
    score.className = "dimension-score";

    const barWrap = document.createElement("span");
    barWrap.className = "dimension-score-bar";
    const barFill = document.createElement("span");
    barFill.className = "dimension-score-bar-fill";
    barFill.style.width = `${Math.round(dim.score * 100)}%`;
    barWrap.appendChild(barFill);

    const scoreText = document.createElement("span");
    scoreText.textContent = `${Math.round(dim.score * 100)}%`;

    score.appendChild(barWrap);
    score.appendChild(scoreText);

    li.appendChild(nameWrap);
    li.appendChild(score);
    dimensionList.appendChild(li);
  });
}

// -----------------------------------------------------------------------
// Γενική συνάρτηση απόδοσης λίστας πηγών (αρχές / κανονισμοί / ενέργειες),
// συμπεριλαμβανομένης της νομικής παραπομπής όταν υπάρχει
// -----------------------------------------------------------------------
function renderSourceList(listEl, entries, emptyMessage) {
  listEl.innerHTML = "";

  if (!entries || entries.length === 0) {
    const li = document.createElement("li");
    li.className = "empty-note";
    li.textContent = emptyMessage;
    listEl.appendChild(li);
    return;
  }

  entries.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "source-item";

    const text = document.createElement("p");
    text.className = "source-text";
    text.textContent = entry.text;

    const meta = document.createElement("p");
    meta.className = "source-meta";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = categoryLabel(entry.category);
    meta.appendChild(tag);

    let sourceLine = `${t("sourceLabel")}: ${entry.source}`;
    if (entry.article) {
      sourceLine += ` — ${entry.article}`;
    }
    meta.appendChild(document.createTextNode(sourceLine));

    if (entry.url) {
      const link = document.createElement("a");
      link.href = entry.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "source-link";
      link.textContent = currentLang === "el" ? "Πρωτότυπο κείμενο ↗" : "View primary source ↗";
      meta.appendChild(link);
    }

    li.appendChild(text);
    li.appendChild(meta);
    listEl.appendChild(li);
  });
}

// -----------------------------------------------------------------------
// Μετάφραση του κλειδιού κατηγορίας στην τρέχουσα γλώσσα εμφάνισης
// -----------------------------------------------------------------------
function categoryLabel(key) {
  return TRANSLATIONS[currentLang].categoryLabels[key] || key;
}

// -----------------------------------------------------------------------
// Ιστορικό αναλύσεων μέσα στη συνεδρία: προσθήκη νέας καταχώρισης (με τα
// πλήρη δεδομένα, ώστε να μπορεί να ξαναποδοθεί ακριβώς όπως ήταν), και
// απόδοση της αναπτυσσόμενης λίστας ιστορικού
// -----------------------------------------------------------------------
function addToHistory(data, request) {
  analysisHistory.unshift({ data, request, timestamp: new Date() });
  if (analysisHistory.length > 20) analysisHistory.pop(); // ανώτατο όριο, όχι απεριόριστη μνήμη
  renderHistoryList();
}

function renderHistoryList() {
  historyCount.textContent = analysisHistory.length;
  historyList.innerHTML = "";

  if (analysisHistory.length === 0) {
    const li = document.createElement("li");
    li.className = "history-empty-note";
    li.textContent = t("historyEmpty");
    historyList.appendChild(li);
    return;
  }

  analysisHistory.forEach((entry) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item";

    const dilemmaSpan = document.createElement("span");
    dilemmaSpan.className = "history-item-dilemma";
    dilemmaSpan.textContent = entry.data.dilemma;

    const metaSpan = document.createElement("span");
    metaSpan.className = "history-item-meta";
    metaSpan.textContent = `${localizedCountryLabel(entry.data.country)} · ${entry.timestamp.toLocaleTimeString(currentLang === "el" ? "el-GR" : "en-US")}`;

    btn.appendChild(dilemmaSpan);
    btn.appendChild(metaSpan);
    btn.addEventListener("click", () => {
      renderReport(entry.data);
      lastRequest = entry.request || null;
      lastResultVisible = true;
      closeHistoryPanel();
    });

    li.appendChild(btn);
    historyList.appendChild(li);
  });
}

function closeHistoryPanel() {
  historyPanel.hidden = true;
  historyBtn.setAttribute("aria-expanded", "false");
}

historyBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  const willOpen = historyPanel.hidden;
  historyPanel.hidden = !willOpen;
  historyBtn.setAttribute("aria-expanded", String(willOpen));
});

// Κλείσιμο του πλαισίου ιστορικού με κλικ οπουδήποτε αλλού στη σελίδα
document.addEventListener("click", (event) => {
  if (!historyPanel.hidden && !historyPanel.contains(event.target) && event.target !== historyBtn) {
    closeHistoryPanel();
  }
});

// -----------------------------------------------------------------------
// Δημιουργία απλού, καθαρού κειμένου της τρέχουσας αναφοράς (για
// αντιγραφή στο πρόχειρο) από ό,τι είναι ήδη αποδομένο στη σελίδα —
// έτσι ταιριάζει πάντα ακριβώς με ό,τι βλέπει ο χρήστης
// -----------------------------------------------------------------------
function buildReportText() {
  const lines = [reportContext.textContent, "", reportDilemma.textContent, ""];

  if (!narrativeBlock.hidden) {
    lines.push(narrativeText.textContent, "");
  }

  lines.push(document.getElementById("section-1-title").textContent + ":");
  Array.from(dimensionList.children).forEach((li) => lines.push("- " + li.textContent.replace(/\s+/g, " ").trim()));
  lines.push("");

  if (!riskSection.hidden) {
    lines.push(document.getElementById("section-risk-title").textContent + ":", riskVerdict.textContent);
    [riskMatchesList, riskObligationsList, riskNotesList].forEach((listEl) => {
      Array.from(listEl.children).forEach((li) => lines.push("- " + li.textContent.replace(/\s+/g, " ").trim()));
    });
    lines.push("");
  }

  const sections = [
    ["section-2-title", principlesList],
    ["section-3-title", regulationsList],
    ["section-4-title", actionsList],
  ];
  sections.forEach(([titleId, listEl]) => {
    lines.push(document.getElementById(titleId).textContent + ":");
    Array.from(listEl.children).forEach((li) => lines.push("- " + li.textContent.replace(/\s+/g, " ").trim()));
    lines.push("");
  });

  return lines.join("\n").trim();
}

copyBtn.addEventListener("click", async () => {
  const label = document.getElementById("copy-btn-label");
  try {
    await navigator.clipboard.writeText(buildReportText());
    const original = label.textContent;
    label.textContent = t("copiedLabel");
    setTimeout(() => { label.textContent = original; }, 1600);
  } catch (err) {
    showError(t("copyFailed"));
  }
});

printBtn.addEventListener("click", () => {
  window.print();
});

renderHistoryList(); // αρχική κατάσταση: "καμία ανάλυση ακόμα"


// -----------------------------------------------------------------------
// Φωτεινό / σκούρο θέμα: εναλλαγή σε κάθε πάτημα, με αποθήκευση της
// επιλογής στον browser. Χωρίς αποθηκευμένη επιλογή ακολουθείται η
// ρύθμιση του λειτουργικού συστήματος (prefers-color-scheme).
// -----------------------------------------------------------------------
const THEME_KEY = "aithical-theme";
const themeToggle = document.getElementById("theme-toggle");

function isDark() {
  return document.documentElement.getAttribute("data-theme") === "dark";
}

function updateThemeButton() {
  const label = isDark() ? t("themeToLight") : t("themeToDark");
  themeToggle.setAttribute("aria-label", label);
  themeToggle.setAttribute("title", label);
  themeToggle.setAttribute("aria-pressed", String(isDark()));
}

function setTheme(dark, persist) {
  if (dark) {
    document.documentElement.setAttribute("data-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  if (persist) {
    try { localStorage.setItem(THEME_KEY, dark ? "dark" : "light"); } catch (e) { /* χωρίς αποθήκευση */ }
  }
  updateThemeButton();
}

themeToggle.addEventListener("click", () => setTheme(!isDark(), true));

// Αν ο χρήστης δεν έχει επιλέξει ρητά, ακολουθεί αλλαγές του συστήματος
try {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (err) { /* αγνόηση */ }
    if (!saved) setTheme(e.matches, false);
  });
} catch (e) { /* παλιοί browsers */ }

updateThemeButton();

// -----------------------------------------------------------------------
// Εξηγήσεις ενοτήτων (εικονίδιο «i»): εμφανίζονται με hover σε υπολογιστή
// και με πάτημα σε κινητό/πληκτρολόγιο. Κλείνουν με Esc, με δεύτερο
// πάτημα ή με πάτημα οπουδήποτε αλλού. Μετατοπίζονται οριζόντια αν
// χρειαστεί, ώστε να μη βγαίνουν ποτέ έξω από την οθόνη.
// -----------------------------------------------------------------------
function keepTipOnScreen(tip) {
  tip.style.left = "";
  tip.style.removeProperty("--arrow-left");
  const rect = tip.getBoundingClientRect();
  const overflow = rect.right - (window.innerWidth - 12);
  if (overflow > 0) {
    // Μετατόπιση του πλαισίου προς τα αριστερά, και του «βέλους» κατά το
    // ίδιο ποσό προς τα δεξιά, ώστε να δείχνει πάντα το εικονίδιο.
    tip.style.left = `${parseFloat(getComputedStyle(tip).left) - overflow}px`;
    tip.style.setProperty("--arrow-left", `${16 + overflow}px`);
  }
}

function closeAllTips(except) {
  document.querySelectorAll(".info-btn[aria-expanded='true']").forEach((btn) => {
    if (btn === except) return;
    btn.setAttribute("aria-expanded", "false");
    document.getElementById(btn.getAttribute("aria-controls")).classList.remove("open");
  });
}

document.querySelectorAll(".info-wrap").forEach((wrap) => {
  const btn = wrap.querySelector(".info-btn");
  const tip = wrap.querySelector(".info-tip");

  wrap.addEventListener("mouseenter", () => keepTipOnScreen(tip));
  btn.addEventListener("focus", () => keepTipOnScreen(tip));
  btn.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = btn.getAttribute("aria-expanded") !== "true";
    closeAllTips(btn);
    btn.setAttribute("aria-expanded", String(willOpen));
    tip.classList.toggle("open", willOpen);
    if (willOpen) keepTipOnScreen(tip);
  });
});

document.addEventListener("click", () => closeAllTips(null));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeAllTips(null);
});
