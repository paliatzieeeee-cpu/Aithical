# AI-thical — AI Ethics & Legal Advisory Prototype

A Flask + TensorFlow + HTML/CSS/JS prototype built for the MSc thesis
*"Aithical: The AI Consulting LLM for Ethical and Legal issues about AI"*
(Konstantinos Soldatos, MSc Artificial Intelligence, Aegean College).

A user describes a business dilemma involving AI, picks a country or US state
and a business function, and gets back a structured, sourced advisory report:

```
dilemma text
   -> TensorFlow classifier: 5 ethical dimensions
      (transparency, fairness, non-maleficence, accountability, privacy)
   -> curated knowledge base, filtered by country/state + sector
   -> semantic RAG match against a bibliography of AI-ethics literature
   -> optional live web search (clearly marked as unverified)
   -> report: dimensions, principles, regulations (real legal citations),
      recommended actions, matched literature, optional AI-drafted summary
```

- **Bilingual** (English/Greek) UI and knowledge base; the dilemma's language
  is detected automatically.
- **Coverage:** all 27 EU member states and all 50 US states + D.C., falling
  back to EU-wide or US-federal rules where no local law exists.
- **Auditability:** every principle, regulation and action shown traces to a
  named source in `data/knowledge_base.json`.
- **UI extras:** per-session history, copy report, print / save as PDF.
- **Protection:** `/api/analyze` is rate-limited (10 requests per minute per
  visitor), which matters once the app is reachable from the internet.

---

## 1. Requirements

- **Python 3.10 or 3.11**
- About 2 GB of free disk space (TensorFlow plus the multilingual embedding
  model, downloaded once)
- Internet the first time you train (downloads a ~470 MB embedding model)
- **Windows only:** enable *Long Path Support* **before** installing
  dependencies. TensorFlow contains file paths longer than Windows' default
  260-character limit, and without this setting the install silently ends up
  incomplete. In PowerShell **run as Administrator**:

  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```

  Then **restart the computer**.

Optional, only for the matching feature:

- [Ollama](https://ollama.com) — free, local LLM (nothing leaves your machine)
- An OpenAI or Anthropic API key — paid alternative to Ollama
- A [Tavily](https://www.tavily.com) API key — free, for live web search

---

## 2. First-time setup

### 2.1 Open the project

Extract the archive and open the `aithicist` folder (the one containing
`app.py`) in your terminal or editor.

### 2.2 Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

`(venv)` should appear at the start of the prompt. Keep **one** virtual
environment only; having both `venv` and `.venv` in the same folder is the
most common cause of "it works in the terminal but not in PyCharm" (see
Troubleshooting).

### 2.3 Install dependencies

```bash
python -m pip install -r requirements.txt
```

Check that TensorFlow installed correctly:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

It should print `2.21.0` (the `oneDNN` lines above it are harmless info
messages).

### 2.4 Train the classifier

```bash
python -m model.train_classifier
```

The first run downloads the embedding model; training itself takes under a
minute on a laptop CPU and saves `saved_model/ethics_classifier_head.keras`.
Without a trained model the app still runs, using a simple keyword-matching
fallback instead.

### 2.5 Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000**. The core app (classification, knowledge
base, bilingual UI, citations) works with no further setup.

---

## 3. Optional features

All settings live in a file named **`.env`** in the project root. Create it
once from the template:

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

Edit **`.env`**, not `.env.example` — the app only reads `.env`. If you already
have a `.env`, don't overwrite it; add the new lines by hand.

### 3.1 Local AI summary with Ollama (free, local)

1. Install Ollama from https://ollama.com.
2. Pull a model:
   ```bash
   ollama pull llama3.1
   ```
3. Make sure it's running: open http://localhost:11434 in a browser. If
   nothing answers, run `ollama serve`.
4. In `.env`:
   ```
   LLM_PROVIDER=ollama
   ```
5. Restart `python app.py`.

This enables the "AI-drafted summary" checkbox and a classification second
opinion for dilemmas the local model is unsure about. Small local models
write noticeably weaker Greek than English.

### 3.2 OpenAI or Anthropic instead of Ollama

1. In `.env`: `LLM_PROVIDER=anthropic` (or `openai`) and the matching
   `ANTHROPIC_API_KEY=...` (or `OPENAI_API_KEY=...`).
2. Install the SDK: `python -m pip install anthropic` (or `openai`).
3. Optional, to also use it as a classification second opinion (off by
   default because it costs per call): `LLM_CLASSIFICATION_FALLBACK=true`.
4. Restart `python app.py`.

Trade-off worth stating in the thesis: an external API usually gives better
summaries and better Greek, but the dilemma text leaves your machine; Ollama
keeps everything local.

### 3.3 Live web search (Tavily)

1. Create a free Tavily account (no credit card, 1,000 searches per month)
   and copy your API key.
2. In `.env`: `TAVILY_API_KEY=your-key`
3. Restart `python app.py`.

Results appear in their own section, marked *live, unverified*. Results that
don't mention the selected country are filtered out, and if a specific search
finds nothing relevant the app retries with a broader one. If the key is
missing or a request fails, the section says so instead of failing silently.

### 3.4 Public access through your own domain (Cloudflare Tunnel)

The app can be reached from the internet while it runs on your laptop,
without opening ports:

- **Quick, temporary link:** `cloudflared tunnel --url http://localhost:5000`
  (the address changes every time).
- **Permanent domain:** create a named tunnel in the Cloudflare Zero Trust
  dashboard, add a public hostname pointing to `http://localhost:5000`, then
  run the connector command the dashboard gives you (it contains your token).

In both cases the site is online only while `python app.py` **and** the
tunnel are running. Run the tunnel only when you need it.

---

## 4. Everyday workflow

1. Open a terminal in the project folder and activate the environment
   (`venv\Scripts\activate`).
2. If you use Ollama, make sure it's running.
3. `python app.py` and open http://127.0.0.1:5000.
4. If you want public access, start the tunnel in a second terminal.

After editing a **Python** file or `templates/index.html`, stop the server
(<kbd>Ctrl+C</kbd>) and start it again. After editing **CSS or JS**, a hard
refresh (<kbd>Ctrl+Shift+R</kbd>) is enough.

---

## 5. How the classifier is trained and measured

**Model.** A frozen multilingual sentence encoder
(`paraphrase-multilingual-MiniLM-L12-v2`) turns each dilemma into a vector; a
small TensorFlow/Keras head (64 → 32 units, dropout 0.5/0.4, L2
regularisation) predicts the five dimensions. The head was deliberately made
smaller and more regularised after evaluation showed it memorised the wording
of the training set.

**Evaluation.** `python -m model.evaluate_classifier` prints two parts:

1. **5-fold cross-validation** on the training set, plus the best detection
   threshold for each dimension.
2. **A frozen held-out set of 62 hand-written examples** that never appear in
   training (about 15 positives per dimension, 15 in Greek, 8 neutral). Its
   topics are deliberately crossed with labels the model tends not to
   associate with them (hiring × privacy, children × accountability, health ×
   fairness), so it measures what matters: whether the model detects the
   ethical *issue* rather than guessing from the *topic*.

Rules for honest measurement:

- **Never edit the held-out set** (in `model/evaluate_classifier.py`).
  Changing it makes new results incomparable with old ones.
- The script warns automatically if a held-out example has leaked into
  `training_data.py`. If it does, remove it from the training data.
- Judge a change by the held-out part, not by cross-validation alone. With a
  small dataset, cross-validation can look good while the model generalises
  poorly.
- The random seed is fixed, so the same data always gives the same numbers.

---

## 6. Improving the classifier — the full cycle

Every improvement follows the same loop:

```
baseline -> add reviewed data -> check balance -> retrain -> evaluate -> compare with baseline
```

### Step 1 — Record a baseline

```bash
python -m model.train_classifier
python -m model.evaluate_classifier
```

Save the output. This is what every later change is compared against.

### Step 2a — Add data from real usage (`review_log`)

Every `/api/analyze` call is logged to `data/usage_log.jsonl`. The log never
trains the model by itself: letting a model learn from its own predictions
would reinforce its mistakes. Review it with:

```bash
python -m model.review_log
```

For each logged dilemma you see the text and the model's scores:

| Key | Action |
|---|---|
| Enter | accept the suggestion (top dimension if ≥ 70%, top two if ≥ 40%, otherwise top three) |
| `privacy,fairness` | use your own labels |
| `n` | neutral (no ethical dimension) |
| `d` | drop (tests, spam, unclear or ambiguous text); never enters training |
| `s` | skip for now |
| `q` | quit |

Exact duplicates are removed automatically before review starts. Dropped
entries are recorded as `discarded` in `data/usage_log_reviewed.jsonl`, so if
the exact same text is submitted again later it is removed automatically too.

### Step 2b — Add systematically generated data (`augment_with_llm`)

Evaluation showed the model had learned **topic shortcuts** — hiring →
fairness, children → harm, health → harm — because in hand-written data each
topic nearly always came with the same label. Small manual additions only
moved the errors around. This script generates candidates on a
**topic × dimension grid**, so every topic appears with every label. Each of
the 12 topics gets 8 cells: the 5 single dimensions, 2 random pairs, and one
**neutral** cell — ordinary questions about using AI in that context with no
ethical issue at all (e.g. choosing or setting up a tool). The neutral cells
matter: they teach the model that mentioning AI, or a topic like hiring, is
not an ethical problem by itself. It uses the `LLM_PROVIDER` from `.env`.

```bash
python -m model.augment_with_llm --dry-run       # show the prompt, no calls
python -m model.augment_with_llm --max-cells 3   # quick quality check
python -m model.augment_with_llm                 # full grid (96 calls)
```

Useful options:

| Option | Meaning |
|---|---|
| `--langs en` | English only (recommended with small local models) |
| `--topics hiring,health` | run only some topics, to split the work |
| `--per-cell 3` | examples requested per cell |
| `--seed 7` | a different seed gives different pairs and more variety |
| `--neutral-per-topic 2` | more neutral cells per topic (default 1) |
| `--fill-missing` | run only the grid cells that have no candidates yet |
| `--cells-for fairness,neutral` | only single-dimension cells for these labels (top-up) |

With Ollama on a CPU the full grid takes roughly 35–90 minutes. If calls time
out, add `OLLAMA_TIMEOUT=300` to `.env`.

Candidates go to `data/augmentation_candidates.jsonl`. Duplicates and
anything too similar to a held-out example are rejected automatically.
**Nothing enters the training set until you approve it:**

```bash
python -m model.review_augmentation
```

| Key | Action |
|---|---|
| Enter | accept with the proposed labels |
| `privacy,fairness` | accept with your own labels |
| `n` | accept as neutral |
| `d` | discard (unnatural text, poor Greek, ambiguous) |
| `s` | skip for now |
| `q` | quit |

Be strict: a generated example whose label is wrong does more damage than a
missing one.

### Step 2c — Check the balance before training

Approving everything the generator produces does **not** automatically give a
balanced dataset: some cells fail, random pairs can repeat, and one dimension
may already dominate. A dominant dimension becomes the model's "default
guess"; a rare one gets missed. Check before training:

```bash
python -m model.dataset_stats
```

It shows how many examples each dimension and the neutral class have now,
and what the totals would be if all pending candidates were approved. If the
most frequent dimension has more than 1.2× the examples of the rarest, or
neutrals are under 10%, it prints the exact command to top up only what is
missing, for example:

```bash
python -m model.augment_with_llm --langs en --seed 7 --cells-for fairness,accountability,neutral
```

`--cells-for` generates single-dimension cells (no pairs) only for the listed
labels, so the dimensions that are already frequent don't grow further.
If some cells failed during generation, recover just those with:

```bash
python -m model.augment_with_llm --langs en --fill-missing
```

Repeat *top up → review → dataset_stats* until the report says the balance is
good. While a dimension is over-represented, discard doubtful candidates of
that dimension rather than accepting them.

### Step 3 — Retrain, evaluate, compare

```bash
python -m model.train_classifier
python -m model.evaluate_classifier
```

Compare the held-out part with your baseline from Step 1.

### Step 4 — Update detection thresholds (only after a confirmed improvement)

The app uses a separate detection threshold per dimension, in
`DETECTION_THRESHOLDS` in `engine/advisor.py`. `evaluate_classifier` prints
suggested values. Copy them **only** when the new model is clearly better on
the held-out set; thresholds tuned on a worse model make the app worse.

### Where new examples land in `training_data.py`

`model/training_data.py` has two separate blocks that the scripts write into
automatically:

- **LLM-augmented examples** — inserted above the line
  `# --- end of LLM-augmented examples ---`
- **Reviewed real-usage prompts** — inserted above the line
  `# --- neutral / low-signal examples`

Don't delete or move these two marker lines. **Keep a backup of
`training_data.py`** before replacing it with a new version from elsewhere,
so your own reviewed additions are never lost.

---

## 7. Project structure

```
aithicist/
├── app.py                     # Flask app: routes, rate limiting, live search
├── requirements.txt
├── .env.example               # template for .env
├── engine/
│   ├── advisor.py             # classify -> retrieve -> compose; thresholds;
│   │                          # bilingual + EU/US jurisdiction logic
│   ├── bibliography.py        # semantic RAG over the literature corpus
│   ├── llm_client.py          # Ollama / OpenAI / Anthropic client
│   ├── live_search.py         # live web search (Tavily)
│   └── usage_log.py           # logs real usage for later review
├── model/
│   ├── training_data.py       # labelled dilemmas (EN + EL)
│   ├── ethics_classifier.py   # embedding + TF classifier head
│   ├── train_classifier.py    # training -> saved_model/
│   ├── evaluate_classifier.py # cross-validation + frozen held-out set
│   ├── review_log.py          # review real usage -> training data
│   ├── augment_with_llm.py    # generate candidates (topic x dimension grid)
│   ├── dataset_stats.py       # label balance report + top-up suggestion
│   └── review_augmentation.py # review generated candidates -> training data
├── data/
│   ├── knowledge_base.json    # principles / regulations / actions
│   └── bibliography.json      # literature corpus for RAG
├── templates/index.html
├── static/css/style.css
├── static/js/i18n.js          # EN/EL translations
├── static/js/main.js          # form, rendering, history, export
└── saved_model/               # created by training (not in the archive)
```

Git-ignored working files created while you use the app: `.env`,
`data/usage_log*.jsonl`, `data/augmentation_*.jsonl`, `saved_model/`.

---

## 8. Troubleshooting (mostly Windows)

| Symptom | Cause and fix |
|---|---|
| `No module named 'tensorflow.python'`, or pip reports a path that doesn't exist | Windows Long Path Support is off (see section 1). Enable it, restart, then `pip uninstall tensorflow tf-keras -y`, `pip cache purge`, `pip install --no-cache-dir -r requirements.txt`. |
| Works in the terminal but not with PyCharm's Run button | PyCharm uses a different interpreter (e.g. `.venv` instead of `venv`). *Settings → Project → Python Interpreter → Add Interpreter → Local → Existing*, and choose `venv\Scripts\python.exe`. Check which one the terminal uses with `python -c "import sys; print(sys.executable)"`. |
| `Keras 3 ... not yet supported in Transformers` | `tf-keras` is missing: `pip install -r requirements.txt`. |
| `TemplateNotFound: index.html`, or `index.html` keeps disappearing | Windows Defender wrongly flags it and deletes it. Add the project folder (and your downloads folder) as an exclusion: *Windows Security → Virus & threat protection → Manage settings → Exclusions*. Check with `Get-MpThreatDetection` in PowerShell. |
| The app says the Tavily key is missing although you set it | The key is in `.env.example` instead of `.env`. |
| A UI change doesn't show up | Browser cache: hard refresh (<kbd>Ctrl+Shift+R</kbd>) or an incognito window. |
| Ollama calls time out during augmentation | Add `OLLAMA_TIMEOUT=300` to `.env`, or use a smaller model. |
| `Too many requests` in the UI | The rate limit (10 per minute per visitor); wait a minute. |

---

## 9. Extending it for the thesis

- **Countries, sectors, sources:** add entries to `data/knowledge_base.json`.
  Each needs `category`, `country`, `sector`, `type` (`principle` /
  `regulation` / `action`), bilingual `text`, `source`, and for regulations
  also `article`, `url`, `citation`. An entry that concerns several ethical
  dimensions can also list them in `categories`; it is shown when any of them
  is detected.
- **Business sectors:** 12 sectors are defined in `sectors` / `sector_labels`
  (the form's dropdown is built from them). Entries tagged with a specific
  sector are always shown for that sector, whatever the classifier detects,
  as the sector's baseline rules; `General / Other` entries are filtered by
  the detected dimensions. To add a sector, add it to both lists and give it
  its own regulations and actions.
- **Bibliography for RAG:** add entries to `data/bibliography.json` (`id`,
  `citation`, `url`, and a bilingual `summary` in your own words). Picked up
  on restart. Verify that every source exists before adding it.
- **Training data:** preferably through the review scripts in section 6, so
  every addition has an audit trail.

---

## 10. Notes and limitations

- The training set is small (about 240 hand-written examples, EN + EL, before
  augmentation). It demonstrates the architecture; it is not a
  production-grade classifier.
- A frozen sentence encoder mostly encodes a text's *topic*. With little
  data, the classifier head learns topic shortcuts; the held-out set is
  designed to expose this, and grid-based augmentation is designed to reduce
  it.
- LLM-generated training data is only as reliable as its human review.
- Regulatory citations were verified individually, but a final legal review
  is still recommended before treating any of it as authoritative.
- The embedding step uses `sentence-transformers` (PyTorch); the trained
  classifier head is TensorFlow/Keras.
- The app gives general guidance, not legal advice, and says so in every
  report.
