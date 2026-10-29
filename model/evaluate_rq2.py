"""
Evaluation for research question 2: does classify -> retrieve -> compose give
traceable guidance?

Usage (from the project root; parts 2 and 3 need GROQ_API_KEY in .env):

    python -m model.evaluate_rq2 --out rq2_results.json
    python -m model.evaluate_rq2 --skip-llm        # retrieval only, no API calls

1. Retrieval: for the 24 scenarios of model/rq2_cases.py, which expected
   provisions appear among the sources of the report (knowledge base and
   risk-tier legal basis).
2. Summary faithfulness: every legal reference in the LLM summary (articles,
   annexes, numbered acts) is checked against the sources the LLM was given.
   Unsupported references are listed for manual review.
3. Direct LLM baseline: the same model is asked the scenario directly, with no
   retrieved sources. Its answer is scored against the same gold set, and
   every cited article is checked against the size of the instrument.

The LLM is called with the production settings (Groq, LLM_PROVIDER is set
for this process only; .env is not changed).
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import argparse
import json
import os
import re
import time

from dotenv import load_dotenv

from engine.advisor import US_STATES, analyze, load_knowledge_base
from engine.bibliography import retrieve_bibliography
from engine.llm_client import LLMUnavailable, _call_groq, generate_narrative
from engine.risk_tier import assess_risk_tier
from model.ethics_classifier import EthicsClassifier
from model.rq2_cases import CASES, CASE_STUDIES, AIA, GDPR, CCD, MDR, UCPD, CRD, DSM, EQ, GR_AI, CCPA, TITLE7

INSTRUMENTS = [AIA, GDPR, CCD, MDR, UCPD, CRD, DSM, EQ, GR_AI, CCPA, TITLE7]

# Πλήθος άρθρων κάθε νομοθετήματος, για τον έλεγχο ανύπαρκτων άρθρων
MAX_ARTICLE = {"AI Act": (AIA, 113), "GDPR": (GDPR, 99), "Directive 2023/2225": (CCD, 50),
               "Directive 2005/29/EC": (UCPD, 21), "MDR": (MDR, 123), "Directive 2019/790": (DSM, 32),
               "Directive 2011/83/EU": (r"2011/83", 35), "Directive 2000/78/EC": (EQ, 21)}

DIRECT_SYSTEM_PROMPT = (
    "You are an expert in EU and national law on artificial intelligence, data protection and consumer "
    "protection. Answer the user's question about their planned use of AI."
)
DIRECT_USER_PROMPT = (
    "Country: {country}\nBusiness function: {sector}\nSituation: {text}\n\n"
    "List the specific legal provisions that apply (name of the act and article, annex or point), one per "
    "line, each with a one-sentence explanation. Then state the EU AI Act risk level, if it applies."
)

REF_RE = re.compile(
    r"(?:Art(?:icle|s?\.)?|Άρθρ\w*)\s*(\d+)"                      # άρθρο
    r"|(?:Annex|Παράρτημα|Παρ\.)\s*([IVX]+)"                      # παράρτημα
    r"|\b(\d{4}/\d{1,4}|\d{1,4}/\d{4})\b",                          # αριθμός πράξης (2016/679, 5321/2026)
    re.IGNORECASE,
)


# -----------------------------------------------------------------------
# Βοηθητικές συναρτήσεις: αναφορές της αναφοράς της εφαρμογής
# -----------------------------------------------------------------------
def _pick(value):
    return value.get("en") if isinstance(value, dict) else (value or "")


def report_references(result):
    """Every legal reference of a report as a list of strings, one per provision."""
    refs = []
    for entry in result["regulations"] + result["principles"] + result["actions"]:
        context = f"{entry.get('source', '')} {entry.get('citation', '')}"
        for field in (_pick(entry.get("article")), entry.get("citation", "")):
            for segment in (field or "").split(";"):
                own = any(re.search(p, segment) for p in INSTRUMENTS)
                refs.append(segment if own else f"{context} | {segment}")
    for match in result["ai_act_risk"].get("matches", []):
        refs.append(f"EU AI Act {match['ref']}")
    return refs


def found(expected, refs):
    _, instrument, provision = expected
    return any(re.search(instrument, r) and re.search(provision, r) for r in refs)


def run_app(case, classifier, kb):
    risk = assess_risk_tier(case["text"], sector=case["sector"],
                            in_eu=not (case["country"] in US_STATES or case["country"] == "United States"),
                            lang=case["lang"])
    result = analyze(case["text"], case["country"], case["sector"], classifier, kb=kb, lang=case["lang"],
                     extra_dimensions=risk["dimensions"])
    result["ai_act_risk"] = risk
    result["further_reading"] = retrieve_bibliography(case["text"], lang=case["lang"])
    return result


def context_text(result):
    """The text the LLM receives as sources (same fields as the narrative prompt)."""
    parts = []
    for e in result["principles"] + result["regulations"] + result["actions"]:
        parts += [e.get("text", ""), e.get("source", ""), str(e.get("article", ""))]
    for m in result["ai_act_risk"].get("matches", []):
        parts.append(m["ref"])
    parts += result["ai_act_risk"].get("notes", [])
    parts += [b.get("citation", "") + " " + b.get("summary", "") for b in result["further_reading"]]
    return " ".join(parts)


def refs_in(text):
    out = set()
    for m in REF_RE.finditer(text):
        if m.group(1):
            out.add(("article", m.group(1)))
        elif m.group(2):
            out.add(("annex", m.group(2).upper()))
        else:
            out.add(("act", m.group(3)))
    return out


def with_retry(fn, *args, **kwargs):
    for attempt in range(5):
        try:
            return fn(*args, **kwargs)
        except LLMUnavailable as exc:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
            print(f"    retry after: {exc}")


# -----------------------------------------------------------------------
# Έλεγχος ανύπαρκτων άρθρων στην απευθείας απάντηση του LLM
# -----------------------------------------------------------------------
def nonexistent_articles(text):
    problems = []
    for line in text.split("\n"):
        for name, (pattern, maximum) in MAX_ARTICLE.items():
            if not re.search(pattern, line):
                continue
            for m in re.finditer(r"(?:Art(?:icle|s?\.)?|Άρθρ\w*)\s*(\d+)", line, re.IGNORECASE):
                if int(m.group(1)) > maximum:
                    problems.append(f"{name} Art. {m.group(1)}: {line.strip()[:120]}")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--rescore", default=None,
                        help="JSON of an earlier run: score its saved LLM answers again, with no API calls")
    args = parser.parse_args()
    saved = json.load(open(args.rescore, encoding="utf-8")) if args.rescore else None

    load_dotenv()
    os.environ["LLM_PROVIDER"] = "groq"                      # μόνο για αυτή τη διεργασία
    os.environ["LLM_CLASSIFICATION_FALLBACK"] = "false"      # όπως στη δημόσια έκδοση
    kb = load_knowledge_base()
    classifier = EthicsClassifier()
    out = {"retrieval": [], "summaries": [], "direct": []}

    # ---------------------------------------------------------------
    # 1. Ανάκτηση
    # ---------------------------------------------------------------
    total = hit = 0
    results = {}
    for case in CASES:
        result = run_app(case, classifier, kb)
        results[case["id"]] = result
        refs = report_references(result)
        rows = [(label, found(exp, refs)) for exp in case["expected"] for label in [exp[0]]]
        total += len(rows)
        hit += sum(ok for _, ok in rows)
        n_sources = len(result["regulations"]) + len(result["principles"]) + len(result["actions"])
        out["retrieval"].append({"id": case["id"], "tier": result["ai_act_risk"]["tier"],
                                 "dimensions": [d["key"] for d in result["detected_dimensions"]],
                                 "n_sources": n_sources, "expected": rows})
        missed = [label for label, ok in rows if not ok]
        print(f"{case['id']} {case['country']:<12} tier={result['ai_act_risk']['tier']:<12} "
              f"{sum(ok for _, ok in rows)}/{len(rows)}" + (f"  missed: {missed}" if missed else ""))
    print(f"\nRetrieval: {hit}/{total} expected provisions found ({hit / total:.0%})")
    full = sum(all(ok for _, ok in r["expected"]) for r in out["retrieval"])
    print(f"Scenarios with every expected provision: {full}/{len(CASES)}")

    if not args.skip_llm:
        # -----------------------------------------------------------
        # 2. Πιστότητα των συνόψεων του LLM
        # -----------------------------------------------------------
        print("\nSummaries:")
        for case in CASES + CASE_STUDIES:
            result = results.get(case["id"]) or run_app(case, classifier, kb)
            if saved:
                narrative = next(s["narrative"] for s in saved["summaries"] if s["id"] == case["id"])
            else:
                narrative = with_retry(
                    generate_narrative, dilemma=case["text"], country=result["country"], sector=result["sector"],
                    detected_dimensions=result["detected_dimensions"], principles=result["principles"],
                    regulations=result["regulations"], actions=result["actions"],
                    literature=result["further_reading"], risk=result["ai_act_risk"], lang=case["lang"])
                time.sleep(3)
            given = refs_in(context_text(result))
            cited = refs_in(narrative)
            unsupported = sorted(cited - given)
            out["summaries"].append({"id": case["id"], "lang": case["lang"], "narrative": narrative,
                                     "cited": sorted(cited), "unsupported": unsupported})
            print(f"  {case['id']}: {len(cited)} references, unsupported: {unsupported or '-'}")

        # -----------------------------------------------------------
        # 3. Απευθείας ερώτηση στο LLM, χωρίς ανάκτηση
        # -----------------------------------------------------------
        print("\nDirect LLM:")
        d_total = d_hit = 0
        for case in CASES:
            if saved:
                answer = next(d["answer"] for d in saved["direct"] if d["id"] == case["id"])
            else:
                answer = with_retry(_call_groq, DIRECT_SYSTEM_PROMPT, DIRECT_USER_PROMPT.format(**case))
                time.sleep(3)
            lines = [l for l in answer.split("\n") if l.strip()]
            rows = [(exp[0], found(exp, lines)) for exp in case["expected"]]
            d_total += len(rows)
            d_hit += sum(ok for _, ok in rows)
            problems = nonexistent_articles(answer)
            out["direct"].append({"id": case["id"], "answer": answer, "expected": rows, "nonexistent": problems})
            print(f"  {case['id']}: {sum(ok for _, ok in rows)}/{len(rows)}"
                  + (f"  nonexistent: {problems}" if problems else ""))
        print(f"\nDirect LLM: {d_hit}/{d_total} expected provisions named ({d_hit / d_total:.0%})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"Raw results written to {args.out}")


if __name__ == "__main__":
    main()
