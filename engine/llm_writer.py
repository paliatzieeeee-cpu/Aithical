"""
Optional LLM "phrasing" layer for Aithicist.

This module is the hybrid piece discussed for the thesis: the TensorFlow
classifier and knowledge-base retrieval (engine/advisor.py) remain the
source of truth for *what* is said — every claim traces back to a cited
principle, regulation, or action. This module only asks an LLM to phrase
those already-retrieved sources into a smoother, more natural consulting
paragraph. The prompt explicitly forbids introducing new claims, which
keeps the hallucination surface small and keeps the report auditable.

Design choices worth defending in the thesis write-up:
- Off by default. Sending a user's dilemma text to a third-party API is
  the one step in the pipeline that leaves the local machine, which is
  in tension with the tool's own subject matter (advising companies on
  data handling). The web app only calls this when the user explicitly
  opts in (see the "generate_narrative" flag in app.py).
- Provider-agnostic. Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic
  in your environment (see .env.example) and this module calls the
  matching SDK. Both SDKs are imported lazily so the app still runs
  with neither installed, as long as the narrative feature isn't used.
- Grounded prompting. The prompt includes only the dilemma, the detected
  dimensions, and the exact retrieved source texts — never the whole
  knowledge base — and instructs the model to use only that material.
"""

import os

SYSTEM_PROMPT = (
    "You are a compliance-writing assistant. You will be given a business "
    "dilemma involving AI, a set of detected ethical dimensions, and a list "
    "of sourced principles, regulations, and recommended actions retrieved "
    "from a curated knowledge base. Write a short (120-180 word) advisory "
    "paragraph in plain, professional English that weaves these points "
    "together for the reader. Rules: "
    "(1) Use ONLY the information given to you below — do not introduce any "
    "new fact, law, statistic, or recommendation that is not already present "
    "in the provided sources. "
    "(2) Do not invent citations or sources. "
    "(3) If the provided material is thin, keep the paragraph short rather "
    "than padding it with invented content. "
    "(4) End with a one-sentence reminder that this is general guidance, "
    "not legal advice."
)


class NarrativeUnavailable(Exception):
    """Raised when the narrative step can't run (no key, no provider, API error)."""


def _build_user_prompt(dilemma, country, sector, detected_dimensions, principles, regulations, actions):
    def fmt(entries):
        if not entries:
            return "  (none retrieved)"
        return "\n".join(f"  - {e['text']} [source: {e['source']}]" for e in entries)

    dims = ", ".join(d["label"] for d in detected_dimensions) or "none strongly detected"

    return (
        f"Country: {country}\n"
        f"Business function: {sector}\n"
        f"Dilemma: {dilemma}\n\n"
        f"Detected ethical dimensions: {dims}\n\n"
        f"Principles:\n{fmt(principles)}\n\n"
        f"Regulatory considerations:\n{fmt(regulations)}\n\n"
        f"Recommended actions:\n{fmt(actions)}\n"
    )


def _call_openai(user_prompt: str) -> str:
    from openai import OpenAI  # lazy import

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise NarrativeUnavailable("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        max_tokens=350,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(user_prompt: str) -> str:
    import anthropic  # lazy import

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise NarrativeUnavailable("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    response = client.messages.create(
        model=model,
        max_tokens=350,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_narrative(dilemma, country, sector, detected_dimensions, principles, regulations, actions) -> str:
    """Return an LLM-phrased paragraph grounded in the given retrieved sources.

    Raises NarrativeUnavailable if no provider/key is configured or the call fails.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider not in ("openai", "anthropic"):
        raise NarrativeUnavailable(
            "Set LLM_PROVIDER to 'openai' or 'anthropic' in your environment to enable this feature."
        )

    user_prompt = _build_user_prompt(dilemma, country, sector, detected_dimensions, principles, regulations, actions)

    try:
        if provider == "openai":
            return _call_openai(user_prompt)
        return _call_anthropic(user_prompt)
    except NarrativeUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - network/SDK errors
        raise NarrativeUnavailable(f"LLM call failed: {exc}") from exc
