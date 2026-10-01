"""
Optional live web search for recent legislative/regulatory updates.

This is a free (1,000 credits/month, no credit card) search via the
official Tavily API — replacing an earlier unofficial DuckDuckGo-scraping
approach (the `ddgs` package), which turned out to be unreliable in
practice: undocumented, unannounced rate-limiting; results with no
clear relevance signal for smaller countries. Tavily is a proper,
documented REST API built specifically for feeding search results into
LLM/AI applications like this one, so failures are explicit (a real
HTTP error) rather than a silent, ambiguous empty list.

Used only to surface a few *live* links alongside the curated, cited
knowledge base — never as a replacement for it. The curated knowledge
base (data/knowledge_base.json) remains the source of every claim made
in the structured report; this module only adds "recent sources worth
checking" links, kept in their own clearly-labeled section in the UI,
since anything found this way is unverified — it could be outdated,
off-topic, or simply wrong despite matching keywords.

This matters for the thesis' auditability argument: the report's core
sections (principles, regulations, actions) stay 100% traceable to a
vetted source. Live search results are additive and explicitly flagged
as unverified, so the distinction between "we checked this" and "here's
something recent you might want to check yourself" stays visible to the
reader rather than being blurred together.

Setup: create a free account at https://www.tavily.com (no credit card),
copy your API key, and set TAVILY_API_KEY in your .env file.
"""

# -----------------------------------------------------------------------
# Εισαγωγές βιβλιοθηκών
# -----------------------------------------------------------------------
import os

import requests

TAVILY_ENDPOINT = "https://api.tavily.com/search"
REQUEST_TIMEOUT_SECONDS = 10


# -----------------------------------------------------------------------
# Εξαίρεση που σηματοδοτεί ότι η ζωντανή αναζήτηση δεν είναι διαθέσιμη
# -----------------------------------------------------------------------
class LiveSearchUnavailable(Exception):
    """Raised when TAVILY_API_KEY isn't configured, or the search fails."""


# -----------------------------------------------------------------------
# Αναζήτηση στον ιστό μέσω του επίσημου API της Tavily για πρόσφατες πηγές
# -----------------------------------------------------------------------
def search_recent_legal_updates(query: str, max_results: int = 4) -> list:
    """Search for recent pages matching `query` via the Tavily API.

    Returns a list of {"title": ..., "url": ..., "snippet": ...}.
    Raises LiveSearchUnavailable if TAVILY_API_KEY isn't set, or the
    request fails (bad key, no internet, rate limit, timeout, etc.) —
    always as an explicit, readable error rather than a silent empty list.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise LiveSearchUnavailable(
            "TAVILY_API_KEY is not set. Get a free key at https://www.tavily.com "
            "(no credit card needed) and add it to your .env file."
        )

    try:
        response = requests.post(
            TAVILY_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.HTTPError as exc:
        # Ξεχωριστό, ρητό μήνυμα για λάθος/άκυρο κλειδί (401/403), ώστε να
        # μη μπερδεύεται με γενικό σφάλμα δικτύου
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            raise LiveSearchUnavailable(
                "Tavily rejected the API key (unauthorized). Check TAVILY_API_KEY in your .env file."
            ) from exc
        if status == 429:
            raise LiveSearchUnavailable(
                "Tavily rate limit reached (free tier). Try again later, or check your monthly credit usage."
            ) from exc
        raise LiveSearchUnavailable(f"Tavily request failed: {exc}") from exc
    except requests.exceptions.RequestException as exc:  # network/timeout errors
        raise LiveSearchUnavailable(f"Live search failed: {exc}") from exc

    # Μετατροπή των ακατέργαστων αποτελεσμάτων της Tavily σε ενιαία,
    # απλή μορφή — ίδια δομή με πριν, ώστε να μη χρειάζεται καμία αλλαγή
    # στον υπόλοιπο κώδικα (app.py) που καταναλώνει αυτή τη συνάρτηση
    out = []
    for r in payload.get("results", []):
        url = r.get("url")
        if not url:
            continue
        out.append({
            "title": r.get("title", "") or url,
            "url": url,
            "snippet": r.get("content", ""),
        })
    return out
