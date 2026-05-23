"""v2.14 Phase 2 — query-intent classifier for targeted HyDE bridging.

Deterministic heuristics. No LLM call, no network. Used by
`retrieve_hybrid_reranked(..., auto_intent_hyde=True)` to opt-in to
local-vLLM HyDE generation with an intent-specific HyDE system prompt.

Targets the omlx embedder's per-doc deficits documented in
`docs/QUALITY_SNAPSHOT_2026-05-22_v2.13_p1_omlx.md`:

  - German content (`ATZ_Elektronik`) -12.5pp R@1 vs Dashscope
  - Code-dense docs (`Python_Cookbook`, `IRJET`,
    `Hybrid_electric_vehicles`, `Greenhouse_Design`) -12pp R@1

Bridges those at QUERY TIME via a content-aware HyDE call rather
than maintaining parallel embedder collections (which would create
permanent architectural fragmentation per
`memory/feedback_fix_extraction_not_judge.md` "architectural entropy"
rule).

Returns one of:
  - "code"               — query asks about programming / code syntax
  - "minority_language"  — query is in a non-English language
  - None                 — default English / general intent

The classifier is intentionally cheap and conservative: a False
return means HyDE stays off (default v2.13 retrieval path), which
preserves the latency + cost profile for the ~90% of queries that
are general English. Only the targeted ~10% get the +1-2s HyDE
latency and the intent-specific prompt.
"""
from __future__ import annotations

import re


# Code keywords that strongly signal "asking about programming code"
# when present in a natural-language query.
_CODE_KEYWORDS = frozenset({
    # Programming-language verbs (strong signal — rare in English prose)
    "def", "class", "import", "lambda", "yield", "async", "await",
    # Code-domain nouns (medium signal — common in tech English too)
    "iterator", "generator", "decorator", "comprehension",
    "callable", "subclass", "metaclass", "constructor", "destructor",
    "kwargs", "args", "regex", "traceback", "exception",
    "namespace", "snippet", "syntax",
    # Code-context modifiers
    "method", "function", "module", "package", "library",
    "dictionary", "tuple",
    # Programming-language names (strong context signal)
    "python", "javascript", "typescript", "golang", "rust", "java",
    "cpp", "c++", "kotlin", "swift", "ruby", "perl", "haskell",
    # Tooling
    "pip", "npm", "yarn", "cargo", "venv", "conda",
    "pytest", "unittest", "fastapi", "django", "flask", "numpy",
    "pandas", "torch", "sklearn", "functools", "itertools", "asyncio",
})

# Symbol patterns that signal code: dunder names, arrow operators,
# parenthesized empty args, backtick-quoted identifiers, dotted module.attr.
_CODE_SYMBOLS_RE = re.compile(
    r"(__\w+__|->|=>|::|\(\)|\bself\b|`[^`]+`|\b[a-z_][\w]*\.[a-z_][\w]*)"
)

# Language-specific high-frequency stopwords + tokens NEVER seen in
# English. Two-tier model: a single STRONG signal (uniquely-non-English
# token like "welche", "qu'est-ce", "hoe", "wat") is enough; or 2+
# common-stopword hits (with overlap risk like "die" / "het" → require
# co-occurrence).
_LANG_STRONG_SIGNALS = frozenset({
    # German — never English
    "welche", "welcher", "welches", "wie", "was", "warum", "weshalb",
    "verbindungen", "kommunikation", "fahigkeit", "fahigkeiten",
    "erlauben", "verwendung", "moglich", "moglichkeit", "deutsche",
    "uber",
    # Dutch — never English (well, "wat" is a quasi-typo of "what" but
    # in isolation that's accepted noise)
    "hoe", "wat", "wanneer", "waarom", "welke", "welk",
    "nederlands", "nederland",
    # French — never English
    "quelle", "quel", "quoi", "comment", "pourquoi", "lorsque",
    "moyen", "moyens", "donnees", "donneur",
})

_LANG_STOPWORDS = {
    "de": frozenset({
        "und", "der", "die", "das", "ist", "ein", "eine", "mit",
        "von", "fur", "nicht", "auch", "sich", "auf", "den", "dem",
        "des", "kann", "es", "im", "zu", "zum", "zur",
    }),
    "nl": frozenset({
        "het", "een", "is", "van", "voor", "niet", "ook", "zich",
        "op", "bij", "naar", "uit", "kan", "moeten", "willen",
        "de", "te", "om",
    }),
    "fr": frozenset({
        "le", "la", "les", "et", "est", "une", "des", "ou", "pour",
        "ne", "pas", "aussi", "sur", "dans", "avec", "sans", "peut",
        "il", "elle", "nous", "vous",
    }),
}


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


def classify_intent(query: str) -> str | None:
    """Return the detected intent for `query`, or None for default.

    Order of checks (highest-precision-first):
      1. `code` — explicit code symbols OR 2+ code keywords in tokens
      2. `minority_language` — non-ASCII ratio > 5% (catches umlauts,
         accents) OR 2+ same-language stopwords
      3. None — default English / general

    Tokenization is `\\b[a-zA-Z_]\\w*\\b` (Latin-script word chars).
    """
    if not query or not query.strip():
        return None
    q = query.strip()
    q_lower = q.lower()

    # 1. Code intent
    if _CODE_SYMBOLS_RE.search(q):
        return "code"
    tokens = re.findall(r"\b[a-zA-Z_]\w*\b", q_lower)
    code_hits = sum(1 for t in tokens if t in _CODE_KEYWORDS)
    if code_hits >= 2:
        return "code"

    # 2. Minority-language intent — three independent signals,
    # any one suffices:
    #   (a) non-ASCII density > 3% (umlauts/accents in 8-12-word
    #       queries, e.g. "Welche Fähigkeiten brauche ich für diesen
    #       Beruf?" at 4.3%)
    #   (b) one+ STRONG-signal token (unique-non-English word like
    #       "welche", "kommunikation", "hoe", "quoi") — catches all-
    #       ASCII non-English queries (e.g. "Welche Verbindungen
    #       erlauben CAN Bus Kommunikation?")
    #   (c) 2+ common-stopword hits in the SAME language (the high-
    #       precision fallback for queries that lack strong signals
    #       but pack stopword density)
    if _non_ascii_ratio(q) > 0.03:
        return "minority_language"
    word_set = set(tokens)
    if word_set & _LANG_STRONG_SIGNALS:
        return "minority_language"
    for lang, stopwords in _LANG_STOPWORDS.items():
        if len(word_set & stopwords) >= 2:
            return "minority_language"

    return None
