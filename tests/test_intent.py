"""v2.14 Phase 2 — unit tests for the query-intent classifier.

Deterministic heuristics → these tests don't need any external
service. Pipeline-level integration (auto_intent_hyde dispatching
to HyDE with the right provider + intent kwarg) is tested in
`test_retrieval_pipeline.py` against mocked HyDE/qdrant.
"""
from mmrag_v2.retrieval.intent import classify_intent


# ── Default / no intent ──────────────────────────────────────────────


def test_empty_query_returns_none():
    assert classify_intent("") is None
    assert classify_intent("   ") is None


def test_general_english_query_returns_none():
    cases = [
        "What is the capital of France?",
        "How does the immune system work?",
        "Explain photosynthesis briefly",
        "Who wrote War and Peace?",
        "Recipe for sourdough bread",
        "The history of the Roman Empire",
    ]
    for q in cases:
        assert classify_intent(q) is None, f"general English misclassified: {q!r}"


# ── Code intent ──────────────────────────────────────────────────────


def test_code_intent_explicit_code_symbols():
    """Symbol patterns (dunder names, arrow operators, ()) → code."""
    cases = [
        "How do I implement __getitem__ in a custom class?",
        "When should I use type hints with ->?",
        "What's the difference between () and []?",
        "Explain `functools.reduce`",
    ]
    for q in cases:
        assert classify_intent(q) == "code", f"expected code: {q!r}"


def test_code_intent_two_or_more_keywords():
    """2+ code keywords in tokenized query → code."""
    cases = [
        "How to write a Python iterator with yield",
        "What is a decorator and how do I use it in a function definition",
        "Explain the difference between generator and iterator",
        "How does an async function with await work",
    ]
    for q in cases:
        assert classify_intent(q) == "code", f"expected code: {q!r}"


def test_code_intent_single_keyword_not_enough():
    """A single ambiguous keyword (e.g. 'class') in English prose → None."""
    cases = [
        "What time is the class scheduled for?",  # 'class' is a single keyword, no symbols
        "Tell me about the function of the liver.",  # 'function' once, no symbols
    ]
    for q in cases:
        result = classify_intent(q)
        assert result != "code", f"single-keyword false-positive: {q!r} → {result!r}"


# ── Minority-language intent ─────────────────────────────────────────


def test_german_query_via_umlauts():
    cases = [
        "Was ist die größte Stadt Deutschlands?",
        "Welche Fähigkeiten brauche ich für diesen Beruf?",
    ]
    for q in cases:
        assert classify_intent(q) == "minority_language", q


def test_german_query_via_stopwords():
    """Ascii-only German query (no umlauts) caught by stopword density."""
    q = "Was ist die richtige Antwort und wie kann ich das herausfinden"
    assert classify_intent(q) == "minority_language"


def test_french_query_via_accents():
    cases = [
        "Quelle est la différence entre ces méthodes?",
        "Où se trouve le bâtiment principal de l'université?",
    ]
    for q in cases:
        assert classify_intent(q) == "minority_language", q


def test_dutch_query_via_stopwords():
    """Ascii-only Dutch query caught by stopword density."""
    q = "Hoe kan ik het beste een fiets onderhouden van een Nederlands merk"
    assert classify_intent(q) == "minority_language"


def test_spanish_query_via_tilde():
    """5%+ non-ASCII (the ñ + tildes) → minority_language."""
    q = "¿Cuál es la diferencia entre estos métodos?"
    assert classify_intent(q) == "minority_language"


# ── Precedence: code beats minority-language ────────────────────────


def test_code_intent_takes_precedence_over_minority_language():
    """A non-English code query (e.g. German programming question with
    Python keywords) is classified as `code`, not `minority_language` —
    code intent is checked first and is the higher-precision signal."""
    q = "Wie schreibe ich eine Python iterator mit yield und __next__?"
    # Has __next__ (code symbol) + 'iterator' + 'yield' keywords.
    # The HyDE-side code prompt is more useful here than the language
    # prompt; the model will still write the answer in the question's
    # language because that's baked into the code prompt + the model's
    # general behavior.
    assert classify_intent(q) == "code"


# ── Tokenizer sanity ────────────────────────────────────────────────


def test_classifier_is_case_insensitive_on_keywords():
    """Code keywords match regardless of case in the query."""
    assert classify_intent("How do I write a CLASS with a DECORATOR") == "code"


def test_classifier_handles_punctuation():
    """Quoted/punctuated queries don't break tokenization."""
    assert classify_intent("Explain `lambda` and `yield` in Python.") == "code"
