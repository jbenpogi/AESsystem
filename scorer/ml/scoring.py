from functools import lru_cache
from collections import Counter
from pathlib import Path

import html
import re

import joblib
import pandas as pd
import language_tool_python
from .features import extract_features

tool = language_tool_python.LanguageTool('en-US')

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "scorer" / "ml" / "artifacts" / "essay_score_model.joblib"

EXPLANATION_KEYWORDS = [
    "because",
    "for example",
    "such as",
    "therefore",
    "as a result",
    "for instance",
]

VAGUE_PHRASES = [
    "something",
    "many things",
    "in some way",
    "many ways",
    "people say",
    "it is a topic",
    "a lot",
]


ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


LOREM_TOKENS = {
    "lorem",
    "ipsum",
    "dolor",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "eiusmod",
    "tempor",
    "incididunt",
    "labore",
    "dolore",
    "magna",
    "aliqua",
    "minim",
    "veniam",
    "quis",
    "nostrud",
    "exercitation",
    "ullamco",
    "laboris",
    "nisi",
    "aliquip",
    "commodo",
    "consequat",
    "duis",
    "aute",
    "irure",
    "reprehenderit",
    "voluptate",
    "velit",
    "esse",
    "cillum",
    "fugiat",
    "nulla",
    "pariatur",
    "excepteur",
    "sint",
    "occaecat",
    "cupidatat",
    "proident",
    "culpa",
    "officia",
    "deserunt",
    "mollit",
    "anim",
    "laborum",
}


def _placeholder_text_metrics(text):
    lowered = (text or "").lower()
    tokens = re.findall(r"\b[a-z']+\b", lowered)
    if not tokens:
        return {
            "token_count": 0,
            "stopword_ratio": 0.0,
            "lorem_ratio": 0.0,
            "lorem_phrase_count": 0,
        }

    token_count = len(tokens)
    stop_count = sum(1 for token in tokens if token in ENGLISH_STOPWORDS)
    lorem_count = sum(1 for token in tokens if token in LOREM_TOKENS)
    return {
        "token_count": token_count,
        "stopword_ratio": stop_count / token_count,
        "lorem_ratio": lorem_count / token_count,
        "lorem_phrase_count": lowered.count("lorem ipsum"),
    }


def _is_placeholder_filler_text(text):
    """
    Detect common placeholder / filler text (e.g., "lorem ipsum") so it cannot earn
    a high content score. This is intentionally conservative to avoid false positives.
    """
    metrics = _placeholder_text_metrics(text)

    token_count = metrics["token_count"]
    if token_count == 0:
        return False, metrics

    lorem_ratio = metrics["lorem_ratio"]
    stopword_ratio = metrics["stopword_ratio"]
    lorem_phrase_count = metrics["lorem_phrase_count"]

    # Strong lorem-ipsum signal: lots of classic tokens or obvious phrase + non-English ratio.
    if lorem_ratio >= 0.18 and token_count >= 20:
        return True, metrics

    if lorem_phrase_count >= 1:
        # If the phrase appears but the text looks like normal English (high stopwords),
        # assume it's being referenced rather than used as filler.
        if stopword_ratio <= 0.12:
            return True, metrics

    # Generic non-English / gibberish: extremely low English stopword usage in a long response.
    if token_count >= 80 and stopword_ratio <= 0.02:
        return True, metrics

    return False, metrics

GRAMMAR_CHECK_MAX_WORDS = 250


@lru_cache(maxsize=1)
def load_ml_model():
    """Load the trained ASAP model from disk and cache it in memory."""
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def _count_words(text):
    return len(re.findall(r'\b\w+\b', text or ""))


def _count_sentences(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text or "") if s.strip()]
    return len(sentences)


def _count_paragraphs(text):
    paragraphs = [p for p in (text or "").split("\n") if p.strip()]
    return len(paragraphs)


def _count_explanation_keywords(text):
    lowered = (text or "").lower()
    return sum(1 for phrase in EXPLANATION_KEYWORDS if phrase in lowered)


def _count_vagueness(text):
    lowered = (text or "").lower()
    return sum(1 for phrase in VAGUE_PHRASES if phrase in lowered)


def _legacy_repetition_score(text):
    """Legacy repetition ratio used during model training (keep stable for ML features)."""
    words = re.findall(r"\b[a-z']+\b", (text or "").lower())
    if not words:
        return 0.0

    counts = Counter(words)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(words)


def _ngram_repeat_ratio(tokens, n=3, min_content_tokens=1):
    """Return a small 0..1 ratio of repeated n-gram occurrences."""
    if n <= 1:
        return 0.0

    token_count = len(tokens)
    if token_count < (n * 3):
        return 0.0

    ngrams = []
    for i in range(token_count - n + 1):
        gram = tokens[i : i + n]
        content_count = sum(
            1 for t in gram if (t not in ENGLISH_STOPWORDS and len(t) > 2)
        )
        if content_count < min_content_tokens:
            continue
        ngrams.append(tuple(gram))

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    return repeated_occurrences / len(ngrams)


def _repetition_score(text):
    """
    Repetition score used for quality penalties.

    Unlike the legacy metric, this focuses on *content-word* repetition (ignoring
    stopwords) so normal reuse of common words doesn't dominate the score.
    Returns a value in [0, 1].
    """
    tokens = re.findall(r"\b[a-z']+\b", (text or "").lower())
    if not tokens:
        return 0.0

    content_tokens = [
        t for t in tokens
        if t not in ENGLISH_STOPWORDS and len(t) > 2
    ]
    if len(content_tokens) < 20:
        content_tokens = [t for t in tokens if len(t) > 2]

    if not content_tokens:
        return 0.0

    counts = Counter(content_tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    base_ratio = repeated / len(content_tokens)

    # Boost for repeated short phrases (captures loops like "X is important").
    trigram_repeat = _ngram_repeat_ratio(tokens, n=3, min_content_tokens=1)
    score = base_ratio + (0.35 * trigram_repeat)
    return max(0.0, min(round(score, 4), 1.0))


@lru_cache(maxsize=50000)
def _cached_grammar_error_count(snippet):
    if not snippet:
        return 0.0

    try:
        return float(len(tool.check(snippet)))
    except Exception:
        return 0.0


def _grammar_error_count(text):
    words = (text or "").split()
    if not words:
        return 0.0

    snippet = " ".join(words[:GRAMMAR_CHECK_MAX_WORDS])
    return _cached_grammar_error_count(snippet)


def _build_ml_feature_frame(question, essay, model=None):
    """Build a single-row DataFrame and align it to the trained model columns."""
    extracted = extract_features(question, essay)

    feature_row = {
        "essay": essay or "",
        "semantic_similarity": float(extracted.get("semantic_similarity", 0)),
        "word_count": float(extracted.get("word_count", 0)),
        "sentence_count": float(extracted.get("sentence_count", 0)),
        "average_sentence_length": float(extracted.get("avg_sentence_length", 0)),
        "paragraph_count": float(extracted.get("paragraph_count", 0)),
        "explanation_keyword_count": float(extracted.get("explanation_count", 0)),
        "vagueness_count": float(_count_vagueness(essay)),
        "repetition_score": float(_legacy_repetition_score(essay)),
        "grammar_error_count": float(extracted.get("grammar_error_count", 0)),
    }

    feature_frame = pd.DataFrame([feature_row])

    if model is not None and hasattr(model, "feature_names_in_"):
        ordered_columns = list(model.feature_names_in_)

        for column_name in ordered_columns:
            if column_name not in feature_frame.columns:
                feature_frame[column_name] = 0.0

        feature_frame = feature_frame.loc[:, ordered_columns]

    return feature_frame


def predict_ml_score(question, essay):
    """Predict a score from 0 to 100 using the saved RandomForest model."""
    model = load_ml_model()
    if model is None:
        return 0.0

    features = _build_ml_feature_frame(question, essay, model)

    try:
        prediction = float(model.predict(features)[0])
    except Exception:
        return 0.0

    return round(max(0.0, min(prediction, 100.0)), 2)


def scale_component_score(raw_score, raw_max, custom_points):
    if raw_max <= 0 or custom_points <= 0:
        return 0.0

    ratio = max(0.0, min(raw_score, raw_max)) / raw_max
    return round(ratio * custom_points, 2)


# -----------------------------
# HIGHLIGHT ERRORS
# -----------------------------
def highlight_errors(text, matches):
    highlighted_text = text

    for match in sorted(matches, key=lambda m: m.offset, reverse=True):
        start = match.offset
        error_length = getattr(match, "error_length", None)
        if error_length is None:
            error_length = getattr(match, "errorLength", 0)
        end = start + error_length

        if start < 0 or end > len(highlighted_text) or start >= end:
            continue

        wrong_part = highlighted_text[start:end]
        safe_wrong = html.escape(wrong_part)
        safe_msg = html.escape(match.message)

        replacement = (
            f'<span style="text-decoration: underline; '
            f'text-decoration-color: red; color: red;" '
            f'title="{safe_msg}">{safe_wrong}</span>'
        )

        highlighted_text = (
            highlighted_text[:start] +
            replacement +
            highlighted_text[end:]
        )

    return highlighted_text.replace("\n", "<br>")


# -----------------------------
# GRAMMAR SCORE
# -----------------------------
def grammar_score(essay, grammar_points=30):
    matches = tool.check(essay)
    score = _grammar_score_from_matches(matches, grammar_points)

    return round(score, 2), matches


def _grammar_score_from_matches(matches, grammar_points=30):
    errors = len(matches)
    deduction = errors * 1.5
    return max(0, grammar_points - deduction)


# -----------------------------
# PENALTY CALCULATIONS FOR CONTENT QUALITY
# -----------------------------
def _calculate_vagueness_penalty(essay, max_penalty=5):
    """
    Calculate penalty (0-5 points) for vague writing.
    Higher vagueness_count = higher penalty.
    """
    vagueness_count = _count_vagueness(essay)
    if vagueness_count == 0:
        return 0.0
    # Scale penalty: 1-2 vague phrases = 2pts, 3+ = 5pts
    penalty = min(2 + (vagueness_count - 1) * 1.5, max_penalty)
    return penalty


def _calculate_repetition_penalty(essay, max_penalty=5):
    """
    Calculate penalty (0-5 points) for repetitive writing.
    Higher repetition_score = higher penalty.
    """
    rep_score = _repetition_score(essay)
    if rep_score < 0.2:
        return 0.0
    # Scale penalty only after repetition is clearly excessive.
    penalty = min(1.5 + (rep_score - 0.2) * 20, max_penalty)
    return penalty


# -----------------------------
# SEMANTIC RELEVANCE PART (ML-BASED CONTENT SCORE)
# -----------------------------
def content_relevance_score(
    question,
    essay,
    content_points=50,
    min_words=250,
):
    """
    Compute content score from the ML model and apply strict quality caps so
    shallow / repetitive / vague essays score lower.
    """
    is_placeholder, _ = _is_placeholder_filler_text(essay)
    if is_placeholder:
        return 0.0

    ml_score = predict_ml_score(question, essay)

    # Normalize and stretch ML score
    ml_score = (ml_score - 50) * 1.5 + 50

    # Clamp to 0–100
    ml_score = max(0.0, min(100.0, ml_score))

    # Stricter baseline mapping from ML score (0–100) to content_points.
    content_score = ((ml_score - 20.0) / 80.0) * content_points

    lowered = (essay or "").lower()

    repetition_score = _repetition_score(essay)
    vagueness_count = sum(lowered.count(phrase) for phrase in VAGUE_PHRASES)
    explanation_keyword_count = sum(lowered.count(phrase) for phrase in EXPLANATION_KEYWORDS)

    word_count = _count_words(essay)
    sentence_count = _count_sentences(essay)
    average_sentence_length = (word_count / sentence_count) if sentence_count else 0.0

    # Strong quality scaling factor
    quality_factor = 1.0
    if repetition_score > 0.2:
        quality_factor -= 0.15
    if vagueness_count > 10:
        quality_factor -= 0.15
    if explanation_keyword_count < 3:
        quality_factor -= 0.2
    if average_sentence_length < 10:
        quality_factor -= 0.1

    quality_factor = max(0.4, min(1.0, quality_factor))
    content_score = content_score * quality_factor
    
    # Clamp to 0-content_points range
    content_score = max(0.0, min(content_score, content_points))
    
    return round(content_score, 2)


# -----------------------------
# FORMATTING AND EFFORT SCORE
# -----------------------------
def formatting_effort_score(essay, formatting_points=20, min_words=250, max_words=1000):
    word_count = len(re.findall(r'\b\w+\b', essay))
    paragraphs = [p for p in essay.split("\n") if p.strip()]

    score = formatting_points

    if word_count < min_words:
        score *= 0.5
    elif word_count > max_words:
        score *= 0.8

    if len(paragraphs) < 2:
        score *= 0.9

    return round(score, 2)


# -----------------------------
# FINAL TOTAL SCORE
# -----------------------------
def total_score(
    question,
    essay,
    content_points=50,
    grammar_points=30,
    formatting_points=20,
    min_words=250,
    max_words=1000,
    debug=False,
):
    """
    Compute total essay score using calibrated component-based rubric:
    - ML model with calibrated scoring: ((ml_score - 40) / 60) * content_points
    - Apply penalties for vagueness and repetition (0-5 each)
    - Grammar errors → grammar score (0–grammar_points)
    - Format/effort → formatting score (0–formatting_points)
    - Final = content + grammar + formatting (0–100)
    
    Returns: (final_score, grammar_score, content_score, formatting_score, highlighted_essay)
    """
    is_placeholder, placeholder_metrics = _is_placeholder_filler_text(essay)

    matches = tool.check(essay)

    # Component 1: ML-based content score with calibration and penalties
    if is_placeholder:
        content_score = 0.0
    else:
        content_score = content_relevance_score(
            question=question,
            essay=essay,
            content_points=content_points,
            min_words=min_words,
        )

    if content_points > 0:
        content_ratio = content_score / content_points
    else:
        # If content has no weight, don't downscale other components.
        content_ratio = 1.0

    content_ratio = max(0.0, min(float(content_ratio), 1.0))
    component_scale = 0.5 + 0.5 * content_ratio
    
    # Component 2: Grammar score
    grammar_raw = _grammar_score_from_matches(matches, grammar_points)
    grammar_score_final = max(0.0, min(grammar_raw, grammar_points))
    grammar_score_final = grammar_score_final * component_scale
    grammar_score_final = max(0.0, min(grammar_score_final, grammar_points))
    
    # Component 3: Formatting score
    if is_placeholder:
        formatting_raw = 0.0
        formatting_score_final = 0.0
    else:
        formatting_raw = formatting_effort_score(essay, formatting_points, min_words, max_words)
        formatting_score_final = max(0.0, min(formatting_raw, formatting_points))

    formatting_score_final = formatting_score_final * component_scale
    formatting_score_final = max(0.0, min(formatting_score_final, formatting_points))
    
    # Final score: sum of all components
    final_score = content_score + grammar_score_final + formatting_score_final
    final_score = max(0.0, min(final_score, 100.0))
    
    # Highlight grammar errors
    highlighted = highlight_errors(essay, matches)
    
    # Optional debug prints
    if debug:
        if is_placeholder:
            ml_score = 0.0
            vagueness_penalty = 0.0
            repetition_penalty = 0.0
        else:
            ml_score = predict_ml_score(question, essay)
            vagueness_penalty = _calculate_vagueness_penalty(essay)
            repetition_penalty = _calculate_repetition_penalty(essay)

        if is_placeholder:
            print(
                "[DEBUG] Placeholder/filler detected; forcing content and formatting scores to 0. "
                f"(lorem_ratio={placeholder_metrics['lorem_ratio']:.3f}, "
                f"stopword_ratio={placeholder_metrics['stopword_ratio']:.3f}, "
                f"lorem_phrase_count={placeholder_metrics['lorem_phrase_count']})"
            )
        print(f"[DEBUG] ML Score (0–100): {ml_score:.2f}")
        print(f"[DEBUG] Vagueness Penalty: -{vagueness_penalty:.2f}")
        print(f"[DEBUG] Repetition Penalty: -{repetition_penalty:.2f}")
        print(f"[DEBUG] Content Score (0–{content_points}): {content_score:.2f}")
        print(f"[DEBUG] Grammar Score (0–{grammar_points}): {grammar_score_final:.2f}")
        print(f"[DEBUG] Formatting Score (0–{formatting_points}): {formatting_score_final:.2f}")
        print(f"[DEBUG] Final Score (0–100): {final_score:.2f}")
    
    return (
        round(final_score, 2),
        round(grammar_score_final, 2),
        round(content_score, 2),
        round(formatting_score_final, 2),
        highlighted,
    )