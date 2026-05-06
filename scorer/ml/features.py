import language_tool_python
import re

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
tool = language_tool_python.LanguageTool('en-US')

ASAP_PROMPTS = {
    1: "Do computers have a positive or negative effect on people?",
    2: "Describe a time when you felt strongly about injustice. What did you do?",
    3: "What is your opinion of using technology to help students learn?",
    4: "In your opinion, does the use of an honor code lead to greater integrity?",
    5: "Which is more important: seeking knowledge or money? Argue your position.",
    6: "Should cell phones be allowed in schools? Explain your position.",
    7: "What characteristics of a good reader or writer are most important? Rate the passage.",
    8: "Describe an interesting personality. Explain with two examples from the passage.",
}

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


def resolve_prompt(question=None, essay_set=None):
    if question and str(question).strip():
        return str(question).strip()

    try:
        essay_set = int(essay_set) if essay_set is not None else None
    except (TypeError, ValueError):
        essay_set = None

    if essay_set in ASAP_PROMPTS:
        return ASAP_PROMPTS[essay_set]

    return ""


def compute_semantic_similarity(question, essay):
    if not question or not essay:
        return 0.0

    embeddings = model.encode([question, essay], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity[0][0])


def count_explanation_keywords(text):
    lowered = (text or "").lower()
    return sum(1 for phrase in EXPLANATION_KEYWORDS if phrase in lowered)


def count_vagueness(text):
    lowered = (text or "").lower()
    return sum(1 for phrase in VAGUE_PHRASES if phrase in lowered)


def compute_repetition_score(text):
    words = re.findall(r"\b[a-z']+\b", (text or "").lower())
    if not words:
        return 0.0

    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(words)


def count_words(text):
    return len(re.findall(r'\b\w+\b', text or ""))


def count_sentences(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text or "") if s.strip()]
    return len(sentences)


def count_paragraphs(text):
    paragraphs = [p for p in (text or "").split("\n") if p.strip()]
    return len(paragraphs)


def count_grammar_errors(text):
    words = (text or "").split()
    if not words:
        return 0.0

    snippet = " ".join(words[:250])
    try:
        return float(len(tool.check(snippet)))
    except Exception:
        return 0.0


def extract_features(question, essay, essay_set=None):
    essay = essay or ""
    prompt = resolve_prompt(question, essay_set)
    word_count = count_words(essay)
    sentence_count = count_sentences(essay)
    average_sentence_length = word_count / sentence_count if sentence_count else 0.0
    paragraph_count = count_paragraphs(essay)
    explanation_keyword_count = count_explanation_keywords(essay)
    vagueness_count = count_vagueness(essay)
    repetition_score = compute_repetition_score(essay)
    grammar_error_count = count_grammar_errors(essay)
    semantic_similarity = compute_semantic_similarity(prompt, essay)

    return {
        "semantic_similarity": semantic_similarity,
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "average_sentence_length": float(average_sentence_length),
        "avg_sentence_length": float(average_sentence_length),
        "grammar_error_count": float(grammar_error_count),
        "paragraph_count": float(paragraph_count),
        "explanation_keyword_count": float(explanation_keyword_count),
        "explanation_count": float(explanation_keyword_count),
        "vagueness_count": float(vagueness_count),
        "repetition_score": float(repetition_score),
    }