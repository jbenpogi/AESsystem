from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import language_tool_python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
ASAP_DATASET_PATH = ROOT_DIR / "data" / "asap_clean.csv"
DEFAULT_DATASET_PATH = ASAP_DATASET_PATH
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "essay_component_models.joblib"

CONTENT_TARGET = "content_relevance_score"
GRAMMAR_TARGET = "grammar_mechanics_score"
FORMATTING_TARGET = "formatting_effort_score"

TARGET_COLUMNS = [CONTENT_TARGET, GRAMMAR_TARGET, FORMATTING_TARGET]

BASE_COMPONENT_MAX = {
    CONTENT_TARGET: 50.0,
    GRAMMAR_TARGET: 30.0,
    FORMATTING_TARGET: 20.0,
}
 
GRAMMAR_CHECK_MAX_WORDS = 250

NUMERIC_FEATURE_COLUMNS = [
    "word_count",
    "sentence_count",
    "grammar_error_count",
]


def _word_count(text):
    return len(re.findall(r"\b\w+\b", text or ""))


@lru_cache(maxsize=1)
def _get_grammar_tool():
    return language_tool_python.LanguageTool("en-US")


@lru_cache(maxsize=50000)
def _grammar_error_count_cached(snippet):
    if not snippet:
        return 0.0

    try:
        return float(len(_get_grammar_tool().check(snippet)))
    except Exception:
        return 0.0


def _grammar_error_count(text):
    tokens = (text or "").split()
    if not tokens:
        return 0.0

    # Limit text length for practical training runtime while preserving signal.
    snippet = " ".join(tokens[:GRAMMAR_CHECK_MAX_WORDS])
    return _grammar_error_count_cached(snippet)


def _build_feature_row(question, essay):
    essay = essay or ""

    word_count = _word_count(essay)
    sentences = [segment.strip() for segment in re.split(r"[.!?]", essay) if segment.strip()]
    sentence_count = len(sentences)
    grammar_error_count = _grammar_error_count(essay)

    return {
        "essay": essay,
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "grammar_error_count": float(grammar_error_count),
    }


def build_feature_frame(data_frame):
    rows = [_build_feature_row("", essay) for essay in data_frame["essay"]]
    return pd.DataFrame(rows)


def normalize_training_dataframe(data_frame):
    required_columns = {"essay", *TARGET_COLUMNS}
    if required_columns.issubset(data_frame.columns):
        normalized = data_frame[["essay", *TARGET_COLUMNS]].copy()
        normalized["essay"] = normalized["essay"].fillna("").astype(str)
        for column in TARGET_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna(subset=TARGET_COLUMNS)
        return normalized

    asap_columns = {"essay", "normalized_score"}
    if asap_columns.issubset(data_frame.columns):
        normalized_score = pd.to_numeric(data_frame["normalized_score"], errors="coerce")
        normalized_score = normalized_score.clip(lower=0.0, upper=100.0)

        normalized = pd.DataFrame(
            {
                "essay": data_frame["essay"].fillna("").astype(str),
                CONTENT_TARGET: normalized_score * 0.5,
                GRAMMAR_TARGET: normalized_score * 0.3,
                FORMATTING_TARGET: normalized_score * 0.2,
            }
        )

        normalized[CONTENT_TARGET] = normalized[CONTENT_TARGET].clip(
            lower=0.0,
            upper=BASE_COMPONENT_MAX[CONTENT_TARGET],
        )
        normalized[GRAMMAR_TARGET] = normalized[GRAMMAR_TARGET].clip(
            lower=0.0,
            upper=BASE_COMPONENT_MAX[GRAMMAR_TARGET],
        )
        normalized[FORMATTING_TARGET] = normalized[FORMATTING_TARGET].clip(
            lower=0.0,
            upper=BASE_COMPONENT_MAX[FORMATTING_TARGET],
        )
        normalized = normalized.dropna(subset=TARGET_COLUMNS)
        return normalized

    expected = "essay, content_relevance_score, grammar_mechanics_score, formatting_effort_score"
    alternative = "essay, normalized_score (optional essay_set for prompt labels)"
    raise ValueError(
        "Dataset schema is not supported. Expected either "
        f"[{expected}] or [{alternative}]."
    )


def _make_regression_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "essay_tfidf",
                TfidfVectorizer(ngram_range=(1, 2), max_features=6000),
                "essay",
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def train_and_save_models(dataset_path=DEFAULT_DATASET_PATH, model_path=MODEL_PATH):
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw_data_frame = pd.read_csv(dataset_path)
    data_frame = normalize_training_dataframe(raw_data_frame)

    feature_frame = build_feature_frame(data_frame)

    train_indices, test_indices = train_test_split(
        list(range(len(feature_frame))),
        test_size=0.2,
        random_state=42,
    )

    x_train = feature_frame.iloc[train_indices]
    x_test = feature_frame.iloc[test_indices]

    models = {}
    metrics = {}

    for column in TARGET_COLUMNS:
        y_train = data_frame.iloc[train_indices][column]
        y_test = data_frame.iloc[test_indices][column]

        model = _make_regression_pipeline()
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        models[column] = model
        metrics[column] = {
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        }

    artifact = {
        "version": 2,
        "model_family": "RandomForestRegressor",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "models": models,
        "metrics": metrics,
        "base_component_max": BASE_COMPONENT_MAX,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    _cached_model_bundle.cache_clear()

    return metrics


@lru_cache(maxsize=1)
def _cached_model_bundle(model_path_string):
    model_path = Path(model_path_string)
    if not model_path.exists():
        return None

    try:
        return joblib.load(model_path)
    except Exception:
        return None


def load_model_bundle(model_path=MODEL_PATH):
    return _cached_model_bundle(str(Path(model_path).resolve()))


def predict_component_scores(question, essay, model_path=MODEL_PATH):
    artifact = load_model_bundle(model_path)
    if not artifact:
        return None

    models = artifact.get("models", {})
    if not models:
        return None

    input_frame = pd.DataFrame([_build_feature_row(question, essay)])
    raw_max = artifact.get("base_component_max", BASE_COMPONENT_MAX)

    predictions = {}
    for target_column in TARGET_COLUMNS:
        model = models.get(target_column)
        if model is None:
            return None

        predicted = float(model.predict(input_frame)[0])
        ceiling = float(raw_max.get(target_column, 100.0))
        predictions[target_column] = max(0.0, min(predicted, ceiling))

    return predictions