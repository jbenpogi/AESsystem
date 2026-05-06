import argparse
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from scorer.ml.features import extract_features

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "asap_clean.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "scorer" / "ml" / "artifacts" / "essay_score_model.joblib"

REQUIRED_COLUMNS = {"essay_set", "essay", "domain1_score", "normalized_score"}

NUMERIC_FEATURE_COLUMNS = [
    "semantic_similarity",
    "word_count",
    "sentence_count",
    "average_sentence_length",
    "paragraph_count",
    "explanation_keyword_count",
    "vagueness_count",
    "repetition_score",
    "grammar_error_count",
]

PROMPT_BY_ESSAY_SET = {
    1: "Do computers have a positive or negative effect on people?",
    2: "Describe a time when you felt strongly about injustice. What did you do?",
    3: "What is your opinion of using technology to help students learn?",
    4: "In your opinion, does the use of an honor code lead to greater integrity?",
    5: "Which is more important: seeking knowledge or money? Argue your position.",
    6: "Should cell phones be allowed in schools? Explain your position.",
    7: "What characteristics of a good reader or writer are most important? Rate the passage.",
    8: "Describe an interesting personality. Explain with two examples from the passage.",
}


def prompt_for_essay_set(essay_set):
    try:
        essay_set = int(essay_set)
    except (TypeError, ValueError):
        return ""

    return PROMPT_BY_ESSAY_SET.get(essay_set, "")


def extract_training_features(essay_set, essay):
    prompt = prompt_for_essay_set(essay_set)
    features = extract_features(prompt, essay, essay_set=essay_set)
    features["essay"] = essay or ""
    return features


def prepare_dataset(dataset_path):
    data_frame = pd.read_csv(dataset_path)

    missing_columns = REQUIRED_COLUMNS.difference(data_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data_frame = data_frame[["essay_set", "essay", "normalized_score"]].copy()
    data_frame["essay"] = data_frame["essay"].fillna("").astype(str)
    data_frame["essay_set"] = pd.to_numeric(data_frame["essay_set"], errors="coerce")
    data_frame["essay_set"] = data_frame["essay_set"].fillna(0).astype(int)
    data_frame["normalized_score"] = pd.to_numeric(data_frame["normalized_score"], errors="coerce")
    data_frame = data_frame.dropna(subset=["normalized_score"])
    data_frame["normalized_score"] = data_frame["normalized_score"].clip(lower=0.0, upper=100.0)

    rows = [
        extract_training_features(essay_set, essay)
        for essay_set, essay in zip(data_frame["essay_set"], data_frame["essay"])
    ]
    feature_frame = pd.DataFrame(rows)
    target = data_frame["normalized_score"].astype(float)

    return feature_frame, target


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "essay_tfidf",
                TfidfVectorizer(ngram_range=(1, 2), max_features=8000),
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


def train_model(dataset_path, output_path):
    features, target = prepare_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model_pipeline = build_pipeline()
    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, output_path)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "rows": int(len(features)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a RandomForest essay scoring model on ASAP cleaned data."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to ASAP cleaned CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to output joblib model file.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    metrics = train_model(dataset_path, output_path)

    print("Training completed.")
    print(f"Rows used: {metrics['rows']}")
    print(f"Model saved to: {output_path}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()