from pathlib import Path
import sys

import joblib
import pandas as pd

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from scorer.ml.features import extract_features


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "essay_score_model.joblib"


def build_feature_frame(question, essay, model):
    """Create a single-row DataFrame that matches the trained model columns."""
    extracted = extract_features(question, essay)

    feature_row = {
        "essay": essay,
        "semantic_similarity": extracted["semantic_similarity"],
        "word_count": extracted["word_count"],
        "sentence_count": extracted["sentence_count"],
        "average_sentence_length": extracted["average_sentence_length"],
        "avg_sentence_length": extracted["avg_sentence_length"],
        "paragraph_count": extracted["paragraph_count"],
        "explanation_keyword_count": extracted["explanation_keyword_count"],
        "explanation_count": extracted["explanation_count"],
        "vagueness_count": extracted["vagueness_count"],
        "repetition_score": extracted["repetition_score"],
        "grammar_error_count": extracted["grammar_error_count"],
    }

    feature_frame = pd.DataFrame([feature_row])

    required_columns = list(model.feature_names_in_)
    missing_columns = [column for column in required_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    feature_frame = feature_frame.loc[:, required_columns]
    return feature_frame


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    question = "Should computers benefit society?"
    essay = (
        "Computers help students learn and communicate, but they should be used responsibly. "
        "For example, they can help with research, writing, and collaboration."
    )

    feature_frame = build_feature_frame(question, essay, model)
    prediction = float(model.predict(feature_frame)[0])

    print(f"Model loaded from: {MODEL_PATH}")
    print("Feature columns match model.feature_names_in_: yes")
    print(f"Predicted score: {prediction:.2f}")


if __name__ == "__main__":
    main()