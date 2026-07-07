import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Tweets.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_dataset.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "test_data.csv"

MODEL_PATH = ROOT_DIR / "models" / "sentiment_pipeline.joblib"
METRICS_PATH = ROOT_DIR / "models" / "model_metrics.json"


def load_and_prepare_dataset() -> pd.DataFrame:
    """
    Load raw dataset and convert it into a standard format:
    text, sentiment
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Place Tweets.csv inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    required_columns = {"text", "airline_sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Expected columns 'text' and 'airline_sentiment' in Tweets.csv. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[["text", "airline_sentiment"]].copy()
    df.rename(columns={"airline_sentiment": "sentiment"}, inplace=True)

    df.dropna(subset=["text", "sentiment"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)

    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]

    df["text"] = df["text"].astype(str)
    df["cleaned_text"] = df["text"].apply(clean_text)

    df = df[df["cleaned_text"].str.len() > 2]

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df


def train_model() -> None:
    """
    Train TF-IDF + Logistic Regression sentiment classifier.
    """
    df = load_and_prepare_dataset()

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "class_distribution": df["sentiment"].value_counts().to_dict(),
        "total_samples": int(len(df)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    test_df = pd.DataFrame(
        {
            "text": X_test,
            "actual_sentiment": y_test,
            "predicted_sentiment": y_pred,
        }
    )
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("Model training completed successfully.")
    print(f"Total samples used: {len(df)}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Test predictions saved to: {TEST_DATA_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")


if __name__ == "__main__":
    train_model()