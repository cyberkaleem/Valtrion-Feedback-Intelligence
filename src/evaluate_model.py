from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "sentiment_pipeline.joblib"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "test_data.csv"

REPORT_PATH = ROOT_DIR / "outputs" / "classification_report.txt"
CONFUSION_MATRIX_PATH = ROOT_DIR / "outputs" / "confusion_matrix.png"


def evaluate_model() -> None:
    """
    Evaluate saved sentiment model using the saved test dataset.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train_model")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError("Test data not found. Run: python -m src.train_model")

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df["text"]
    y_test = test_df["actual_sentiment"]

    y_pred = model.predict(X_test)

    labels = ["negative", "neutral", "positive"]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, colorbar=False)
    plt.title("Sentiment Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    print("Model evaluation completed successfully.")
    print(f"Classification report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    evaluate_model()