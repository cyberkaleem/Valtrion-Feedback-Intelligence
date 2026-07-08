import sys
from pathlib import Path

import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sentiment_pipeline.joblib"


def debug_prediction(text: str, top_n: int = 20):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(MODEL_PATH)

    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    X = tfidf.transform([text])

    prediction = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_
        print("\nPrediction:", prediction)
        print("Probabilities:")
        for label, prob in zip(classes, probabilities):
            print(f"  {label}: {prob:.4f}")

    feature_names = np.array(tfidf.get_feature_names_out())
    nonzero_indices = X.nonzero()[1]

    print("\nFeatures detected in this input:")
    detected_features = feature_names[nonzero_indices]
    detected_values = X[0, nonzero_indices].toarray().flatten()

    for feature, value in sorted(
        zip(detected_features, detected_values),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {feature:<25} tfidf={value:.4f}")

    print("\nClass-wise contribution scores:")
    for class_index, class_label in enumerate(classifier.classes_):
        coef = classifier.coef_[class_index]
        contributions = []

        for idx in nonzero_indices:
            feature = feature_names[idx]
            tfidf_value = X[0, idx]
            weight = coef[idx]
            contribution = tfidf_value * weight
            contributions.append((feature, contribution, weight, tfidf_value))

        contributions = sorted(
            contributions,
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        total_score = sum(item[1] for item in contributions) + classifier.intercept_[class_index]

        print(f"\nClass: {class_label}")
        print(f"Intercept: {classifier.intercept_[class_index]:.4f}")
        print(f"Approx score: {total_score:.4f}")
        print("Top contributions:")

        for feature, contribution, weight, tfidf_value in contributions[:top_n]:
            print(
                f"  {feature:<25} contribution={contribution:.4f} "
                f"weight={weight:.4f} tfidf={tfidf_value:.4f}"
            )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = input("Enter text to debug: ")

    debug_prediction(input_text)