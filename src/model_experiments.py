import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import the custom preprocessing function from the existing project structure
from src.preprocessing import clean_text

def run_experiments():
    """
    Loads data, evaluates models with fresh TF-IDF vectorizers, prints a comparison,
    and saves the best model and experiment summaries to disk.
    """
    # 1. Define Paths 
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
    RESULTS_PATH = MODEL_DIR / "experiment_results.json"
    METRICS_PATH = MODEL_DIR / "model_metrics.json"

    # 2. Check if dataset exists, print clean error if missing
    if not DATA_PATH.exists():
        print(f"[ERROR] Cleaned dataset not found at: {DATA_PATH}")
        print("Please ensure the preprocessing step has been run and the file exists.")
        sys.exit(1)

    # Ensure the models directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Load the dataset
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text', 'sentiment'])
    
    X = df['text']
    y = df['sentiment']

    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}\n")

    # 5. Define Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight="balanced"
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(
                max_iter=3000, 
                random_state=42, 
                class_weight="balanced"
            )
        ),
        "ComplementNB": ComplementNB()
    }

    # Tracking variables for the best model
    experiment_results = {}
    best_macro_f1 = -1.0
    best_model_name = ""
    best_pipeline = None
    best_metrics = {}

    # Define standard label order for evaluation
    label_order = ["negative", "neutral", "positive"]

    # Print Table Header
    print("-" * 85)
    print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Neutral F1':<10}")
    print("-" * 85)

    # 6. Train and Evaluate each model
    for model_name, classifier in models.items():
        
        # Create a fresh TF-IDF Vectorizer for each pipeline
        tfidf_vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=30000
        )

        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('classifier', classifier)
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Calculate metrics and cast to standard Python float for safe JSON serialization
        acc = float(accuracy_score(y_test, y_pred))
        mac_f1 = float(f1_score(y_test, y_pred, average='macro'))
        wgt_f1 = float(f1_score(y_test, y_pred, average='weighted'))
        
        # Get full classification report enforcing the standard label order
        report_dict = classification_report(
            y_test, y_pred, 
            labels=label_order, 
            output_dict=True, 
            zero_division=0
        )
        
        # Extract neutral class F1-score safely
        neutral_f1 = float(report_dict.get('neutral', {}).get('f1-score', 0.0))

        # Store results for this model
        model_metrics = {
            "accuracy": acc,
            "macro_f1": mac_f1,
            "weighted_f1": wgt_f1,
            "neutral_f1": neutral_f1,
            "classification_report": report_dict
        }
        experiment_results[model_name] = model_metrics

        # Print to table
        print(f"{model_name:<20} | {acc:<10.4f} | {mac_f1:<10.4f} | {wgt_f1:<12.4f} | {neutral_f1:<10.4f}")

        # Update best model tracking
        if mac_f1 > best_macro_f1:
            best_macro_f1 = mac_f1
            best_model_name = model_name
            best_pipeline = pipeline
            best_metrics = {
                "best_model": model_name,
                "metrics": model_metrics
            }

    print("-" * 85)
    print(f"\nBest Model: **{best_model_name}** with Macro F1: {best_macro_f1:.4f}\n")

    # 7. Save the best model
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"[SUCCESS] Best model pipeline saved to: {MODEL_PATH}")

    # 8. Save all experiment results to JSON
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"[SUCCESS] All experiment metrics saved to: {RESULTS_PATH}")

    # 9. Save best model summary to JSON
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, indent=4)
    print(f"[SUCCESS] Best model summary saved to: {METRICS_PATH}")


if __name__ == "__main__":
    run_experiments()