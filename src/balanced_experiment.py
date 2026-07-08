import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import the custom preprocessing function from the existing project structure
from src.preprocessing import clean_text

def create_pipeline():
    """
    Creates a fresh instance of the TF-IDF and Logistic Regression pipeline.
    """
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=30000
    )
    
    classifier = LogisticRegression(
        max_iter=1000, 
        class_weight="balanced", 
        random_state=42
    )
    
    return Pipeline([
        ('tfidf', tfidf),
        ('classifier', classifier)
    ])

def run_balanced_experiment():
    """
    Loads dataset, creates an untouched test set, trains on imbalanced vs balanced 
    (downsampled) training data, compares results, and saves the best model.
    """
    # 1. Define Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
    RESULTS_PATH = MODEL_DIR / "balanced_experiment_results.json"
    METRICS_PATH = MODEL_DIR / "model_metrics.json"

    # Ensure the models directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Check if dataset exists
    if not DATA_PATH.exists():
        print(f"[ERROR] Cleaned dataset not found at: {DATA_PATH}")
        sys.exit(1)

    # 3. Load the dataset
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text', 'sentiment'])
    
    X = df['text']
    y = df['sentiment']

    # 4. Split data (Keep test set untouched and realistic)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Original Training samples: {len(X_train)}")
    print(f"Untouched Testing samples: {len(X_test)}\n")

    # 5. Balance the training set by downsampling
    train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    print("Original training class distribution:")
    print(train_df["sentiment"].value_counts())
    print()
    
    # Find the size of the smallest class in the training set
    min_class_size = train_df['sentiment'].value_counts().min()
    print(f"Smallest class size in training data: {min_class_size}")
    
    # Downsample all classes to match the smallest class
    balanced_train_df = (
    train_df.groupby("sentiment", group_keys=False)
    .sample(n=min_class_size, random_state=42)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
    print("Balanced training class distribution:")
    print(balanced_train_df["sentiment"].value_counts())
    print()
    
    X_train_bal = balanced_train_df['text']
    y_train_bal = balanced_train_df['sentiment']
    
    print(f"Balanced Training samples: {len(X_train_bal)}\n")

    # 6. Define the two experimental setups
    experiments = {
        "Baseline Model (Imbalanced)": (X_train, y_train),
        "Balanced Model (Downsampled)": (X_train_bal, y_train_bal)
    }

    # Tracking variables for best model
    experiment_results = {}
    best_macro_f1 = -1.0
    best_model_name = ""
    best_pipeline = None
    best_metrics = {}

    label_order = ["negative", "neutral", "positive"]

    # Print Table Header
    print("-" * 105)
    print(f"{'Model Name':<30} | {'Accuracy':<8} | {'Macro F1':<8} | {'Weighted F1':<11} | {'Neg F1':<8} | {'Neu F1':<8} | {'Pos F1':<8}")
    print("-" * 105)

    # 7. Train and Evaluate each setup on the untouched test set
    for model_name, (X_tr, y_tr) in experiments.items():
        
        # Create a fresh pipeline
        pipeline = create_pipeline()

        # Train the model
        pipeline.fit(X_tr, y_tr)

        # Make predictions on the untouched test set
        y_pred = pipeline.predict(X_test)

        # Calculate metrics and cast to standard float for JSON serialization
        acc = float(accuracy_score(y_test, y_pred))
        mac_f1 = float(f1_score(y_test, y_pred, average='macro'))
        wgt_f1 = float(f1_score(y_test, y_pred, average='weighted'))
        
        # Get classification report
        report_dict = classification_report(
            y_test, y_pred, 
            labels=label_order, 
            output_dict=True, 
            zero_division=0
        )
        
        # Extract individual class F1-scores securely
        neg_f1 = float(report_dict.get('negative', {}).get('f1-score', 0.0))
        neu_f1 = float(report_dict.get('neutral', {}).get('f1-score', 0.0))
        pos_f1 = float(report_dict.get('positive', {}).get('f1-score', 0.0))

        # 8. Store results
        model_metrics = {
            "accuracy": acc,
            "macro_f1": mac_f1,
            "weighted_f1": wgt_f1,
            "negative_f1": neg_f1,
            "neutral_f1": neu_f1,
            "positive_f1": pos_f1,
            "classification_report": report_dict
        }
        experiment_results[model_name] = model_metrics

        # 9. Print to comparison table
        print(f"{model_name:<30} | {acc:<8.4f} | {mac_f1:<8.4f} | {wgt_f1:<11.4f} | {neg_f1:<8.4f} | {neu_f1:<8.4f} | {pos_f1:<8.4f}")

        # Update best model tracking
        if mac_f1 > best_macro_f1:
            best_macro_f1 = mac_f1
            best_model_name = model_name
            best_pipeline = pipeline
            best_metrics = {
                "best_model": model_name,
                "metrics": model_metrics
            }

    print("-" * 105)
    print(f"\nBest Model: **{best_model_name}** with Macro F1: {best_macro_f1:.4f}\n")

    # 10. Save the best model
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"[SUCCESS] Best model pipeline saved to: {MODEL_PATH}")

    # 11. Save all experiment results to JSON
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"[SUCCESS] Experiment results saved to: {RESULTS_PATH}")

    # 12. Save best model summary to JSON
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, indent=4)
    print(f"[SUCCESS] Best model summary saved to: {METRICS_PATH}")

if __name__ == "__main__":
    run_balanced_experiment()