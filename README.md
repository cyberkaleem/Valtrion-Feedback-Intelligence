
# Valtrion — Feedback Intelligence Platform

```text
__     __    _ _        _             
\ \   / /_ _| | |_ _ __(_) ___  _ __  
 \ \ / / _` | | __| '__| |/ _ \| '_ \ 
  \ V / (_| | | |_| |  | | (_) | | | |
   \_/ \__,_|_|\__|_|  |_|\___/|_| |_|
```

**Feedback Intelligence Platform for Real-Time Customer Feedback Analysis**

Valtrion is a real-time NLP web application that analyzes customer feedback using a transformer-powered sentiment engine and explainable business intelligence layers.

The platform classifies feedback sentiment, compares classical and transformer models, evaluates prediction reliability, detects sentiment intensity, identifies feedback aspects, analyzes emotion signals, assigns operational priority, and audits outputs using an intelligence reliability guard.

---

## Project Overview

Valtrion demonstrates a practical customer feedback intelligence workflow, from text preprocessing and model inference to reliability scoring and Streamlit-based result visualization.

The application allows a user to enter customer feedback, run real-time analysis, and view structured outputs including sentiment, confidence, aspect-level interpretation, emotion signal, priority level, recommended action, and reliability guard status.

The project is designed as a practical feedback intelligence system rather than only a basic sentiment classification experiment.

---

## Key Features

- Real-time customer feedback analysis
- Transformer-based sentiment classification
- Classical TF-IDF + Logistic Regression baseline
- Classical vs transformer model comparison
- Sentiment confidence score
- Sentiment probability distribution
- Decision margin calculation
- Reliability assessment
- Review status classification
- Sentiment intensity analysis
- Aspect-based feedback intelligence
- Mixed feedback detection
- Emotion intelligence
- Priority / urgency classification
- Recommended support action
- Intelligence Reliability Guard
- Guard-aware Streamlit UI banner
- Final verdict generation
- Dark-themed Streamlit interface
- Modular Python project structure
- Git-based project version control

---

## Technology Stack

| Layer | Technologies |
|---|---|
| User Interface | Streamlit |
| Programming Language | Python |
| Classical ML | Scikit-learn, TF-IDF, Logistic Regression |
| Transformer NLP | Hugging Face Transformers |
| Deep Learning Backend | PyTorch |
| Data Handling | Pandas |
| Model Persistence | Joblib |
| Utilities | NumPy, SciPy |
| Version Control | Git, GitHub |

---

## Model Information

| Item | Details |
|---|---|
| Classical Baseline | TF-IDF + Logistic Regression |
| Transformer Model | CardiffNLP RoBERTa Sentiment Transformer |
| Transformer Source | Git, GitHub |

---

## Model Information

| Item | Details |
|---|---|
| Classical Baseline | TF-IDF + Logistic Regression |
| Transformer | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Output Labels | Positive, Neutral, Negative |
| Recommended Engine | Transformer Sentiment Engine |
| Baseline Purpose | Model comparison and technical evaluation |
| UI Framework | Streamlit |

The classical model is stored in:

```text
models/sentiment_pipeline.joblib
```

Model metrics are stored in:

```text
models/model_metrics.json
```

---

## System Architecture

```text
User Feedback
    |
    v
Streamlit Interface
    |
    v
Input Validation
    |
    v
Classical Sentiment Model
TF-IDF + Logistic Regression
    |
    v
Transformer Sentiment Engine
RoBERTa-based Sentiment Model
    |
    v
Model Comparison Layer
    |
    v
Reliability Assessment Layer
    |
    v
Sentiment Intensity Layer
    |
    v
Aspect Intelligence Layer
    |
    v
Emotion Intelligence Layer
    |
    v
Priority Intelligence Layer
    |
    v
Intelligence Reliability Guard
    |
    v
Final Verdict + UI Result Cards
```

---

## Analysis Pipeline

```text
Customer Feedback Input
    |
    v
Validate Text
    |
    v
Run Classical Baseline Prediction
    |
    v
Run Transformer Sentiment Prediction
    |
    v
Compare Model Outputs
    |
    v
Calculate Reliability and Decision Margin
    |
    v
Analyze Sentiment Intensity
    |
    v
Detect Business Aspects
    |
    v
Detect Emotion Signals
    |
    v
Calculate Priority Level
    |
    v
Audit Output Using Intelligence Guard
    |
    v
Display Structured Feedback Intelligence
```

---

## Project Structure

```text
Feedback-Intelligence-system/
│
├── app.py
├── README.md
├── report.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── transformer_sentiment.py
│   ├── reliability_engine.py
│   ├── intensity_engine.py
│   ├── aspect_analyzer.py
│   ├── emotion_detector.py
│   ├── priority_engine.py
│   ├── intelligence_guard.py
│   └── sentiment_engine.py
│
├── models/
│   ├── .gitkeep
│   ├── sentiment_pipeline.joblib
│   ├── model_metrics.json
│   ├── experiment_results.json
│   └── balanced_experiment_results.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│
└── .streamlit/
    └── config.toml
```

---

## Core Modules

| Module | Purpose |
|---|---|
| `preprocessing.py` | Text cleaning and preprocessing |
| `train_model.py` | Classical baseline model training |
| `evaluate_model.py` | Model evaluation workflow |
| `predict.py` | Classical model prediction |
| `transformer_sentiment.py` | Transformer sentiment inference |
| `reliability_engine.py` | Confidence and review status calculation |
| `intensity_engine.py` | Sentiment intensity analysis |
| `aspect_analyzer.py` | Aspect-based feedback interpretation |
| `emotion_detector.py` | Emotion signal detection |
| `priority_engine.py` | Priority / urgency classification |
| `intelligence_guard.py` | Output reliability auditing |
| `sentiment_engine.py` | Main orchestration layer |
| `app.py` | Streamlit user interface |

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Feedback-Intelligence-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```

### 6. Open in Browser

```text
http://localhost:8501
```

---

## Usage

1. Start the Streamlit application.
2. Enter customer feedback in the text input area.
3. Click **Analyze Feedback**.
4. View the recommended sentiment.
5. Review confidence level and reliability status.
6. Check sentiment intensity.
7. Inspect aspect intelligence.
8. Review emotion intelligence.
9. Check priority level and recommended action.
10. View the Intelligence Reliability Guard result.
11. Compare classical baseline and transformer outputs.

---

## Sample Inputs

### Positive Feedback

```text
The service was excellent and I am very happy.
```

Expected behavior:

```text
Sentiment: Positive
Emotion: Satisfaction
Priority: Low
Guard Status: Stable
```

---

### Mixed Feedback

```text
The product was awesome but I was not satisfied with the service.
```

Expected behavior:

```text
Product Quality: Positive
Customer Support: Negative
Overall Sentiment: Negative
Emotion: Disappointment
Priority: Medium
```

---

### Payment Issue

```text
My payment failed but money was deducted.
```

Expected behavior:

```text
Sentiment: Negative
Aspect: Payment and Refund
Emotion: Concern
Priority: High
Recommended Action: Escalate to billing/payment support
```

---

### Critical Security Issue

```text
My account was hacked and there was an unauthorized transaction.
```

Expected behavior:

```text
Sentiment: Negative
Emotion: Concern
Priority: Critical
Recommended Action: Immediate escalation
```

---

## Supported Sentiment Labels

| Label |
|---|
| Positive |
| Neutral |
| Negative |

---

## Supported Aspect Categories

| Aspect |
|---|
| Product Quality |
| Delivery |
| Customer Support |
| Payment and Refund |
| App or Website Experience |
| Price and Value |
| Packaging |
| General Experience |

---

## Supported Emotion Categories

| Emotion |
|---|
| Satisfaction |
| Disappointment |
| Frustration |
| Anger |
| Concern |
| Neutral |

---

## Supported Priority Levels

| Priority Level | Meaning |
|---|---|
| Low | Normal or low-risk feedback |
| Medium | Negative but non-urgent feedback |
| High | Important operational issue requiring attention |
| Critical | Security, fraud, legal, safety, or severe account-risk issue |

---

## Intelligence Reliability Guard

The Intelligence Reliability Guard audits the output of the business intelligence layers.

It checks for:

- Emotion and sentiment conflicts
- High priority without strong evidence
- Critical priority without critical-risk cues
- Low-confidence aspect predictions
- Core sentiment outputs requiring review

Guard status values:

| Guard Status | Meaning |
|---|---|
| Stable | No major reliability issues detected |
| Review Optional | Output is usable, but one layer has uncertainty |
| Review Required | One or more outputs may be inconsistent or risky |

---

## Current Status

| Area | Status |
|---|---|
| Real-time Streamlit UI | Complete |
| Classical Baseline Model | Complete |
| Transformer Sentiment Engine | Complete |
| Model Comparison | Complete |
| Reliability Assessment | Complete |
| Sentiment Intensity | Complete |
| Aspect Intelligence | Complete |
| Emotion Intelligence | Complete |
| Priority Intelligence | Complete |
| Intelligence Reliability Guard | Complete |
| Local Testing | Complete |
| Analytics Dashboard | Future Upgrade |
| Batch CSV Analysis | Future Upgrade |
| Public Deployment | Future Upgrade |

---

## Model Development Notes

A classical TF-IDF + Logistic Regression model was trained as a baseline. It achieved useful performance but showed limitations with contextual language and phrase-level meaning.

For example:

```text
the airline service was very bad
```

The classical baseline can struggle with this type of sentence because it relies on learned token weights. The transformer model handles this better because it can interpret contextual meaning more effectively.

Therefore, Valtrion uses the transformer model as the recommended sentiment engine while keeping the classical model for comparison and technical evaluation.

---

## Limitations

Valtrion uses a hybrid architecture. The core sentiment classifier is transformer-based, while several business intelligence layers are explainable rule-based systems.

Current limitations:

- Rule-based layers may not understand every possible phrase.
- Emotion and priority detection partly depend on predefined cues.
- The current version focuses on real-time single-feedback analysis.
- Batch CSV analysis is not included in the current release.
- Analytics dashboard is planned as a future upgrade.
- The application is currently intended for local execution.
- The system is a decision-support tool, not a fully autonomous customer support replacement.

The Intelligence Reliability Guard helps reduce overconfident outputs by flagging uncertain or potentially inconsistent results.

---

## Future Improvements

- Analytics dashboard for aggregated feedback trends
- CSV upload and batch feedback analysis
- Dedicated recommendation engine
- Public deployment using Streamlit Cloud or a similar platform
- Database integration
- User authentication
- Zero-shot semantic classifier for stronger emotion, aspect, and priority understanding
- API integration with live customer feedback sources
- Exportable feedback reports
- Historical trend analysis
- Docker support
- CI/CD pipeline

---

## Author

**KALEEMULLAH S**

Computer Science Student  
Interests: Artificial Intelligence, Natural Language Processing, Cybersecurity, Software Engineering

GitHub: [cyberkaleem](https://github.com/cyberkaleem)

---

## License

This project is intended for educational and portfolio purposes.

A formal open-source license can be added in a future release.
````
