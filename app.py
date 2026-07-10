import streamlit as st

from src.sentiment_engine import compare_sentiment_models


st.set_page_config(
    page_title="Valtrion — Feedback Intelligence Platform",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .main-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #6b7280;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }

        .section-label {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 700;
        }

        .stTextArea textarea {
            font-size: 1.02rem;
            line-height: 1.5;
        }

        .footer-text {
            color: #6b7280;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_label(value: str) -> str:
    """Convert internal labels into display-friendly text."""
    if value is None:
        return "N/A"

    return str(value).replace("_", " ").title()


def format_percent(value) -> str:
    """Format float probability as percentage."""
    if value is None:
        return "N/A"

    return f"{float(value):.2%}"


def get_sentiment_display(sentiment: str) -> str:
    """Return clean sentiment display label."""
    sentiment = str(sentiment).lower()

    if sentiment == "positive":
        return "Positive"
    if sentiment == "negative":
        return "Negative"
    if sentiment == "neutral":
        return "Neutral"

    return format_label(sentiment)

def generate_final_verdict(result: dict) -> str:
    """
    Generate an executive summary from sentiment, reliability,
    intensity, and aspect analysis.
    """
    sentiment = format_label(result.get("recommended_sentiment"))
    reliability = result.get("reliability", {})
    intensity = result.get("intensity", {})
    aspect_analysis = result.get("aspect_analysis", {})

    confidence_level = format_label(reliability.get("confidence_level"))
    review_status = format_label(reliability.get("review_status"))
    sentiment_intensity = format_label(intensity.get("sentiment_intensity"))

    mixed_feedback = aspect_analysis.get("mixed_feedback", False)
    aspects = aspect_analysis.get("aspects", [])

    if mixed_feedback and aspects:
        aspect_parts = []

        for aspect in aspects:
            aspect_name = aspect.get("aspect", "Unknown Aspect")
            aspect_sentiment = format_label(aspect.get("sentiment"))
            aspect_parts.append(f"{aspect_name} is {aspect_sentiment}")

        aspect_summary = ", while ".join(aspect_parts)

        return (
            f"Mixed feedback detected. {aspect_summary}. "
            f"The overall transformer prediction leans {sentiment} with "
            f"{confidence_level.lower()} confidence and {sentiment_intensity.lower()} intensity. "
            f"Review status: {review_status}."
        )

    return (
        f"Valtrion classifies this feedback as {sentiment} with "
        f"{sentiment_intensity.lower()} intensity. The prediction has "
        f"{confidence_level.lower()} confidence. Review status: {review_status}."
    )


def render_probability_bars(probabilities: dict) -> None:
    """Render probability values as progress bars."""
    ordered_labels = ["negative", "neutral", "positive"]

    for label in ordered_labels:
        probability = float(probabilities.get(label, 0.0))
        st.write(f"**{label.capitalize()}** — {probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))


def analyze_feedback(text: str):
    """Run backend analysis with error handling."""
    with st.spinner("Running Valtrion analysis..."):
        return compare_sentiment_models(text.strip())

def render_sentiment_badge(sentiment: str) -> str:
    """
    Return a clean display label for aspect sentiment.
    """
    sentiment = str(sentiment).lower()

    if sentiment == "positive":
        return "Positive"
    if sentiment == "negative":
        return "Negative"
    if sentiment == "neutral":
        return "Neutral"

    return format_label(sentiment)

if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# Header
st.markdown('<div class="main-title">Valtrion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Feedback Intelligence Platform</div>',
    unsafe_allow_html=True,
)

st.write(
    "A transformer-powered feedback intelligence platform for sentiment "
    "classification, reliability scoring, intensity analysis, and customer insight."
)

st.divider()


# Input section
st.markdown('<div class="section-label">Feedback Input</div>', unsafe_allow_html=True)

sample_texts = {
    "Strong Positive": "The service was excellent and I am very happy.",
    "Strong Negative": "The airline service was very bad.",
    "Neutral": "The product was received yesterday and the package contained all listed items.",
    "Mixed Feedback": "The product quality is good, but the delivery was late.",
}

sample_cols = st.columns(len(sample_texts))

for index, (label, sample) in enumerate(sample_texts.items()):
    with sample_cols[index]:
        if st.button(label, use_container_width=True):
            st.session_state.feedback_text = sample
            st.session_state.last_result = None
            st.rerun()

user_text = st.text_area(
    "Enter customer feedback",
    key="feedback_text",
    height=150,
    placeholder="Paste customer feedback, review text, or support message here...",
)

analyze_clicked = st.button(
    "Analyze Feedback",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    if not st.session_state.feedback_text.strip():
        st.warning("Please enter feedback text before analyzing.")
    else:
        try:
            st.session_state.last_result = analyze_feedback(
                st.session_state.feedback_text
            )
        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.stop()


result = st.session_state.last_result


# Results
if result:
    st.divider()
    st.subheader("Analysis Results")

    recommended_sentiment = result["recommended_sentiment"]
    reliability = result["reliability"]
    intensity = result["intensity"]

    metric_1, metric_2, metric_3, metric_4 = st.columns([1.2, 1.2, 1.1, 1.6])

    with metric_1:
        st.metric(
            "Recommended Sentiment",
            get_sentiment_display(recommended_sentiment),
        )

    with metric_2:
        st.metric(
            "Sentiment Intensity",
            format_label(intensity["sentiment_intensity"]),
        )

    with metric_3:
        st.metric(
            "Confidence Level",
            format_label(reliability["confidence_level"]),
        )

    with metric_4:
        review_status = format_label(reliability["review_status"])

        if "Auto Classified With Model Disagreement" in review_status:
            short_status = "Auto Classified"
        elif review_status == "Review Optional":
            short_status = "Optional"
        elif review_status == "Needs Manual Review":
            short_status = "Manual Review"
        else:
            short_status = review_status

        st.metric(
            "Review Status",
            short_status,
        )

        if short_status != review_status:
            st.caption(review_status)
        final_verdict = generate_final_verdict(result)

    with st.container(border=True):
        st.markdown("#### Final Verdict")
        st.write(final_verdict)

    st.divider()

    # Reliability and intensity
    reliability_col, intensity_col = st.columns(2)

    with reliability_col:
        with st.container(border=True):
            st.markdown("#### Reliability Assessment")
            st.write(
                f"**Confidence Level:** "
                f"{format_label(reliability['confidence_level'])}"
            )
            st.write(
                f"**Decision Margin:** "
                f"{format_percent(reliability['decision_margin'])}"
            )
            st.write(
                f"**Review Status:** "
                f"{format_label(reliability['review_status'])}"
            )
            st.progress(
                min(max(float(reliability["decision_margin"]), 0.0), 1.0)
            )

            with st.expander("Reliability explanation"):
                st.write(reliability["reliability_reason"])

    with intensity_col:
        with st.container(border=True):
            st.markdown("#### Sentiment Intensity")
            st.write(
                f"**Intensity:** "
                f"{format_label(intensity['sentiment_intensity'])}"
            )
            st.write(
                f"**Probability Spread:** "
                f"{format_percent(intensity['probability_spread'])}"
            )
            st.progress(
                min(max(float(intensity["probability_spread"]), 0.0), 1.0)
            )

            with st.expander("Intensity explanation"):
                st.write(intensity["intensity_reason"])

    st.divider()
        # Aspect intelligence
    aspect_analysis = result.get("aspect_analysis", {})

    if aspect_analysis:
        st.divider()
        st.subheader("Aspect Intelligence")

        aspect_count = aspect_analysis.get("aspect_count", 0)
        mixed_feedback = aspect_analysis.get("mixed_feedback", False)

        aspect_metric_1, aspect_metric_2 = st.columns(2)

        with aspect_metric_1:
            st.metric("Detected Aspects", aspect_count)

        with aspect_metric_2:
            st.metric(
                "Mixed Feedback",
                "Yes" if mixed_feedback else "No",
            )

        aspects = aspect_analysis.get("aspects", [])

        if aspects:
            aspect_columns = st.columns(min(len(aspects), 3))

            for index, aspect in enumerate(aspects):
                with aspect_columns[index % len(aspect_columns)]:
                    with st.container(border=True):
                        st.markdown(f"#### {aspect.get('aspect', 'Unknown Aspect')}")
                        st.write(f"**Sentiment:** {render_sentiment_badge(aspect.get('sentiment'))}")
                        st.write(f"**Confidence:** {format_percent(aspect.get('confidence'))}")

                        with st.expander("Evidence"):
                            st.write(aspect.get("evidence", "No evidence available."))

                        with st.expander("Aspect probabilities"):
                            render_probability_bars(aspect.get("probabilities", {}))
    # Model comparison
    st.subheader("Model Comparison")

    classical = result["classical_model"]
    transformer = result["transformer_model"]

    classical_col, transformer_col = st.columns(2)

    with classical_col:
        with st.container(border=True):
            st.markdown("#### Classical Baseline")
            st.caption(classical["model_type"])
            st.write(f"**Sentiment:** {format_label(classical['sentiment'])}")
            st.write(f"**Confidence:** {format_percent(classical['confidence'])}")
            render_probability_bars(classical.get("probabilities", {}))

    with transformer_col:
        with st.container(border=True):
            st.markdown("#### Transformer Engine")
            st.caption(transformer["model_type"])
            if transformer.get("model_name"):
                st.caption(transformer["model_name"])

            st.write(f"**Sentiment:** {format_label(transformer['sentiment'])}")
            st.write(f"**Confidence:** {format_percent(transformer['confidence'])}")
            render_probability_bars(transformer.get("probabilities", {}))

    if result.get("model_agreement"):
        st.success("Classical baseline and transformer engine agree.")
    else:
        st.warning(
            "Classical baseline and transformer engine disagree. "
            "Valtrion uses the transformer recommendation as the final output."
        )

    with st.expander("Raw analysis JSON"):
        st.json(result)


st.divider()
st.markdown(
    '<div class="footer-text">Valtrion — Feedback Intelligence Platform</div>',
    unsafe_allow_html=True,
)