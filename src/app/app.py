from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from openrouter_client import ask_llm, get_default_model
from rag.simple_rag import build_qa_chain

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mental Health Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
DATA_CLEAN_DIR = DATA_DIR / "clean"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

CLASSICAL_REPORTS_DIR = REPORTS_DIR / "tables" / "classical"
TRANSFORMER_REPORTS_DIR = REPORTS_DIR / "transformers"
CLINICAL_TABLES_DIR = REPORTS_DIR / "tables" / "clinical"

DEFAULT_CLEAN_DATA_PATH = DATA_CLEAN_DIR / "mental_health_detection_clean.csv"

FINAL_TEST_METRICS_PATH = CLASSICAL_REPORTS_DIR / "final_test_metrics.csv"
NESTED_CV_SUMMARY_PATH = CLASSICAL_REPORTS_DIR / "nested_cv_summary.csv"
NORMAL_CV_SUMMARY_PATH = CLASSICAL_REPORTS_DIR / "normal_cv_summary.csv"
GLOBAL_CLINICAL_REVIEW_PATH = CLINICAL_TABLES_DIR / "global_comparison_for_clinical_review.csv"

MODEL_CANDIDATES = {
    "LinearSVC Balanced": [
        MODELS_DIR / "best_classical_model.joblib",
        PROJECT_ROOT / "src" / "model" / "best_classical_model.joblib",
    ],
    "Hybrid SVC": [
        MODELS_DIR / "hybrid_svc_model.joblib",
    ],
    "MentalBERT": [
        MODELS_DIR / "mentalbert_pipeline.joblib",
        MODELS_DIR / "mental_bert_pipeline.joblib",
        MODELS_DIR / "mentalbert_model.joblib",
    ],
    "BERT Base": [
        MODELS_DIR / "bert_base_pipeline.joblib",
        MODELS_DIR / "bert_pipeline.joblib",
        MODELS_DIR / "bert_model.joblib",
    ],
}

CLASS_LABELS = [
    "ADHD",
    "Anxiety",
    "Autism",
    "Bipolar",
    "BPD",
    "Depression",
    "Schizophrenia",
]


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# ============================================================
# CUSTOM CSS
# ============================================================

def load_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34,211,238,0.08), transparent 25%),
                radial-gradient(circle at top right, rgba(167,139,250,0.08), transparent 20%),
                linear-gradient(135deg, #0b1220 0%, #111827 45%, #0f172a 100%);
            color: #e5e7eb;
        }

        .block-container {
            padding-top: 1.15rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1450px;
        }

        header, footer, #MainMenu {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .main-title {
            font-size: 2.45rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.15rem;
            letter-spacing: -0.03em;
        }

        .subtitle {
            font-size: 1rem;
            color: #94a3b8;
            margin-bottom: 1.6rem;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.90));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.28);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 1.75rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.45rem;
        }

        .hero-text {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.7;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 22px;
            padding: 22px 20px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.22);
            min-height: 152px;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.5rem;
            word-break: break-word;
        }

        .metric-delta {
            font-size: 0.92rem;
            font-weight: 700;
        }

        .accent-cyan { color: #22d3ee; }
        .accent-gold { color: #fbbf24; }
        .accent-green { color: #34d399; }
        .accent-purple { color: #a78bfa; }
        .accent-red { color: #fb7185; }

        .section-box {
            background: rgba(15,23,42,0.76);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 22px;
            padding: 24px;
            margin-top: 1rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.18);
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.8rem;
        }

        .section-text {
            color: #cbd5e1;
            line-height: 1.7;
            font-size: 0.97rem;
        }

        .small-badge {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            color: #e2e8f0;
            background: rgba(255,255,255,0.07);
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .result-box {
            background: linear-gradient(135deg, rgba(17,24,39,0.98), rgba(15,23,42,0.96));
            border: 1px solid rgba(34,211,238,0.18);
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 12px 26px rgba(0,0,0,0.22);
        }

        .result-label {
            color: #94a3b8;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .result-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        .divider-line {
            height: 1px;
            border: none;
            background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.12), rgba(255,255,255,0.02));
            margin: 1.25rem 0;
        }

        .history-card {
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 0.8rem;
        }

        .history-title {
            color: #f8fafc;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .history-meta {
            color: #94a3b8;
            font-size: 0.88rem;
            margin-bottom: 0.45rem;
        }

        .history-text {
            color: #cbd5e1;
            font-size: 0.94rem;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UTILITIES
# ============================================================

def metric_card(
    label: str,
    value: str,
    delta_text: str,
    accent_class: str = "accent-cyan",
) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-delta {accent_class}">{escape(delta_text)}</div>
    </div>
    """


def render_header() -> None:
    st.markdown(
        '<div class="main-title">🧠 Mental Health Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Clinical-style NLP dashboard for mental health text analysis, model monitoring, prediction review and deployment-ready demo.</div>',
        unsafe_allow_html=True,
    )


def find_existing_model_path(model_name: str) -> Optional[Path]:
    candidate_paths = MODEL_CANDIDATES.get(model_name, [])
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_dataset_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "total_texts": None,
        "num_classes": len(CLASS_LABELS),
        "class_names": CLASS_LABELS,
        "dataset_loaded": False,
        "dataset_path": None,
    }

    if DEFAULT_CLEAN_DATA_PATH.exists():
        try:
            df = pd.read_csv(DEFAULT_CLEAN_DATA_PATH)
            info["dataset_loaded"] = True
            info["dataset_path"] = str(DEFAULT_CLEAN_DATA_PATH)
            info["total_texts"] = len(df)

            if "category" in df.columns:
                unique_classes = sorted(df["category"].dropna().astype(str).unique().tolist())
                if unique_classes:
                    info["class_names"] = unique_classes
                    info["num_classes"] = len(unique_classes)
        except Exception:
            pass

    return info


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def safe_get_first_value(df: Optional[pd.DataFrame], candidate_cols: List[str]) -> Optional[Any]:
    if df is None or df.empty:
        return None
    for col in candidate_cols:
        if col in df.columns:
            return df[col].iloc[0]
    return None


def format_metric(value: Optional[Any], decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


@st.cache_data(show_spinner=False)
def load_monitoring_artifacts() -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "final_test_df": load_csv_if_exists(FINAL_TEST_METRICS_PATH),
        "nested_cv_df": load_csv_if_exists(NESTED_CV_SUMMARY_PATH),
        "normal_cv_df": load_csv_if_exists(NORMAL_CV_SUMMARY_PATH),
        "global_review_df": load_csv_if_exists(GLOBAL_CLINICAL_REVIEW_PATH),
    }


@st.cache_resource(show_spinner=False)
def load_joblib_model(model_name: str) -> Tuple[Optional[Any], Optional[Path], Optional[str]]:
    model_path = find_existing_model_path(model_name)

    if model_path is None:
        return None, None, f"No local file found for {model_name}. Running in demo mode."

    try:
        model = joblib.load(model_path)
        return model, model_path, None
    except Exception as exc:
        return None, model_path, f"Model found but could not be loaded: {exc}"


def fake_demo_prediction(text: str) -> Tuple[str, float, pd.DataFrame]:
    text_lower = text.lower()

    heuristics = [
        ("Schizophrenia", ["voices", "watching me", "paranoid", "they are after me", "hallucination"]),
        ("Depression", ["hopeless", "empty", "worthless", "sad", "don't want to live"]),
        ("Anxiety", ["panic", "nervous", "can't breathe", "worry", "anxious"]),
        ("ADHD", ["can't focus", "distracted", "concentrate", "restless", "forget"]),
        ("Bipolar", ["extremely energetic", "no sleep", "unstoppable", "racing thoughts"]),
        ("Autism", ["overstimulated", "social cues", "sensory", "routine"]),
        ("BPD", ["abandoned", "intense emotions", "empty inside", "unstable relationships"]),
    ]

    predicted_label = "Anxiety"
    confidence = 0.62

    for label, keywords in heuristics:
        if any(keyword in text_lower for keyword in keywords):
            predicted_label = label
            confidence = 0.87
            break

    rows = []
    base = 0.03
    for label in CLASS_LABELS:
        score = base
        if label == predicted_label:
            score = confidence
        rows.append({"Class": label, "Probability": score})

    prob_df = pd.DataFrame(rows)
    prob_df["Probability"] = prob_df["Probability"] / prob_df["Probability"].sum()
    prob_df["Probability"] = prob_df["Probability"].round(4)
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    final_confidence = float(prob_df.loc[0, "Probability"])
    final_label = str(prob_df.loc[0, "Class"])

    return final_label, final_confidence, prob_df


def predict_with_model(
    model: Any,
    text: str,
) -> Tuple[str, Optional[float], pd.DataFrame]:
    prediction = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        class_names = list(model.classes_) if hasattr(model, "classes_") else CLASS_LABELS
        prob_df = pd.DataFrame(
            {
                "Class": class_names,
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False).reset_index(drop=True)
        confidence = float(prob_df.iloc[0]["Probability"])
        return str(prediction), confidence, prob_df

    prob_df = pd.DataFrame(
        {
            "Class": [str(prediction)],
            "Probability": [1.0],
        }
    )
    return str(prediction), None, prob_df


def save_prediction_to_history(
    text: str,
    model_name: str,
    predicted_label: str,
    confidence: Optional[float],
    mode: str,
) -> None:
    st.session_state.history.insert(
        0,
        {
            "text": text,
            "model": model_name,
            "label": predicted_label,
            "confidence": confidence,
            "mode": mode,
        },
    )
    st.session_state.history = st.session_state.history[:8]


def render_probability_table(prob_df: pd.DataFrame) -> None:
    display_df = prob_df.copy()
    display_df["Probability"] = (display_df["Probability"] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_sample_text_buttons() -> str:
    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Quick Test Samples</div>
            <div class="section-text">
                Click one of the examples below to quickly test the interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    sample_1 = "I feel like people are watching me all the time, and sometimes I hear voices even when no one is around."
    sample_2 = "I have been feeling hopeless, exhausted, and empty for weeks, and I struggle to get out of bed."
    sample_3 = "My thoughts are racing, I barely sleep, and I feel like I can do anything right now."

    selected_sample = ""

    with col1:
        if st.button("Load Schizophrenia-like sample", use_container_width=True):
            selected_sample = sample_1
    with col2:
        if st.button("Load Depression-like sample", use_container_width=True):
            selected_sample = sample_2
    with col3:
        if st.button("Load Bipolar-like sample", use_container_width=True):
            selected_sample = sample_3

    return selected_sample


def answer_with_openrouter(user_prompt: str) -> str:
    system_prompt = (
        "You are a careful assistant for a mental health NLP project dashboard. "
        "You help explain the project, metrics, model choices, deployment logic, and ethical framing. "
        "Do not present outputs as medical diagnosis. Be concise, clear, and professional."
    )

    return ask_llm(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )


# ============================================================
# SIDEBAR
# ============================================================

def build_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.4rem 0 1rem 0;">
                <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc;">Mental Health</div>
                <div style="color: #94a3b8; font-size: 0.92rem;">NLP Clinical Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=["Overview", "Predictions", "Monitoring", "Chat", "History", "About"],
            icons=["grid-fill", "activity", "bar-chart-line-fill", "chat-dots-fill", "clock-history", "info-circle-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#67e8f9", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "6px 0px",
                    "padding": "12px 14px",
                    "border-radius": "12px",
                    "color": "#cbd5e1",
                    "--hover-color": "rgba(255,255,255,0.06)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(90deg, rgba(34,211,238,0.18), rgba(59,130,246,0.18))",
                    "color": "#ffffff",
                    "font-weight": "700",
                },
            },
        )

        st.markdown("---")
        st.markdown("**System Status**")
        st.caption("Dashboard shell ready")
        st.caption("Prediction pipeline ready")
        st.caption("Demo mode fallback enabled")
        st.caption(f"OpenRouter model: {get_default_model()}")

        if st.button("Clear prediction history", use_container_width=True):
            st.session_state.history = []
            st.success("History cleared.")

    return selected


# ============================================================
# PAGES
# ============================================================

def render_overview() -> None:
    render_header()
    dataset_info = load_dataset_info()
    artifacts = load_monitoring_artifacts()

    total_texts = dataset_info["total_texts"] if dataset_info["total_texts"] is not None else "N/A"
    classes = dataset_info["num_classes"]

    final_test_df = artifacts["final_test_df"]
    critical_recall = safe_get_first_value(final_test_df, ["critical_recall"])
    champion_model = safe_get_first_value(final_test_df, ["champion_model", "model_name", "model"])

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Early Mental Health Risk Screening from Text</div>
            <div class="hero-text">
                A premium dashboard for mental health NLP classification, designed to showcase
                model predictions, deployment readiness, monitoring logic and clinically-inspired review workflows.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(metric_card("Total Texts", str(total_texts), "Dataset status", "accent-cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Classes", str(classes), "Multi-class NLP", "accent-purple"), unsafe_allow_html=True)
    with c3:
        st.markdown(
            metric_card("Critical Recall", format_metric(critical_recall), "Loaded from final test metrics", "accent-green"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "Champion Model",
                str(champion_model) if champion_model is not None else "N/A",
                "Loaded from artifacts",
                "accent-gold",
            ),
            unsafe_allow_html=True,
        )

    class_badges = "".join([f'<span class="small-badge">{escape(label)}</span>' for label in dataset_info["class_names"]])

    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">Detected Clinical Categories</div>
            <div class="section-text">
                {class_badges}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_message = (
        f"Connected to dataset: {dataset_info['dataset_path']}"
        if dataset_info["dataset_loaded"]
        else "Dataset file not found yet. The app still works in presentation mode."
    )

    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">Project Overview</div>
            <div class="section-text">
                {escape(dataset_message)}<br><br>
                This dashboard is structured to support prediction workflows, model loading, probability inspection,
                monitoring panels, session history, and deployment demonstration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_predictions() -> None:
    render_header()

    selected_sample = render_sample_text_buttons()
    default_text = selected_sample if selected_sample else ""

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Text Classification</div>
                <div class="section-text">
                    Paste a text sample, choose a model, and run a prediction.
                    If a local model file is not available, the app automatically switches to demo mode.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_text = st.text_area(
            "Enter text to analyze",
            value=default_text,
            height=220,
            placeholder="Write or paste text here...",
        )

        selected_model = st.selectbox(
            "Select model",
            ["LinearSVC Balanced", "Hybrid SVC", "MentalBERT", "BERT Base"],
            index=0,
        )

        analyze_clicked = st.button("Analyze Text", type="primary", use_container_width=True)

    with right_col:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Prediction Tips</div>
                <div class="section-text">
                    Use natural language examples similar to user posts or patient-style narratives.
                    The current interface supports local classical models first. Transformer slots are also prepared for future integration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
            return

        with st.spinner("Running prediction..."):
            model, model_path, load_error = load_joblib_model(selected_model)

            if model is not None:
                try:
                    predicted_label, confidence, prob_df = predict_with_model(model, user_text)
                    mode = "real"
                except Exception as exc:
                    st.error(f"Prediction failed with the loaded model: {exc}")
                    predicted_label, confidence, prob_df = fake_demo_prediction(user_text)
                    mode = "demo-fallback"
            else:
                predicted_label, confidence, prob_df = fake_demo_prediction(user_text)
                mode = "demo"

        save_prediction_to_history(
            text=user_text,
            model_name=selected_model,
            predicted_label=predicted_label,
            confidence=confidence,
            mode=mode,
        )

        st.session_state.last_prediction = {
            "text": user_text,
            "model": selected_model,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "mode": mode,
            "model_path": str(model_path) if model_path else None,
        }

        result_col, details_col = st.columns([1, 1])

        with result_col:
            confidence_text = f"{confidence * 100:.2f}%" if confidence is not None else "N/A"
            mode_label = "Real model" if mode == "real" else "Demo mode"

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Prediction Result</div>
                    <div class="result-value">{escape(predicted_label)}</div>
                    <div class="result-label">Confidence</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #22d3ee; margin-bottom: 0.9rem;">{escape(confidence_text)}</div>
                    <div class="divider-line"></div>
                    <div class="result-label">Selected Model</div>
                    <div style="color: #f8fafc; font-weight: 700; margin-bottom: 0.6rem;">{escape(selected_model)}</div>
                    <div class="result-label">Execution Mode</div>
                    <div style="color: #cbd5e1; font-weight: 600;">{escape(mode_label)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if load_error:
                st.info(load_error)

            if model_path:
                st.caption(f"Loaded from: {model_path}")

        with details_col:
            st.markdown(
                """
                <div class="section-box">
                    <div class="section-title">Class Probabilities</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_probability_table(prob_df)

        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Submitted Text</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(user_text)


def render_monitoring() -> None:
    render_header()

    artifacts = load_monitoring_artifacts()

    final_test_df = artifacts["final_test_df"]
    nested_cv_df = artifacts["nested_cv_df"]
    normal_cv_df = artifacts["normal_cv_df"]
    global_review_df = artifacts["global_review_df"]

    macro_f1 = safe_get_first_value(final_test_df, ["f1_macro", "macro_f1"])
    macro_recall = safe_get_first_value(final_test_df, ["recall_macro", "macro_recall"])
    critical_recall = safe_get_first_value(final_test_df, ["critical_recall"])
    champion_model = safe_get_first_value(final_test_df, ["champion_model", "model_name", "model"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card("Macro F1", format_metric(macro_f1), "Final test metric", "accent-cyan"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card("Macro Recall", format_metric(macro_recall), "Final test metric", "accent-green"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("Critical Recall", format_metric(critical_recall), "Priority classes", "accent-gold"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "Champion Model",
                str(champion_model) if champion_model is not None else "N/A",
                "Loaded from artifacts",
                "accent-purple",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Monitoring Panel</div>
            <div class="section-text">
                This page displays real evaluation artifacts when the CSV files are available in the project folders.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Final Test Metrics")
    if final_test_df is not None:
        st.dataframe(final_test_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Missing file: {FINAL_TEST_METRICS_PATH}")

    st.markdown("### Nested CV Summary")
    if nested_cv_df is not None:
        st.dataframe(nested_cv_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Missing file: {NESTED_CV_SUMMARY_PATH}")

    st.markdown("### Normal CV Summary")
    if normal_cv_df is not None:
        st.dataframe(normal_cv_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Missing file: {NORMAL_CV_SUMMARY_PATH}")

    st.markdown("### Global Clinical Review")
    if global_review_df is not None:
        st.dataframe(global_review_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Missing file: {GLOBAL_CLINICAL_REVIEW_PATH}")


def render_chat() -> None:
    render_header()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Project Copilot</div>
            <div class="section-text">
                Ask questions about the project, its purpose, triage logic, business value,
                metrics, development approach, or deployment setup.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"LLM model configured: {get_default_model()}")

    if st.session_state.qa_chain is None:
        try:
            st.session_state.qa_chain = build_qa_chain()
        except Exception:
            st.session_state.qa_chain = None

    if st.button("Clear chat", use_container_width=False):
        st.session_state.chat_messages = []
        st.success("Chat cleared.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask something about your project...")

    if user_prompt:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_prompt}
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ""

                if st.session_state.qa_chain is not None:
                    try:
                        result = st.session_state.qa_chain.invoke({"query": user_prompt})
                        answer = result["result"]

                        source_docs = result.get("source_documents", [])
                        if source_docs:
                            source_names = []
                            for doc in source_docs:
                                source_path = doc.metadata.get("source", "")
                                if source_path:
                                    source_names.append(Path(source_path).name)

                            if source_names:
                                unique_names = list(dict.fromkeys(source_names))
                                answer += "\n\n**Sources:** " + ", ".join(unique_names)

                    except Exception as exc:
                        answer = (
                            "RAG assistant failed, so I switched to OpenRouter.\n\n"
                            f"Technical detail: {exc}\n\n"
                        )

                if not answer:
                    try:
                        answer = answer_with_openrouter(user_prompt)
                    except Exception as exc:
                        answer = (
                            "I could not answer with either the RAG assistant or OpenRouter.\n\n"
                            f"OpenRouter error: {exc}\n\n"
                            "Check your .env file, OPENROUTER_API_KEY, and installed dependencies."
                        )

                st.markdown(answer)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )


def render_history() -> None:
    render_header()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Prediction History</div>
            <div class="section-text">
                This section stores the most recent predictions from the current session only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No predictions yet in this session.")
        return

    for item in st.session_state.history:
        confidence_text = (
            f"{item['confidence'] * 100:.2f}%"
            if item["confidence"] is not None
            else "N/A"
        )
        st.markdown(
            f"""
            <div class="history-card">
                <div class="history-title">{escape(str(item['label']))}</div>
                <div class="history-meta">
                    Model: {escape(str(item['model']))} | Confidence: {escape(confidence_text)} | Mode: {escape(str(item['mode']))}
                </div>
                <div class="history-text">{escape(str(item['text']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about() -> None:
    render_header()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">About This Dashboard</div>
            <div class="section-text">
                This Streamlit application was designed as a professional showcase interface for a mental health
                NLP classification project. It combines a premium visual design with practical deployment logic:
                model loading, text prediction, confidence display, monitoring artifacts and session history.
                <br><br>
                It is intended for demonstration, portfolio and MVP presentation purposes.
                It is not a diagnostic device.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Recommended Next Steps</div>
            <div class="section-text">
                1. Connect your real saved classical model.<br>
                2. Add transformer inference endpoints or transformer pipelines.<br>
                3. Show top-k classes and richer explanation panels.<br>
                4. Add charts from real evaluation CSV files.<br>
                5. Deploy on Streamlit Community Cloud or another platform.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    load_custom_css()
    page = build_sidebar()

    if page == "Overview":
        render_overview()
    elif page == "Predictions":
        render_predictions()
    elif page == "Monitoring":
        render_monitoring()
    elif page == "Chat":
        render_chat()
    elif page == "History":
        render_history()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()