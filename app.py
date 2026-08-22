import os
import time
import uuid
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config import (
    DATA_DIR,
    DEFAULT_LLM_PROVIDER,
    GROQ_MODEL,
    GROQ_API_KEY,
    MEDICAL_DISCLAIMER,
)
from src.document_loader import (
    load_all_documents,
    get_indexed_documents_summary,
)
from src.text_splitter import split_documents
from src.vectorstore import (
    index_chunks_into_vectorstore,
    get_vectorstore_count,
)
from src.retriever import create_hybrid_retriever, invalidate_bm25_cache
from src.rag_chain import MedicalRAGPipeline
from src.clinical_tools import (
    classify_blood_pressure,
    calculate_renal_function,
    stratify_cvd_risk,
)
from src.clinical_expert import (
    analyze_patient_case,
    check_drug_interactions,
    generate_patient_education_guide,
    get_guidelines_comparison,
)
from src.pdf_generator import (
    generate_clinical_soap_pdf,
    generate_patient_leaflet_pdf,
)
from src.evaluator import RAGEvaluator

# Page Configuration - Fully Responsive Multi-Device Clinical Platform
st.set_page_config(
    page_title="MediRAG | Clinical Intelligence & Guidelines Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

# -----------------------------------------------------------------------------
# FULLY RESPONSIVE ENTERPRISE CLINICAL DESIGN SYSTEM (All Screen Sizes)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #060913;
        --accent-cyan: #06b6d4;
        --accent-blue: #38bdf8;
        --accent-indigo: #6366f1;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Fluid Viewport Foundations */
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        min-height: 100dvh !important;
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        overflow-x: hidden !important;
    }

    .stApp {
        min-height: 100dvh !important;
        width: 100% !important;
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 85% 20%, rgba(99, 102, 241, 0.06) 0%, transparent 45%) !important;
        color: var(--text-main) !important;
        letter-spacing: -0.01em;
        overflow-x: hidden !important;
    }

    /* Responsive Main Container (Scales Fluidly Across Desktop, Tablets & Mobile) */
    .block-container {
        padding-top: clamp(0.8rem, 2vw, 1.5rem) !important;
        padding-bottom: clamp(6.5rem, 10vw, 8.5rem) !important;
        padding-left: clamp(0.75rem, 3vw, 2rem) !important;
        padding-right: clamp(0.75rem, 3vw, 2rem) !important;
        max-width: min(100%, 1150px) !important;
        margin: 0 auto !important;
    }

    /* Minimal Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.35);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #040711 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Brand Box */
    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(99, 102, 241, 0.08) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 12px 14px;
        border-radius: 10px;
        margin-bottom: 18px;
    }

    .brand-icon-monogram {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        color: #ffffff;
        box-shadow: 0 0 16px rgba(14, 165, 233, 0.35);
        flex-shrink: 0;
    }

    .brand-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: -0.5px;
        color: #ffffff;
        line-height: 1.1;
    }

    .brand-badge {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #38bdf8;
        text-transform: uppercase;
    }

    /* Responsive Horizontally-Scrollable Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 6px !important;
        background: rgba(15, 23, 42, 0.75) !important;
        padding: 6px 8px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 18px !important;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }

    .stTabs [data-baseweb="tab"] {
        flex-shrink: 0 !important;
        white-space: nowrap !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        font-weight: 600 !important;
        font-size: clamp(0.78rem, 1.4vw, 0.88rem) !important;
        color: #94a3b8 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9 !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.22) 0%, rgba(99, 102, 241, 0.18) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15) !important;
    }

    /* Responsive Clinical Cards */
    .clinical-card {
        background: rgba(13, 21, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: clamp(12px, 2vw, 18px);
        margin-bottom: 14px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        word-break: break-word;
    }

    .clinical-card-header {
        font-family: 'Outfit', sans-serif;
        font-size: clamp(1rem, 2vw, 1.18rem);
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .metric-badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .metric-badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .metric-badge-safe {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* Hero / Prompt Cards */
    .hero-wrapper {
        text-align: center;
        padding: clamp(20px, 4vw, 36px) 10px 16px 10px;
    }

    .hero-monogram {
        width: clamp(42px, 6vw, 52px);
        height: clamp(42px, 6vw, 52px);
        border-radius: 12px;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: clamp(1.3rem, 3vw, 1.7rem);
        color: #ffffff;
        margin-bottom: 12px;
        box-shadow: 0 0 24px rgba(14, 165, 233, 0.4);
    }

    .hero-heading {
        font-family: 'Outfit', sans-serif;
        font-size: clamp(1.35rem, 3.5vw, 1.8rem);
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: clamp(0.85rem, 1.6vw, 0.95rem);
        max-width: 600px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }

    /* Telemetry row */
    .telemetry-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }

    .telemetry-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.72rem;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Disclaimer Card */
    .disclaimer-pill {
        background: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-left: 3px solid #f59e0b;
        color: #fbbf24;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 12px;
        font-size: 0.76rem;
        line-height: 1.45;
    }

    /* Citations */
    .citation-wrapper {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }

    .citation-source-tag {
        font-size: 0.76rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.5px;
    }

    /* ------------------------------------------------------------------------- */
    /* RESPONSIVE CHATGPT-STYLE FLOATING CAPSULE DOCK (All Screen Sizes)         */
    /* ------------------------------------------------------------------------- */
    div[data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        background: linear-gradient(180deg, rgba(6, 9, 19, 0) 0%, rgba(6, 9, 19, 0.88) 30%, #060913 95%) !important;
        padding-top: 16px !important;
        pointer-events: none !important;
    }

    div[data-testid="stBottom"] > div {
        background: transparent !important;
        max-width: min(calc(100% - 1.5rem), 850px) !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-bottom: clamp(10px, 2vh, 22px) !important;
        pointer-events: auto !important;
    }

    /* Single Outer Capsule Border */
    div[data-testid="stChatInput"] {
        background: rgba(13, 21, 38, 0.96) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 24px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.7), 0 0 1px 1px rgba(255, 255, 255, 0.05) inset !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        padding: 4px 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8, 0 12px 40px rgba(14, 165, 233, 0.25) !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0 !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: clamp(0.88rem, 1.8vw, 0.95rem) !important;
        color: #f8fafc !important;
        line-height: 1.5 !important;
        padding: 6px 10px !important;
        resize: none !important;
    }

    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        border: none !important;
        outline: none !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.35) !important;
        transition: transform 0.2s ease !important;
        flex-shrink: 0 !important;
    }

    /* Mobile Adaptations */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.6rem !important;
            padding-bottom: 5.5rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
        }

        .hero-heading {
            font-size: 1.35rem !important;
        }

        .hero-sub {
            font-size: 0.84rem !important;
        }

        .clinical-card {
            padding: 12px 12px !important;
            border-radius: 10px !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px !important;
            font-size: 0.76rem !important;
        }

        div[data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "session_default": {
            "title": "Clinical Consultation",
            "created_at": time.time(),
            "messages": [],
        }
    }

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "session_default"

if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = get_vectorstore_count()

if st.session_state.current_session_id not in st.session_state.chat_sessions:
    st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]

current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
active_messages = current_session["messages"]


# -----------------------------------------------------------------------------
# SIDEBAR (Obsidian ChatGPT-Style Control Center)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-icon-monogram">M</div>
        <div>
            <div class="brand-title">MediRAG</div>
            <div class="brand-badge">Clinical Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat Button
    if st.button("+ New Clinical Consultation", use_container_width=True, type="primary"):
        new_sid = f"session_{uuid.uuid4().hex[:6]}"
        st.session_state.chat_sessions[new_sid] = {
            "title": "New Consultation",
            "created_at": time.time(),
            "messages": [],
        }
        st.session_state.current_session_id = new_sid
        st.rerun()

    # Session History List (ChatGPT Style)
    st.markdown("<p style='font-size: 0.74rem; color: #64748b; font-weight: 700; letter-spacing: 0.8px; margin: 12px 0 6px 2px;'>CONSULTATION SESSIONS</p>", unsafe_allow_html=True)
    for sid, sdata in list(st.session_state.chat_sessions.items())[::-1]:
        is_selected = (sid == st.session_state.current_session_id)
        btn_caption = f"{sdata['title']}"
        if st.button(btn_caption, key=f"session_btn_{sid}", use_container_width=True, disabled=is_selected):
            st.session_state.current_session_id = sid
            st.rerun()

    st.markdown("---")

    # Model & Retrieval Settings
    with st.expander("Model & Retrieval Engine", expanded=False):
        provider = st.selectbox("LLM Provider", options=["groq", "gemini", "ollama", "huggingface"], index=0)
        
        api_key_val = ""
        if provider == "groq":
            api_key_val = st.text_input("Groq API Key", value=GROQ_API_KEY or os.getenv("GROQ_API_KEY", ""), type="password")
            model_selected = st.selectbox(
                "Inference Model",
                options=[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                    "qwen/qwen3.6-27b",
                ],
                index=0,
            )
        elif provider == "gemini":
            api_key_val = st.text_input("Gemini API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
            model_selected = st.selectbox("Inference Model", options=["gemini-1.5-flash", "gemini-2.0-flash"], index=0)
        elif provider == "ollama":
            model_selected = st.text_input("Ollama Model Name", value="llama3.2:3b")
        else:
            model_selected = "Qwen/Qwen2.5-1.5B-Instruct"

        top_k_chunks = st.slider("Top Chunks (k)", min_value=2, max_value=8, value=6)
        enable_reranker = st.toggle("Stage-2 Cross-Encoder Reranker", value=True)

    # Guidelines Repository Summary
    doc_summary = get_indexed_documents_summary(DATA_DIR)
    with st.expander(f"Guidelines Corpus ({len(doc_summary)} Documents)", expanded=False):
        st.caption(f"Indexed Chunks: **{st.session_state.indexed_count} vectors**")
        for doc in doc_summary:
            st.markdown(f"- **{doc['filename']}** ({doc['total_pages']} pgs)")

        uploaded_file = st.file_uploader("Upload Guideline PDF", type=["pdf"])
        if uploaded_file is not None:
            save_path = DATA_DIR / uploaded_file.name
            if not save_path.exists():
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {uploaded_file.name}")

        if st.button("Re-index Guidelines", use_container_width=True):
            with st.spinner("Rebuilding indexes..."):
                invalidate_bm25_cache()
                raw_docs = load_all_documents(DATA_DIR)
                if raw_docs:
                    chunks = split_documents(raw_docs)
                    index_chunks_into_vectorstore(chunks, clear_existing=True)
                    st.session_state.indexed_count = get_vectorstore_count()
                    st.success("Indexing complete.")
                    st.rerun()

    # Clear Active Consultation
    if st.button("Clear Consultation History", use_container_width=True):
        current_session["messages"] = []
        st.rerun()


# -----------------------------------------------------------------------------
# MAIN VIEW: 7 SPECIALIZED CLINICAL TABS (Fully Responsive)
# -----------------------------------------------------------------------------
tab_chat, tab_case, tab_ddi, tab_calc, tab_guidelines, tab_patient, tab_eval = st.tabs([
    "💬 Clinical Consultation",
    "📋 Patient Case & SOAP",
    "💊 Drug Interactions (DDI)",
    "🧮 Rapid Calculators",
    "⚖️ Guideline Consensus",
    "📄 Patient Leaflet",
    "📊 Benchmark Suite",
])


# =============================================================================
# TAB 1: 💬 CLINICAL CONSULTATION (PURE CHATGPT LIVE CHAT)
# =============================================================================
with tab_chat:
    if not active_messages:
        st.markdown("""
        <div class="hero-wrapper">
            <div class="hero-monogram">M</div>
            <div class="hero-heading">Clinical Decision Support System</div>
            <p class="hero-sub">
                Zero-hallucination medical AI grounded exclusively in official WHO and NICE guidelines with verified page citations and continuous multi-turn memory.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("First-Line Antihypertensive Classes\n\nWhat are the recommended first-line drug classes for adult hypertension?", use_container_width=True):
                st.session_state.temp_prompt = "What are the recommended first-line pharmacological treatment classes for hypertension in adults according to official guidelines?"
                st.rerun()

            if st.button("Pregnancy Contraindications & Safety\n\nWhich antihypertensive drug classes are strictly contraindicated in pregnant women?", use_container_width=True):
                st.session_state.temp_prompt = "Which antihypertensive drug classes are strictly contraindicated in pregnant women, and what are the preferred safe alternatives?"
                st.rerun()

        with col2:
            if st.button("Target Blood Pressure Thresholds\n\nWhat is the recommended target blood pressure goal for patients with comorbidities?", use_container_width=True):
                st.session_state.temp_prompt = "What are the recommended target blood pressure thresholds for adults with hypertension and cardiovascular comorbidities?"
                st.rerun()

            if st.button("Elderly Patient Protocol (Age >= 80)\n\nWhat are the target blood pressure numbers and dosing strategies for patients over 80?", use_container_width=True):
                st.session_state.temp_prompt = "What are the specific blood pressure targets and dosing considerations for elderly patients aged 80 years and older?"
                st.rerun()

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Render Active Messages
    for msg in active_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                if "metadata" in msg:
                    m = msg["metadata"]
                    st.markdown(f"""
                    <div class="telemetry-row">
                        <span class="telemetry-chip">Latency: {m.get('latency', '0.0')}s</span>
                        <span class="telemetry-chip">{m.get('status', 'Strictly Grounded')}</span>
                        <span class="telemetry-chip">Citations: {m.get('citations_count', 0)}</span>
                    </div>
                    """, unsafe_allow_html=True)

                if "citations" in msg and msg["citations"]:
                    with st.expander(f"Verified Page Citations ({len(msg['citations'])})", expanded=False):
                        for idx, c in enumerate(msg["citations"], 1):
                            st.markdown(f"""
                            <div class="citation-wrapper">
                                <span class="citation-source-tag">Reference {idx}: {c['source']} (Page {c['page']})</span>
                                <p style="color: #cbd5e1; font-style: italic; margin: 4px 0 0 0; font-size: 0.88rem;">"{c['snippet']}"</p>
                            </div>
                            """, unsafe_allow_html=True)

    # Chat Input Area (Responsive Capsule Dock)
    user_input = st.chat_input("Message MediRAG clinical assistant...")
    active_query = user_input or st.session_state.pop("temp_prompt", None)

    if active_query:
        if not active_messages:
            current_session["title"] = active_query[:32] + ("..." if len(active_query) > 32 else "")

        active_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                retriever = create_hybrid_retriever(top_k=top_k_chunks, use_reranker=enable_reranker)
                pipeline = MedicalRAGPipeline(
                    retriever=retriever,
                    provider=provider,
                    api_key=api_key_val,
                    model_name=model_selected,
                    temperature=0.0,
                )

                result_container = {}
                history_turns = active_messages[:-1]
                stream_gen = pipeline.stream_query(active_query, result_container, chat_history=history_turns)

                answer = message_placeholder.write_stream(stream_gen)

                citations = result_container.get("citations", [])
                latency = result_container.get("latency_seconds", 0.0)
                status = result_container.get("grounding_status", "Grounded")

                st.markdown(f"""
                <div class="telemetry-row">
                    <span class="telemetry-chip">Latency: {latency}s</span>
                    <span class="telemetry-chip">{status}</span>
                    <span class="telemetry-chip">Citations: {len(citations)}</span>
                </div>
                """, unsafe_allow_html=True)

                if citations:
                    with st.expander(f"Verified Page Citations ({len(citations)})", expanded=True):
                        for idx, c in enumerate(citations, 1):
                            st.markdown(f"""
                            <div class="citation-wrapper">
                                <span class="citation-source-tag">Reference {idx}: {c['source']} (Page {c['page']})</span>
                                <p style="color: #cbd5e1; font-style: italic; margin: 4px 0 0 0; font-size: 0.88rem;">"{c['snippet']}"</p>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="disclaimer-pill">
                    {MEDICAL_DISCLAIMER}
                </div>
                """, unsafe_allow_html=True)

                active_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "metadata": {
                        "latency": latency,
                        "status": status,
                        "citations_count": len(citations),
                    }
                })

            except Exception as e:
                st.error(f"Clinical Assistant Error: {str(e)}")


# =============================================================================
# TAB 2: 📋 PATIENT CASE & SOAP PATHWAY (WITH PDF EXPORT)
# =============================================================================
with tab_case:
    st.markdown("""
    <div class="clinical-card">
        <div class="clinical-card-header">📋 Patient Case Analyzer & SOAP Decision Pathway</div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Evaluates comprehensive clinical parameters, classifies BP & KDIGO renal stages, audits drug interactions, and synthesizes a full SOAP note with downloadable official PDF report.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("case_analyzer_form"):
        col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
        with col_p1:
            st.markdown("<p style='font-weight:700; color:#38bdf8; font-size:0.85rem;'>1. DEMOGRAPHICS & VITALS</p>", unsafe_allow_html=True)
            p_age = st.number_input("Patient Age (Years)", min_value=18, max_value=110, value=62)
            p_sex = st.selectbox("Biological Sex", options=["Male", "Female"], index=0)
            p_sbp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=260, value=158)
            p_dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=96)
            p_weight = st.number_input("Patient Weight (kg)", min_value=30.0, max_value=250.0, value=78.0)

        with col_p2:
            st.markdown("<p style='font-weight:700; color:#38bdf8; font-size:0.85rem;'>2. COMORBIDITIES & LABS</p>", unsafe_allow_html=True)
            p_diabetes = st.checkbox("Diabetes Mellitus (Type 1 / 2)", value=True)
            p_cvd = st.checkbox("Established CVD (Prior MI / Stroke)", value=False)
            p_ckd = st.checkbox("Chronic Kidney Disease (CKD)", value=False)
            p_smoker = st.checkbox("Current Tobacco Smoker", value=False)
            p_creat = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=15.0, value=1.3, step=0.1)

        with col_p3:
            st.markdown("<p style='font-weight:700; color:#38bdf8; font-size:0.85rem;'>3. ACTIVE MEDICATIONS</p>", unsafe_allow_html=True)
            p_meds = st.multiselect(
                "Active Medication Classes",
                options=[
                    "ace_inhibitor",
                    "arb",
                    "ccb_dhp",
                    "verapamil_diltiazem",
                    "thiazide_diuretic",
                    "spironolactone",
                    "beta_blocker",
                    "nsaid",
                ],
                default=["ace_inhibitor", "spironolactone"],
                format_func=lambda x: {
                    "ace_inhibitor": "ACE Inhibitor (Lisinopril, Enalapril)",
                    "arb": "ARB (Losartan, Valsartan)",
                    "ccb_dhp": "CCB Dihydropyridine (Amlodipine)",
                    "verapamil_diltiazem": "Non-DHP CCB (Verapamil, Diltiazem)",
                    "thiazide_diuretic": "Thiazide-like Diuretic (Indapamide)",
                    "spironolactone": "MRA (Spironolactone, Eplerenone)",
                    "beta_blocker": "Beta Blocker (Bisoprolol, Metoprolol)",
                    "nsaid": "NSAIDs (Ibuprofen, Naproxen)",
                }.get(x, x),
            )
            p_clinician = st.text_input("Clinician Name", value="Dr. Omar Elsayed, MD")
            p_facility = st.text_input("Medical Facility / Unit", value="Cardiology Clinical Services")

        btn_analyze = st.form_submit_button("⚡ Analyze Patient & Synthesize SOAP Pathway", type="primary", use_container_width=True)

    if btn_analyze:
        with st.spinner("Synthesizing clinical decision pathway..."):
            case_res = analyze_patient_case(
                age=p_age,
                is_female=(p_sex == "Female"),
                systolic_bp=p_sbp,
                diastolic_bp=p_dbp,
                has_diabetes=p_diabetes,
                has_cvd=p_cvd,
                has_ckd=p_ckd,
                is_smoker=p_smoker,
                serum_creatinine_mg_dl=p_creat,
                weight_kg=p_weight,
                current_medications=p_meds,
            )

            psum = case_res.get("patient_summary", {})
            st.markdown("### Clinical Evaluation Overview")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("BP Stage", psum.get("bp_stage", "N/A"), delta=psum.get("severity_tier", "Standard"))
            with m_col2:
                st.metric("Target BP Goal", psum.get("target_bp", "< 140/90"))
            with m_col3:
                st.metric("10-Yr CVD Risk", psum.get("cvd_risk_tier", "Moderate"))
            with m_col4:
                st.metric("KDIGO Renal Stage", psum.get("renal_stage", "Pending"), delta=f"eGFR {psum.get('egfr', 'N/A')}")

            # DDI Alerts
            ddi = case_res.get("drug_interactions")
            if ddi and ddi.get("interactions"):
                st.markdown("#### 🚨 Drug Interaction & Contraindication Warnings")
                for inter in ddi["interactions"]:
                    badge_cls = "metric-badge-critical" if inter.get("severity_code") == "critical" else "metric-badge-warning"
                    st.markdown(f"""
                    <div class="clinical-card" style="border-left: 4px solid {'#ef4444' if inter.get('severity_code')=='critical' else '#f59e0b'};">
                        <span class="{badge_cls}">{inter.get('severity')}</span>
                        <div style="font-weight:700; color:#ffffff; margin: 6px 0 4px 0;">{inter.get('title')}</div>
                        <p style="color:#cbd5e1; font-size:0.86rem; margin:0 0 4px 0;"><b>Mechanism:</b> {inter.get('mechanism')}</p>
                        <p style="color:#38bdf8; font-size:0.86rem; margin:0;"><b>Action:</b> {inter.get('clinical_action')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="clinical-card" style="border-left: 4px solid #22c55e;">
                    <span class="metric-badge metric-badge-safe">Medication Safety Verified</span>
                    <p style="color:#cbd5e1; font-size:0.88rem; margin:6px 0 0 0;">No critical drug-drug contraindications detected in the selected regimen.</p>
                </div>
                """, unsafe_allow_html=True)

            # Structured SOAP Display
            soap = case_res.get("soap_note", {})
            st.markdown("#### 📝 Synthesized Clinical SOAP Note")
            with st.container():
                st.markdown(f"**Subjective (S):** {soap.get('subjective')}")
                st.markdown(f"**Objective (O):** {soap.get('objective')}")
                st.markdown(f"**Assessment (A):** {soap.get('assessment')}")
                st.markdown("**Plan (P) - Evidence-Based Clinical Pathway:**")
                for step in case_res.get("clinical_pathway", []):
                    st.markdown(f"- **Step {step['step']}: {step['title']}** — {step['description']} *(Source: {step['guideline']})*")

            # Generate PDF and provide download button
            pdf_data = generate_clinical_soap_pdf(
                case_result=case_res,
                clinician_name=p_clinician,
                institution_name=p_facility,
            )

            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Official Clinical SOAP Report (PDF)",
                data=pdf_data,
                file_name=f"Clinical_SOAP_Report_Age{p_age}_{p_sex}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )


# =============================================================================
# TAB 3: 💊 DRUG-DRUG INTERACTION (DDI) MATRIX
# =============================================================================
with tab_ddi:
    st.markdown("""
    <div class="clinical-card">
        <div class="clinical-card-header">💊 Clinical Drug-Drug Interaction (DDI) Matrix</div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Live contraindication surveillance engine based on WHO, NICE, and KDIGO clinical pharmacology standards.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    with col_pre1:
        if st.button("Preset: Dual RAS Blockade", use_container_width=True):
            st.session_state.ddi_selection = ["ace_inhibitor", "arb"]
            st.rerun()
    with col_pre2:
        if st.button("Preset: Hyperkalemia Risk", use_container_width=True):
            st.session_state.ddi_selection = ["ace_inhibitor", "spironolactone"]
            st.rerun()
    with col_pre3:
        if st.button("Preset: AV Block Risk", use_container_width=True):
            st.session_state.ddi_selection = ["beta_blocker", "verapamil_diltiazem"]
            st.rerun()
    with col_pre4:
        if st.button("Preset: Renal & BP Blunting", use_container_width=True):
            st.session_state.ddi_selection = ["ace_inhibitor", "nsaid", "thiazide_diuretic"]
            st.rerun()

    current_ddi_sel = st.session_state.get("ddi_selection", ["ace_inhibitor", "spironolactone", "nsaid"])

    selected_ddi_meds = st.multiselect(
        "Select Medication Classes to Evaluate",
        options=[
            "ace_inhibitor",
            "arb",
            "ccb_dhp",
            "verapamil_diltiazem",
            "thiazide_diuretic",
            "spironolactone",
            "beta_blocker",
            "nsaid",
        ],
        default=current_ddi_sel,
        format_func=lambda x: {
            "ace_inhibitor": "ACE Inhibitors (Lisinopril, Enalapril, Ramipril)",
            "arb": "ARBs (Losartan, Valsartan, Candesartan)",
            "ccb_dhp": "CCB Dihydropyridine (Amlodipine, Nifedipine)",
            "verapamil_diltiazem": "Non-DHP CCB (Verapamil, Diltiazem)",
            "thiazide_diuretic": "Thiazide Diuretics (Chlorthalidone, Indapamide)",
            "spironolactone": "MRA (Spironolactone, Eplerenone)",
            "beta_blocker": "Beta Blockers (Bisoprolol, Metoprolol, Atenolol)",
            "nsaid": "NSAIDs (Ibuprofen, Naproxen, Diclofenac)",
        }.get(x, x),
    )

    if selected_ddi_meds:
        ddi_eval_res = check_drug_interactions(selected_ddi_meds)
        total_f = ddi_eval_res.get("total_interactions_found", 0)

        if total_f > 0:
            st.markdown(f"#### ⚠️ {total_f} Clinical Interactions Detected")
            for item in ddi_eval_res.get("interactions", []):
                is_crit = item.get("severity_code") == "critical"
                border_c = "#ef4444" if is_crit else "#f59e0b"
                badge_c = "metric-badge-critical" if is_crit else "metric-badge-warning"

                st.markdown(f"""
                <div class="clinical-card" style="border-left: 4px solid {border_c};">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap;">
                        <span style="font-weight:700; font-size:1.05rem; color:#ffffff;">{item.get('title')}</span>
                        <span class="{badge_c}">{item.get('severity')}</span>
                    </div>
                    <p style="color:#94a3b8; font-size:0.85rem; margin:0 0 6px 0;"><b>Involved Drugs:</b> {', '.join(item.get('drugs_involved', []))}</p>
                    <p style="color:#cbd5e1; font-size:0.88rem; margin:0 0 6px 0;"><b>Mechanism:</b> {item.get('mechanism')}</p>
                    <p style="color:#38bdf8; font-size:0.88rem; margin:0;"><b>Recommended Action:</b> {item.get('clinical_action')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="clinical-card" style="border-left: 4px solid #22c55e;">
                <span class="metric-badge metric-badge-safe">Safe Combination</span>
                <p style="color:#cbd5e1; font-size:0.9rem; margin:8px 0 0 0;">
                    No known major or contraindicated interactions found between the selected medication classes.
                </p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: 🧮 RAPID CLINICAL CALCULATORS
# =============================================================================
with tab_calc:
    st.markdown("""
    <div class="clinical-card">
        <div class="clinical-card-header">🧮 Rapid Clinical Decision Calculators</div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Evidence-based algorithmic calculations for Blood Pressure Staging, CKD-EPI 2021 eGFR, Cockcroft-Gault CrCl, and 10-year CVD Risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    calc_bp_col, calc_renal_col, calc_cvd_col = st.columns([1, 1, 1])

    # 1. BP Calculator
    with calc_bp_col:
        st.markdown("<p style='font-weight:700; color:#38bdf8;'>1. Blood Pressure Stager</p>", unsafe_allow_html=True)
        c_sbp = st.number_input("Systolic (mmHg)", 70, 260, 152, key="c_sbp")
        c_dbp = st.number_input("Diastolic (mmHg)", 40, 160, 94, key="c_dbp")
        c_dm = st.checkbox("Diabetes", value=False, key="c_dm")
        c_cvd = st.checkbox("Known CVD", value=False, key="c_cvd")
        c_ckd = st.checkbox("Known CKD", value=False, key="c_ckd")

        bp_res = classify_blood_pressure(c_sbp, c_dbp, c_dm, c_cvd, c_ckd)
        st.markdown(f"""
        <div class="clinical-card" style="padding:10px 14px;">
            <div style="font-weight:700; color:#ffffff;">{bp_res.get('stage')}</div>
            <p style="color:#38bdf8; font-size:0.82rem; margin:4px 0 0 0;">Target BP: <b>{bp_res.get('target_bp')}</b></p>
            <p style="color:#94a3b8; font-size:0.78rem; margin:4px 0 0 0;">{bp_res.get('action_plan')}</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Renal Calculator
    with calc_renal_col:
        st.markdown("<p style='font-weight:700; color:#38bdf8;'>2. Renal Function & eGFR</p>", unsafe_allow_html=True)
        c_creat = st.number_input("Serum Creatinine (mg/dL)", 0.2, 15.0, 1.4, step=0.1, key="c_creat")
        c_age = st.number_input("Age (Years)", 18, 110, 65, key="c_age_r")
        c_sex = st.selectbox("Sex", ["Male", "Female"], key="c_sex_r")
        c_wt = st.number_input("Weight (kg)", 30.0, 250.0, 75.0, key="c_wt_r")

        renal_res = calculate_renal_function(c_creat, c_age, (c_sex == "Female"), c_wt)
        st.markdown(f"""
        <div class="clinical-card" style="padding:10px 14px;">
            <div style="font-weight:700; color:#ffffff;">eGFR: {renal_res.get('egfr_ckd_epi')} mL/min/1.73m²</div>
            <p style="color:#fbbf24; font-size:0.82rem; margin:4px 0 0 0;">KDIGO: <b>{renal_res.get('kdigo_stage')}</b></p>
            <p style="color:#38bdf8; font-size:0.82rem; margin:2px 0 0 0;">CrCl: <b>{renal_res.get('crcl_cockcroft_gault_ml_min')} mL/min</b></p>
            <p style="color:#cbd5e1; font-size:0.78rem; margin:4px 0 0 0;">{renal_res.get('dose_adjustment_advice')}</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. CVD Calculator
    with calc_cvd_col:
        st.markdown("<p style='font-weight:700; color:#38bdf8;'>3. 10-Yr CVD Risk Tier</p>", unsafe_allow_html=True)
        c_cvd_age = st.number_input("Age", 20, 100, 58, key="c_cvd_age")
        c_cvd_sbp = st.number_input("Systolic BP", 80, 260, 145, key="c_cvd_sbp")
        c_cvd_smoke = st.checkbox("Smoker", value=True, key="c_cvd_smoke")
        c_cvd_dm = st.checkbox("Diabetes", value=True, key="c_cvd_dm")

        cvd_res = stratify_cvd_risk(c_cvd_age, c_cvd_sbp, c_cvd_smoke, c_cvd_dm)
        st.markdown(f"""
        <div class="clinical-card" style="padding:10px 14px;">
            <div style="font-weight:700; color:#ffffff;">Risk: {cvd_res.get('risk_tier')}</div>
            <p style="color:#38bdf8; font-size:0.82rem; margin:4px 0 0 0;">Score Points: <b>{cvd_res.get('risk_points')}</b></p>
            <p style="color:#cbd5e1; font-size:0.78rem; margin:4px 0 0 0;">{cvd_res.get('recommendation')}</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 5: ⚖️ GUIDELINES CONSENSUS MATRIX
# =============================================================================
with tab_guidelines:
    st.markdown("""
    <div class="clinical-card">
        <div class="clinical-card-header">⚖️ Multi-Guideline Consensus & Divergence Matrix</div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Side-by-side clinical comparison across World Health Organization (WHO 2021), NICE UK (NG136), and ACC/AHA (2017) hypertension guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    matrix_items = get_guidelines_comparison()
    for item in matrix_items:
        st.markdown(f"#### 📌 {item['topic']}")
        col_g1, col_g2, col_g3 = st.columns([1, 1, 1])
        with col_g1:
            st.markdown(f"""
            <div class="clinical-card" style="border-top: 3px solid #0ea5e9;">
                <span style="font-weight:700; color:#38bdf8; font-size:0.85rem;">WHO 2021 Guideline</span>
                <p style="color:#cbd5e1; font-size:0.86rem; margin:8px 0 0 0;">{item['who_2021']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_g2:
            st.markdown(f"""
            <div class="clinical-card" style="border-top: 3px solid #6366f1;">
                <span style="font-weight:700; color:#818cf8; font-size:0.85rem;">NICE NG136 (UK)</span>
                <p style="color:#cbd5e1; font-size:0.86rem; margin:8px 0 0 0;">{item['nice_ng136']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_g3:
            st.markdown(f"""
            <div class="clinical-card" style="border-top: 3px solid #ec4899;">
                <span style="font-weight:700; color:#f472b6; font-size:0.85rem;">ACC / AHA 2017 (US)</span>
                <p style="color:#cbd5e1; font-size:0.86rem; margin:8px 0 0 0;">{item['acc_aha_2017']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="disclaimer-pill" style="margin-top:-6px; margin-bottom:20px; border-left-color:#38bdf8; color:#e2e8f0; background:rgba(14, 165, 233, 0.08);">
            <b>Global Clinical Consensus:</b> {item['consensus']}
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 6: 📄 PATIENT EDUCATION LEAFLET (WITH PDF EXPORT)
# =============================================================================
with tab_patient:
    st.markdown("""
    <div class="clinical-card">
        <div class="clinical-card-header">📄 Patient Health & Discharge Education Generator</div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Generates clear, patient-centered lifestyle instructions, medication safety rules, and emergency red-flag warning sheets (English & Arabic).
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_leaflet_form"):
        col_lf1, col_lf2, col_lf3 = st.columns([1, 1, 1])
        with col_lf1:
            leaf_cond = st.text_input("Diagnosed Condition", value="Stage 2 Essential Hypertension")
        with col_lf2:
            leaf_target = st.text_input("Target Blood Pressure Goal", value="< 130/80 mmHg")
        with col_lf3:
            leaf_lang = st.selectbox("Leaflet Language", ["Arabic", "English"], index=0)

        btn_gen_leaf = st.form_submit_button("📄 Generate Discharge Leaflet", type="primary", use_container_width=True)

    if btn_gen_leaf:
        guide_res = generate_patient_education_guide(
            condition_name=leaf_cond,
            target_bp=leaf_target,
            language=leaf_lang,
        )

        st.markdown(f"### {guide_res.get('title')}")
        st.info(guide_res.get("condition_summary"))

        col_lr1, col_lr2 = st.columns(2)
        with col_lr1:
            st.markdown("#### 🥗 Lifestyle & Nutrition Rules")
            for rule in guide_res.get("lifestyle_rules", []):
                st.markdown(f"- {rule}")

        with col_lr2:
            st.markdown("#### 💊 Medication Safety Tips")
            for tip in guide_res.get("medication_safety_tips", []):
                st.markdown(f"- {tip}")

        st.markdown("#### 🚨 Acute Emergency Red Flags")
        for flag in guide_res.get("emergency_red_flags", []):
            st.markdown(f"- **{flag}**")
        st.error(guide_res.get("emergency_action"))

        # Generate Leaflet PDF
        leaf_pdf_bytes = generate_patient_leaflet_pdf(guide_res)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Patient Discharge Leaflet (PDF)",
            data=leaf_pdf_bytes,
            file_name=f"Patient_Discharge_Guide_{leaf_lang}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )


# =============================================================================
# TAB 7: 📊 RAG BENCHMARK & EVALUATION SUITE
# =============================================================================
with tab_eval:
    st.markdown("""
    <div class="clinical-card">
        <div class="clinical-card-header">📊 MediRAG Benchmark & RAG Triad Evaluation Suite</div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Automated quality assurance benchmark measuring Faithfulness, Context Relevance, Hit Rate, Safety Refusal Rate, and Latency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Live RAG Triad Benchmark", type="primary"):
        with st.spinner("Executing automated test suite against clinical guidelines..."):
            evaluator = RAGEvaluator()
            bench_res = evaluator.run_benchmark()

            st.markdown("### Benchmark Scorecards")
            ev1, ev2, ev3, ev4, ev5 = st.columns(5)
            with ev1:
                st.metric("Faithfulness", f"{bench_res.get('faithfulness_score', 0)*100:.1f}%")
            with ev2:
                st.metric("Context Relevance", f"{bench_res.get('context_relevance_score', 0)*100:.1f}%")
            with ev3:
                st.metric("Recall Hit Rate", f"{bench_res.get('hit_rate', 0)*100:.1f}%")
            with ev4:
                st.metric("Safety Refusal", f"{bench_res.get('safety_refusal_rate', 0)*100:.1f}%")
            with ev5:
                st.metric("Mean Latency", f"{bench_res.get('mean_latency_seconds', 0):.2f}s")

            st.markdown("#### Test Case Execution Details")
            for idx, r in enumerate(bench_res.get("detailed_results", []), 1):
                status_color = "#22c55e" if r.get("status") == "PASS" else "#ef4444"
                st.markdown(f"""
                <div class="clinical-card" style="padding:10px 14px; margin-bottom:8px; border-left:3px solid {status_color};">
                    <div style="font-weight:700; color:#ffffff;">Test {idx}: {r.get('query')}</div>
                    <p style="color:#94a3b8; font-size:0.8rem; margin:4px 0 0 0;">
                        Status: <b>{r.get('status')}</b> | Faithfulness: {r.get('faithfulness')} | Chunks: {r.get('retrieved_chunks')} | Latency: {r.get('latency')}s
                    </p>
                </div>
                """, unsafe_allow_html=True)
