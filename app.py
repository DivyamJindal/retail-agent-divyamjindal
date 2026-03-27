"""
RetailMind Product Intelligence Agent — Streamlit UI
=====================================================
A conversational AI agent for StyleCraft's product manager to analyse
inventory, pricing, reviews, and catalog performance in real time.

Launch:  python run.py  OR  streamlit run app.py
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailMind — Product Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Glassmorphic Dark UI — Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Base ---- */
    .stApp {
        background: #09090b;
        color: #fafafa;
    }
    section[data-testid="stSidebar"] {
        background: rgba(24, 24, 27, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ---- Glass card for metrics ---- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {
        color: #a1a1aa;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fafafa;
        font-weight: 600;
    }

    /* ---- Chat bubbles ---- */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    /* User messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: rgba(255,255,255,0.03);
    }
    /* Assistant messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255,255,255,0.02);
    }

    /* ---- Intent badges ---- */
    .intent-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
    }
    .pill-INVENTORY { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
    .pill-PRICING   { background: rgba(234,179,8,0.12);  color: #facc15; border: 1px solid rgba(234,179,8,0.25); }
    .pill-REVIEWS   { background: rgba(168,85,247,0.12); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); }
    .pill-CATALOG   { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
    .pill-GENERAL   { background: rgba(161,161,170,0.10); color: #a1a1aa; border: 1px solid rgba(161,161,170,0.20); }

    /* ---- Tool log glass card ---- */
    .tool-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
        font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
    }
    .tool-fn { color: #60a5fa; font-weight: 600; }
    .tool-args { color: #a1a1aa; }

    /* ---- Expander ---- */
    details summary {
        color: #71717a;
        font-size: 0.8rem;
    }

    /* ---- Dividers ---- */
    hr { border-color: rgba(255,255,255,0.06); }

    /* ---- Buttons ---- */
    .stButton > button {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        color: #fafafa;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.18);
    }

    /* ---- Chat input ---- */
    .stChatInput > div {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.03) !important;
    }

    /* ---- Selectbox ---- */
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
    }

    /* ---- Toggle ---- */
    .stToggle label span { color: #a1a1aa; }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("retailmind_products.csv"), pd.read_csv("retailmind_reviews.csv")

products_df, reviews_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 RetailMind")
    st.caption("Product Intelligence for StyleCraft")
    st.divider()

    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    if api_key:
        st.success("Key loaded", icon="✅")
    else:
        st.warning("Enter API key to start", icon="⚠️")

    st.divider()

    # Category Filter
    st.markdown("##### Category Filter")
    categories = ["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"]
    selected_category = st.selectbox("Filter:", categories, index=0, label_visibility="collapsed")

    st.divider()

    # Catalog Summary
    st.markdown("##### 📊 Catalog Summary")
    summary_df = products_df if selected_category == "All Categories" else products_df[products_df["category"] == selected_category]

    total_skus = len(summary_df)
    critical_stock = sum(
        1 for _, r in summary_df.iterrows()
        if r["avg_daily_sales"] > 0 and r["stock_quantity"] / r["avg_daily_sales"] < 7
    )
    avg_margin = round(((summary_df["price"] - summary_df["cost"]) / summary_df["price"] * 100).mean(), 1)
    avg_rating = round(summary_df["avg_rating"].mean(), 1)

    c1, c2 = st.columns(2)
    c1.metric("Total SKUs", total_skus)
    c2.metric("Critical Stock", critical_stock)
    c3, c4 = st.columns(2)
    c3.metric("Avg Margin", f"{avg_margin}%")
    c4.metric("Avg Rating", f"{avg_rating}⭐")

    if selected_category != "All Categories":
        st.caption(f"Filtered to **{selected_category}**")

    st.divider()

    # Controls
    show_logs = st.toggle("Show Agent Logs", value=True)

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.briefing_shown = False
        if "agent" in st.session_state and st.session_state.agent:
            st.session_state.agent.clear_memory()
        st.rerun()

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "briefing_shown" not in st.session_state:
    st.session_state.briefing_shown = False
if "agent" not in st.session_state:
    st.session_state.agent = None

# ---------------------------------------------------------------------------
# Agent Init
# ---------------------------------------------------------------------------
def init_agent():
    from agents.router import ProductIntelligenceAgent
    if st.session_state.agent is None and api_key:
        st.session_state.agent = ProductIntelligenceAgent(api_key=api_key)
    if st.session_state.agent:
        st.session_state.agent.set_category_filter(selected_category)

if api_key:
    init_agent()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
INTENT_LABELS = {
    "INVENTORY": ("📦 INVENTORY", "pill-INVENTORY"),
    "PRICING":   ("💰 PRICING",   "pill-PRICING"),
    "REVIEWS":   ("⭐ REVIEWS",   "pill-REVIEWS"),
    "CATALOG":   ("🛍️ CATALOG",  "pill-CATALOG"),
    "GENERAL":   ("💬 GENERAL",   "pill-GENERAL"),
}

def render_intent(intent):
    label, cls = INTENT_LABELS.get(intent, (intent, "pill-GENERAL"))
    st.markdown(f'<span class="intent-pill {cls}">{label}</span>', unsafe_allow_html=True)

def render_tool_logs(tool_calls):
    if not tool_calls:
        return
    with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
        for i, tc in enumerate(tool_calls):
            args_str = ", ".join(f'{k}="{v}"' if isinstance(v,str) else f"{k}={v}" for k,v in tc["args"].items())
            st.markdown(
                f'<div class="tool-card">'
                f'<span class="tool-fn">{tc["name"]}</span>'
                f'<span class="tool-args">({args_str})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            try:
                st.json(json.loads(tc["result"]), expanded=False)
            except (json.JSONDecodeError, TypeError):
                st.code(str(tc["result"])[:500], language="json")
            if i < len(tool_calls) - 1:
                st.divider()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("## 🧠 RetailMind Product Intelligence")
st.caption("Real-time catalog analysis for StyleCraft — ask anything about inventory, pricing, reviews, or products.")

# ---------------------------------------------------------------------------
# Daily Briefing
# ---------------------------------------------------------------------------
if api_key and not st.session_state.briefing_shown and st.session_state.agent:
    with st.spinner("Generating Daily Briefing..."):
        try:
            briefing = st.session_state.agent.generate_daily_briefing()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**📋 Daily Briefing — StyleCraft**\n\n{briefing}",
                "intent": None,
                "tool_calls": [],
            })
            st.session_state.briefing_shown = True
        except Exception as e:
            st.error(f"Briefing error: {e}")
            st.session_state.briefing_shown = True

# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and show_logs and msg.get("intent"):
            render_intent(msg["intent"])
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_logs and msg.get("tool_calls"):
            render_tool_logs(msg["tool_calls"])

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to start.")
else:
    if prompt := st.chat_input("Ask about inventory, pricing, reviews, or products..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                try:
                    result = st.session_state.agent.process_query(prompt)
                    response = result["response"]
                    intent = result["intent"]
                    tool_calls = result["tool_calls"]

                    if show_logs:
                        render_intent(intent)
                    st.markdown(response)
                    if show_logs and tool_calls:
                        render_tool_logs(tool_calls)
                except Exception as e:
                    response = f"Error: {str(e)}"
                    intent = "ERROR"
                    tool_calls = []
                    st.error(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "intent": intent,
            "tool_calls": tool_calls,
        })
