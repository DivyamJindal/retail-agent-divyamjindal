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

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailMind — Product Intelligence Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3;
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 8px;
        margin-bottom: 8px;
    }

    /* Tool call log styling */
    .tool-log {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
    }
    .tool-log-header {
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .tool-log-args {
        color: #8b949e;
    }
    .tool-log-result {
        color: #7ee787;
        margin-top: 4px;
    }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .intent-INVENTORY { background-color: #1f3a2e; color: #7ee787; border: 1px solid #2ea04370; }
    .intent-PRICING { background-color: #3b2e1a; color: #f0b946; border: 1px solid #d29922; }
    .intent-REVIEWS { background-color: #2a1f3b; color: #d2a8ff; border: 1px solid #8957e5; }
    .intent-CATALOG { background-color: #1a2b3b; color: #58a6ff; border: 1px solid #388bfd; }
    .intent-GENERAL { background-color: #272b33; color: #8b949e; border: 1px solid #484f58; }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #8b949e;
    }

    /* Divider */
    hr {
        border-color: #21262d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data (for sidebar metrics)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    products = pd.read_csv("retailmind_products.csv")
    reviews = pd.read_csv("retailmind_reviews.csv")
    return products, reviews

products_df, reviews_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 RetailMind AI")
    st.caption("Product Intelligence Agent for StyleCraft")

    st.divider()

    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key. Loaded from .env if available.",
    )

    if api_key:
        st.success("API Key configured", icon="✅")
    else:
        st.warning("Enter your OpenAI API key to start", icon="⚠️")

    st.divider()

    # Category Filter
    st.markdown("#### Category Filter")
    categories = ["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"]
    selected_category = st.selectbox(
        "Scope analysis to:",
        categories,
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    # Catalog Summary Panel
    st.markdown("#### 📊 Catalog Summary")

    if selected_category == "All Categories":
        summary_df = products_df
    else:
        summary_df = products_df[products_df["category"] == selected_category]

    total_skus = len(summary_df)

    critical_stock = 0
    for _, row in summary_df.iterrows():
        if row["avg_daily_sales"] > 0:
            days = row["stock_quantity"] / row["avg_daily_sales"]
            if days < 7:
                critical_stock += 1

    avg_margin = round(
        float(((summary_df["price"] - summary_df["cost"]) / summary_df["price"] * 100).mean()), 1
    )
    avg_rating = round(float(summary_df["avg_rating"].mean()), 1)

    col1, col2 = st.columns(2)
    col1.metric("Total SKUs", total_skus)
    col2.metric("Critical Stock", critical_stock)

    col3, col4 = st.columns(2)
    col3.metric("Avg Margin", f"{avg_margin}%")
    col4.metric("Avg Rating", f"{avg_rating} ⭐")

    if selected_category != "All Categories":
        st.caption(f"Showing metrics for **{selected_category}** only")

    st.divider()

    # Clear Chat
    if st.button("🗑️ Clear Chat & Re-Brief", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.briefing_shown = False
        if "agent" in st.session_state:
            st.session_state.agent.clear_memory()
        st.rerun()

    # Show/hide logs toggle
    st.divider()
    show_logs = st.toggle("Show Agent Logs", value=True, help="Display tool calls and routing info")

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
# Initialise Agent
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
# Helper: Render tool call logs
# ---------------------------------------------------------------------------
def render_tool_logs(tool_calls: list[dict]):
    """Render tool call logs as expandable sections."""
    if not tool_calls:
        return

    with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
        for i, tc in enumerate(tool_calls):
            fn_name = tc["name"]
            fn_args = tc["args"]
            fn_result = tc["result"]

            # Tool header
            st.markdown(f"**`{fn_name}()`**")

            # Arguments
            if fn_args:
                args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in fn_args.items())
                st.code(f"{fn_name}({args_str})", language="python")

            # Result (parsed and truncated for readability)
            try:
                result_data = json.loads(fn_result)
                # Show a compact version
                st.json(result_data, expanded=False)
            except (json.JSONDecodeError, TypeError):
                st.code(str(fn_result)[:500], language="json")

            if i < len(tool_calls) - 1:
                st.divider()


def render_intent_badge(intent: str):
    """Render a coloured intent classification badge."""
    labels = {
        "INVENTORY": "📦 INVENTORY",
        "PRICING": "💰 PRICING",
        "REVIEWS": "⭐ REVIEWS",
        "CATALOG": "🛍️ CATALOG",
        "GENERAL": "💬 GENERAL",
    }
    label = labels.get(intent, intent)
    st.markdown(
        f'<span class="intent-badge intent-{intent}">{label}</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------
st.markdown("## 🧠 RetailMind Product Intelligence Agent")
st.caption("Ask about inventory, pricing, customer reviews, or catalog performance.")

# ---------------------------------------------------------------------------
# Daily Briefing
# ---------------------------------------------------------------------------
if api_key and not st.session_state.briefing_shown and st.session_state.agent:
    with st.spinner("Generating Daily Briefing..."):
        try:
            briefing = st.session_state.agent.generate_daily_briefing()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**📋 Daily Briefing — StyleCraft Product Intelligence**\n\n{briefing}",
                "intent": None,
                "tool_calls": [],
            })
            st.session_state.briefing_shown = True
        except Exception as e:
            st.error(f"Error generating briefing: {e}")
            st.session_state.briefing_shown = True

# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        # Show intent badge for assistant messages (if available and logs enabled)
        if msg["role"] == "assistant" and show_logs and msg.get("intent"):
            render_intent_badge(msg["intent"])

        st.markdown(msg["content"])

        # Show tool call logs (if available and logs enabled)
        if msg["role"] == "assistant" and show_logs and msg.get("tool_calls"):
            render_tool_logs(msg["tool_calls"])

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to start.")
else:
    if prompt := st.chat_input("Ask about inventory, pricing, reviews, or products..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
        })

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                try:
                    result = st.session_state.agent.process_query(prompt)

                    response = result["response"]
                    intent = result["intent"]
                    tool_calls = result["tool_calls"]

                    # Show intent badge
                    if show_logs:
                        render_intent_badge(intent)

                    # Show response
                    st.markdown(response)

                    # Show tool call logs
                    if show_logs and tool_calls:
                        render_tool_logs(tool_calls)

                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}"
                    intent = "ERROR"
                    tool_calls = []
                    st.error(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "intent": intent,
            "tool_calls": tool_calls,
        })
