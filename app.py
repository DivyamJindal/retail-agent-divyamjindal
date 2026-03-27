"""
RetailMind Product Intelligence Agent — Streamlit UI
=====================================================
A conversational AI agent for StyleCraft's product manager to analyse
inventory, pricing, reviews, and catalog performance in real time.

Launch:  python run.py  OR  streamlit run app.py
"""

import os
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
# Load Data (for sidebar metrics — tools.py handles its own loading)
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
    st.title("🧠 RetailMind AI")
    st.caption("Product Intelligence Agent for StyleCraft")

    st.divider()

    # API Key Configuration
    api_key = st.text_input(
        "🔑 OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key. Loaded from .env if available.",
    )

    if api_key:
        st.success("API Key configured", icon="✅")
    else:
        st.warning("Please enter your OpenAI API key", icon="⚠️")

    st.divider()

    # Category Filter
    st.subheader("🏷️ Category Filter")
    categories = ["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"]
    selected_category = st.selectbox(
        "Scope analysis to category:",
        categories,
        index=0,
        help="Filter the agent's analysis to a specific product category.",
    )

    st.divider()

    # Always-visible Catalog Summary Panel
    st.subheader("📊 Catalog Summary")

    # Compute metrics (optionally filtered by category)
    if selected_category == "All Categories":
        summary_df = products_df
    else:
        summary_df = products_df[products_df["category"] == selected_category]

    total_skus = len(summary_df)

    # Count critical stock items
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
    col2.metric("Critical Stock", critical_stock, delta=None)

    col3, col4 = st.columns(2)
    col3.metric("Avg Margin", f"{avg_margin}%")
    col4.metric("Avg Rating", f"{avg_rating} ⭐")

    st.divider()

    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.briefing_shown = False
        if "agent" in st.session_state:
            st.session_state.agent.clear_memory()
        st.rerun()

# ---------------------------------------------------------------------------
# Initialise Session State
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
    """Create or update the agent with the current API key and category filter."""
    from agents.router import ProductIntelligenceAgent

    if st.session_state.agent is None and api_key:
        st.session_state.agent = ProductIntelligenceAgent(api_key=api_key)

    if st.session_state.agent:
        st.session_state.agent.set_category_filter(selected_category)

if api_key:
    init_agent()

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.title("🧠 RetailMind Product Intelligence Agent")
st.caption(
    "AI-powered assistant for StyleCraft's product catalog — "
    "inventory, pricing, reviews, and merchandising insights."
)

# ---------------------------------------------------------------------------
# Daily Briefing (auto-generated on startup before user types anything)
# ---------------------------------------------------------------------------
if api_key and not st.session_state.briefing_shown and st.session_state.agent:
    with st.spinner("🔍 Generating Daily Briefing..."):
        try:
            briefing = st.session_state.agent.generate_daily_briefing()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"📋 **Daily Briefing — StyleCraft Product Intelligence**\n\n{briefing}",
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
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
if not api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to start chatting.")
else:
    if prompt := st.chat_input("Ask about inventory, pricing, reviews, or products..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                try:
                    response, intent = st.session_state.agent.process_query(prompt)

                    # Show classified intent as a subtle badge
                    intent_badges = {
                        "INVENTORY": "📦 Inventory",
                        "PRICING": "💰 Pricing",
                        "REVIEWS": "⭐ Reviews",
                        "CATALOG": "🛍️ Catalog",
                        "GENERAL": "💬 General",
                    }
                    badge = intent_badges.get(intent, intent)
                    st.caption(f"Routed to: {badge}")

                    st.markdown(response)

                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}"
                    intent = "ERROR"
                    st.error(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
