# RetailMind Product Intelligence Agent

## Overview
An AI-powered Product Intelligence Agent for **StyleCraft**, a D2C fashion brand with 30 SKUs across 5 categories. The agent answers natural language questions about inventory, pricing, customer reviews, and catalog performance — replacing 4-5 hours of manual weekly analysis with real-time conversational insights.

**Roll Number:** UGDSAI25015

## Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│   LLM Router     │ ← Classifies intent via GPT-4o-mini (not keyword matching)
│  (Intent Class.) │
└────────┬─────────┘
         │
    ┌────┴────┬──────────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼          ▼
INVENTORY  PRICING   REVIEWS    CATALOG    GENERAL
    │         │          │          │          │
    ▼         ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────┐
│          OpenAI Function Calling                 │
│   (LLM selects & invokes the right tool)         │
└────────────────────┬────────────────────────────┘
                     │
    ┌────────────────┼────────────────────┐
    ▼                ▼                    ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│6 Tool Fns│  │ Conversation │  │   Streamlit  │
│(Data Ops)│  │   Memory     │  │     UI       │
└──────────┘  └──────────────┘  └──────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Router** | LLM-based intent classifier (5 intents: INVENTORY, PRICING, REVIEWS, CATALOG, GENERAL) |
| **6 Tools** | `search_products`, `get_inventory_health`, `get_pricing_analysis`, `get_review_insights`, `get_category_performance`, `generate_restock_alert` |
| **Daily Briefing** | Auto-generated on startup: top 3 stockout risks, worst-rated product, lowest margin alert |
| **Conversation Memory** | Multi-turn context for follow-up questions |
| **Streamlit UI** | Chat interface with category filter, catalog summary panel, and clear chat |

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

```bash
git clone https://github.com/DivyamJindal/ai-agent-exam.git
cd ai-agent-exam
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
```

### Run

```bash
python run.py
```

Or alternatively:
```bash
python start.py
```

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini (via OpenAI SDK)
- **Agent Framework**: OpenAI Function Calling with LLM-based Router Pattern
- **UI**: Streamlit
- **Data**: Pandas

## LLM Parameters

| Use Case | Temperature | max_tokens | Reasoning |
|----------|------------|------------|-----------|
| Intent Classification | 0.0 | 10 | Deterministic routing — same query always gets same classification |
| Data Analysis Responses | 0.1 | 1000 | Consistent, data-driven answers with detailed formatting |
| Review Summarisation | 0.3 | 300 | Slightly creative for natural language summaries |
| General Conversation | 0.5 | 500 | Warmer, more conversational tone |
| Daily Briefing | 0.2 | 800 | Factual but readable report format |

## Dataset

- `retailmind_products.csv` — 30 products across Tops, Dresses, Bottoms, Outerwear, Accessories
- `retailmind_reviews.csv` — 40 customer reviews with ratings, titles, and detailed text
