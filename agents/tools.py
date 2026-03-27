"""
RetailMind Product Intelligence — Tool Functions
=================================================
Six tool functions callable by the LLM via OpenAI function calling.
Each tool operates on the product catalog and reviews data,
returning structured JSON that the agent formats into natural language.
"""

import os
import json
import pandas as pd
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
# Load CSVs from project root (relative path — run.py ensures correct cwd)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
products_df = pd.read_csv(os.path.join(_root, "retailmind_products.csv"))
reviews_df = pd.read_csv(os.path.join(_root, "retailmind_reviews.csv"))

# Cache for LLM-generated review summaries (avoids repeated API calls)
_review_cache: dict = {}

# Reference to OpenAI client — set by router.py at init time
_openai_client = None


def set_openai_client(client):
    """Inject the OpenAI client so get_review_insights can call the LLM."""
    global _openai_client
    _openai_client = client


# ---------------------------------------------------------------------------
# Tool 1: search_products
# ---------------------------------------------------------------------------
def search_products(query: str, category: str = None) -> str:
    """Search and return matching products from the CSV based on a text query
    and optional category filter. Uses fuzzy string matching on product names.
    Returns top 5 matches as a list of dicts.
    """
    df = products_df.copy()

    # Apply category filter if provided
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    if df.empty:
        return json.dumps({"matches": [], "message": f"No products found in category '{category}'."})

    # Fuzzy match score for each product against the query
    def _score(name):
        # Check both full match and substring containment
        name_lower = name.lower()
        query_lower = query.lower()
        # Exact substring bonus
        if query_lower in name_lower:
            return 1.0
        return SequenceMatcher(None, query_lower, name_lower).ratio()

    df = df.copy()
    df["_score"] = df["product_name"].apply(_score)
    top = df.nlargest(5, "_score")

    matches = []
    for _, row in top.iterrows():
        matches.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "price": float(row["price"]),
            "stock_quantity": int(row["stock_quantity"]),
            "avg_rating": float(row["avg_rating"]),
            "match_score": round(row["_score"], 2),
        })

    return json.dumps({"query": query, "category_filter": category, "matches": matches})


# ---------------------------------------------------------------------------
# Tool 2: get_inventory_health
# ---------------------------------------------------------------------------
def get_inventory_health(product_id: str) -> str:
    """Return inventory status for a product: current stock, average daily sales,
    estimated days to stockout, and a status flag —
    Critical (<7 days), Low (7-14 days), or Healthy (>14 days).
    """
    product_id = product_id.upper()
    row = products_df[products_df["product_id"] == product_id]

    if row.empty:
        return json.dumps({"error": f"Product {product_id} not found."})

    row = row.iloc[0]
    stock = int(row["stock_quantity"])
    daily_sales = float(row["avg_daily_sales"])
    reorder_level = int(row["reorder_level"])

    # Handle division by zero — if no sales, product won't stock out
    if daily_sales == 0:
        days_to_stockout = float("inf")
        status = "No Sales (Overstocked)"
    else:
        days_to_stockout = round(stock / daily_sales, 1)
        if days_to_stockout < 7:
            status = "Critical"
        elif days_to_stockout < 14:
            status = "Low"
        else:
            status = "Healthy"

    needs_reorder = stock <= reorder_level

    return json.dumps({
        "product_id": product_id,
        "product_name": row["product_name"],
        "stock_quantity": stock,
        "avg_daily_sales": daily_sales,
        "days_to_stockout": days_to_stockout if days_to_stockout != float("inf") else "N/A (no sales)",
        "status": status,
        "reorder_level": reorder_level,
        "needs_reorder": needs_reorder,
    })


# ---------------------------------------------------------------------------
# Tool 3: get_pricing_analysis
# ---------------------------------------------------------------------------
def get_pricing_analysis(product_id: str) -> str:
    """Return pricing intelligence: gross margin %, price positioning
    (Premium / Mid-Range / Budget based on category averages),
    and a flag if margin is below 20%.
    """
    product_id = product_id.upper()
    row = products_df[products_df["product_id"] == product_id]

    if row.empty:
        return json.dumps({"error": f"Product {product_id} not found."})

    row = row.iloc[0]
    price = float(row["price"])
    cost = float(row["cost"])
    category = row["category"]

    # Gross margin calculation
    gross_margin = round((price - cost) / price * 100, 2)
    low_margin_flag = gross_margin < 20

    # Category average price for positioning
    cat_avg = float(products_df[products_df["category"] == category]["price"].mean())

    if price > cat_avg * 1.2:
        positioning = "Premium"
    elif price < cat_avg * 0.8:
        positioning = "Budget"
    else:
        positioning = "Mid-Range"

    result = {
        "product_id": product_id,
        "product_name": row["product_name"],
        "category": category,
        "selling_price": price,
        "cost_price": cost,
        "gross_margin_pct": gross_margin,
        "category_avg_price": round(cat_avg, 2),
        "price_positioning": positioning,
        "low_margin_alert": low_margin_flag,
    }

    if low_margin_flag:
        # Suggest a minimum price to achieve 20% margin
        min_price = round(cost / 0.80, 2)
        result["suggested_min_price"] = min_price
        result["alert_message"] = (
            f"Margin ({gross_margin}%) is below 20%. "
            f"Consider raising price to at least ₹{min_price}."
        )

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 4: get_review_insights
# ---------------------------------------------------------------------------
def get_review_insights(product_id: str) -> str:
    """Summarise customer reviews for a product using an LLM.
    Returns: average rating, total reviews, a 2-sentence sentiment summary,
    and top 2 recurring themes (positive and negative).
    """
    product_id = product_id.upper()

    # Return cached result if available
    if product_id in _review_cache:
        return json.dumps(_review_cache[product_id])

    product_row = products_df[products_df["product_id"] == product_id]
    if product_row.empty:
        return json.dumps({"error": f"Product {product_id} not found."})

    product_name = product_row.iloc[0]["product_name"]

    # Filter reviews for this product
    revs = reviews_df[reviews_df["product_id"] == product_id]

    if revs.empty:
        result = {
            "product_id": product_id,
            "product_name": product_name,
            "avg_rating": float(product_row.iloc[0]["avg_rating"]),
            "total_reviews": 0,
            "sentiment_summary": "No customer reviews available for this product yet.",
            "positive_themes": [],
            "negative_themes": [],
        }
        _review_cache[product_id] = result
        return json.dumps(result)

    avg_rating = round(float(revs["rating"].mean()), 1)
    total_reviews = len(revs)

    # Build review text block for LLM summarisation
    review_block = "\n".join(
        f"- [{row['rating']}/5] {row['review_title']}: {row['review_text']}"
        for _, row in revs.iterrows()
    )

    # Call LLM for summarisation
    if _openai_client is None:
        # Fallback if client not injected — shouldn't happen in normal flow
        result = {
            "product_id": product_id,
            "product_name": product_name,
            "avg_rating": avg_rating,
            "total_reviews": total_reviews,
            "sentiment_summary": "LLM client not available for summarisation.",
            "positive_themes": [],
            "negative_themes": [],
        }
        _review_cache[product_id] = result
        return json.dumps(result)

    # LLM call for review summarisation
    # Temperature 0.3: slightly creative for natural language but mostly factual
    # max_tokens 300: enough for summary + themes, keeps responses concise
    llm_response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,  # Low-ish for factual summaries, slight variation for naturalness
        max_tokens=300,    # Sufficient for 2-sentence summary + 4 themes
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a product review analyst. Analyse the reviews below and respond "
                    "in EXACTLY this JSON format (no extra text):\n"
                    '{"sentiment_summary": "Two sentences summarising overall customer sentiment.",'
                    ' "positive_themes": ["theme1", "theme2"],'
                    ' "negative_themes": ["theme1", "theme2"]}'
                ),
            },
            {
                "role": "user",
                "content": f"Product: {product_name}\n\nCustomer Reviews:\n{review_block}",
            },
        ],
    )

    try:
        llm_data = json.loads(llm_response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        llm_data = {
            "sentiment_summary": llm_response.choices[0].message.content,
            "positive_themes": [],
            "negative_themes": [],
        }

    result = {
        "product_id": product_id,
        "product_name": product_name,
        "avg_rating": avg_rating,
        "total_reviews": total_reviews,
        "sentiment_summary": llm_data.get("sentiment_summary", ""),
        "positive_themes": llm_data.get("positive_themes", []),
        "negative_themes": llm_data.get("negative_themes", []),
    }

    _review_cache[product_id] = result
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 5: get_category_performance
# ---------------------------------------------------------------------------
def get_category_performance(category: str) -> str:
    """Return aggregated category-level metrics: total SKUs, average rating,
    average margin %, total stock units, number of low/critical stock items,
    and the top 3 revenue-generating products (price x avg_daily_sales).
    """
    cat_df = products_df[products_df["category"].str.lower() == category.lower()]

    if cat_df.empty:
        return json.dumps({"error": f"Category '{category}' not found. Valid: Tops, Dresses, Bottoms, Outerwear, Accessories."})

    total_skus = len(cat_df)
    avg_rating = round(float(cat_df["avg_rating"].mean()), 2)
    avg_margin = round(float(((cat_df["price"] - cat_df["cost"]) / cat_df["price"] * 100).mean()), 2)
    total_stock = int(cat_df["stock_quantity"].sum())

    # Count low/critical stock items
    critical_count = 0
    low_count = 0
    for _, row in cat_df.iterrows():
        if row["avg_daily_sales"] > 0:
            days = row["stock_quantity"] / row["avg_daily_sales"]
            if days < 7:
                critical_count += 1
            elif days < 14:
                low_count += 1

    # Top 3 revenue products: estimated daily revenue = price x avg_daily_sales
    cat_df = cat_df.copy()
    cat_df["est_daily_revenue"] = cat_df["price"] * cat_df["avg_daily_sales"]
    top3 = cat_df.nlargest(3, "est_daily_revenue")

    top_products = []
    for _, row in top3.iterrows():
        top_products.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "price": float(row["price"]),
            "avg_daily_sales": float(row["avg_daily_sales"]),
            "est_daily_revenue": round(float(row["est_daily_revenue"]), 2),
        })

    return json.dumps({
        "category": category,
        "total_skus": total_skus,
        "avg_rating": avg_rating,
        "avg_margin_pct": avg_margin,
        "total_stock_units": total_stock,
        "critical_stock_items": critical_count,
        "low_stock_items": low_count,
        "top_3_revenue_products": top_products,
    })


# ---------------------------------------------------------------------------
# Tool 6: generate_restock_alert
# ---------------------------------------------------------------------------
def generate_restock_alert(threshold_days: int = 7) -> str:
    """Scan all products and return those at risk of stockout within
    the specified number of days, sorted by urgency (fewest days first).
    Includes estimated revenue at risk.
    """
    alerts = []

    for _, row in products_df.iterrows():
        daily_sales = float(row["avg_daily_sales"])
        stock = int(row["stock_quantity"])
        price = float(row["price"])

        if daily_sales == 0:
            continue  # No sales = no stockout risk

        days_to_stockout = round(stock / daily_sales, 1)

        if days_to_stockout <= threshold_days:
            # Revenue at risk formula from assignment spec
            revenue_at_risk = round(price * (stock + daily_sales * threshold_days), 2)

            alerts.append({
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "stock_quantity": stock,
                "avg_daily_sales": daily_sales,
                "days_to_stockout": days_to_stockout,
                "revenue_at_risk": revenue_at_risk,
            })

    # Sort by urgency — fewest days remaining first
    alerts.sort(key=lambda x: x["days_to_stockout"])

    return json.dumps({
        "threshold_days": threshold_days,
        "total_at_risk": len(alerts),
        "alerts": alerts,
    })


# ---------------------------------------------------------------------------
# OpenAI Function Calling Schemas
# ---------------------------------------------------------------------------
# These schemas tell the LLM what each tool does, its parameters,
# and when to use it. The LLM uses these to decide which tool to call.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search and return matching products from the catalog based on a text query "
                "and optional category filter. Returns product ID, name, category, price, "
                "stock quantity, and rating for the top 5 matches. Use this for product "
                "discovery, finding specific items, or browsing the catalog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to match against product names (e.g., 'summer dress', 'jeans', 'blazer')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter to narrow search results",
                        "enum": ["Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_health",
            "description": (
                "Return inventory health status for a specific product: current stock, "
                "average daily sales, estimated days to stockout, and a status flag "
                "(Critical/Low/Healthy). Use when asked about stock levels, inventory "
                "status, or stockout risk for a particular product."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID (e.g., 'SC001', 'SC015')",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pricing_analysis",
            "description": (
                "Return pricing intelligence for a specific product: gross margin percentage, "
                "price positioning (Premium/Mid-Range/Budget relative to category average), "
                "and an alert if margin is below 20%. Use when asked about margins, pricing, "
                "profitability, or cost analysis for a particular product."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID (e.g., 'SC001', 'SC015')",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_review_insights",
            "description": (
                "Return an AI-generated summary of customer reviews for a specific product: "
                "average rating, total reviews, a 2-sentence sentiment summary, and top 2 "
                "recurring positive and negative themes. Use when asked about customer feedback, "
                "ratings, complaints, or what customers are saying about a product."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID (e.g., 'SC001', 'SC015')",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_performance",
            "description": (
                "Return aggregated performance metrics for a product category: total SKUs, "
                "average rating, average margin %, total stock, count of low/critical stock "
                "items, and top 3 revenue-generating products. Use when asked about category "
                "overviews, category performance, or comparing categories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category to analyse",
                        "enum": ["Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"],
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_restock_alert",
            "description": (
                "Scan all products and return those at risk of stocking out within the specified "
                "number of days, sorted by urgency. Includes estimated revenue at risk. "
                "Use when asked about restock needs, stockout alerts, or which products "
                "need immediate inventory attention."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold_days": {
                        "type": "integer",
                        "description": "Number of days to look ahead for stockout risk (default: 7)",
                        "default": 7,
                    },
                },
                "required": [],
            },
        },
    },
]

# Map function names to callable references — used by the agent to dispatch
TOOL_MAP = {
    "search_products": search_products,
    "get_inventory_health": get_inventory_health,
    "get_pricing_analysis": get_pricing_analysis,
    "get_review_insights": get_review_insights,
    "get_category_performance": get_category_performance,
    "generate_restock_alert": generate_restock_alert,
}
