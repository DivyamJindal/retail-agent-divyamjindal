"""
RetailMind Product Intelligence — Router Agent
===============================================
LLM-based intent classification + tool-calling agent with conversation memory.

Architecture:
1. Router: LLM classifies user intent → INVENTORY / PRICING / REVIEWS / CATALOG / GENERAL
2. Agent: LLM uses function calling to invoke the right tool(s)
3. Memory: Conversation history enables multi-turn follow-ups
"""

import json
from openai import OpenAI
from agents.tools import (
    TOOL_SCHEMAS,
    TOOL_MAP,
    set_openai_client,
    products_df,
    reviews_df,
)

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

# Router prompt: classifies intent into one of 5 categories
# Temperature 0 ensures deterministic, consistent classification
ROUTER_SYSTEM_PROMPT = """You are an intent classifier for RetailMind, a product intelligence system for StyleCraft — a D2C fashion brand with 30 SKUs across 5 categories (Tops, Dresses, Bottoms, Outerwear, Accessories).

Given a user query and optional conversation history, classify the intent into EXACTLY ONE of these categories:

INVENTORY — Questions about stock levels, stockout risk, restock needs, how long inventory will last, days of stock remaining, which products are running low.
PRICING — Questions about margins, pricing tiers, profitability, cost efficiency, gross margin, which products are overpriced or underpriced.
REVIEWS — Questions about customer feedback, ratings, complaints, what customers are saying, sentiment, product quality perceptions.
CATALOG — Questions about product search, category overviews, top performers, browsing the catalog, finding products, category comparisons.
GENERAL — Greetings, meta questions about the agent itself, general retail knowledge not tied to specific data queries.

IMPORTANT:
- If the query is a follow-up to a previous data question (e.g., "what about its pricing?" after an inventory query), classify based on the NEW intent, not the previous one.
- Respond with ONLY the category name in uppercase. No explanation, no punctuation."""


# Agent prompt: the main conversational agent that uses tools and formats responses
AGENT_SYSTEM_PROMPT = """You are RetailMind AI, a Product Intelligence Agent for StyleCraft — a D2C fashion brand.

Your role is to help StyleCraft's product manager (Priya) make data-driven decisions about inventory, pricing, and product performance. You have access to tools that query the product catalog (30 SKUs across Tops, Dresses, Bottoms, Outerwear, Accessories) and customer reviews.

GUIDELINES:
- Always use the available tools to answer data questions — never guess or make up numbers.
- All prices are in Indian Rupees (INR). Always use the ₹ symbol for currency, never $ or USD.
- Present data clearly with product IDs, names, and specific metrics.
- When a product has Critical or Low stock status, emphasise the urgency.
- When margin is below 20%, flag it as a concern and suggest corrective action.
- For review insights, highlight both positive and negative themes.
- If the user asks about a category, use get_category_performance for the overview.
- If the user asks to find products, use search_products.
- Be concise but thorough. Use bullet points and structured formatting.
- Maintain context from previous messages for follow-up questions.
- For general greetings or meta questions, respond warmly and explain your capabilities.

CATEGORY FILTER: {category_filter}
If a category filter is active (not "All Categories"), scope your analysis to that category when relevant. Mention the active filter in your response."""


# Briefing prompt: generates the daily startup briefing
BRIEFING_SYSTEM_PROMPT = """You are RetailMind AI generating a Daily Briefing for StyleCraft's product manager.

IMPORTANT: All prices and revenue figures are in Indian Rupees (INR). Always use the ₹ symbol, never $ or other currencies.

Format the briefing as a clear, actionable report with these EXACT sections:

🚨 CRITICAL STOCK ALERTS
List the top 3 most critically low-stock products with:
- Product ID and name
- Current stock and daily sales rate
- Estimated days to stockout
- Revenue at risk (in ₹)

⭐ PRODUCT QUALITY FLAG
Show the worst-rated product with:
- Product ID, name, and rating
- One-line summary of why customers are unhappy

💰 PRICING ALERT
Show the product with the lowest gross margin (if below 25%) with:
- Product ID, name, current margin
- Suggested action

Keep it concise and actionable. Use bullet points. Every number should come from the data provided."""


class ProductIntelligenceAgent:
    """Main agent: classifies intent, calls tools, maintains conversation memory."""

    def __init__(self, api_key: str):
        # Initialise OpenAI client
        self.client = OpenAI(api_key=api_key)

        # Inject client into tools module so get_review_insights can call the LLM
        set_openai_client(self.client)

        # Conversation memory — list of {"role": ..., "content": ...} dicts
        self.messages: list[dict] = []

        # Active category filter (set from Streamlit sidebar)
        self.category_filter: str = "All Categories"

    def classify_intent(self, query: str) -> str:
        """Use LLM to classify the user's intent into one of 5 categories.

        This is the Router Pattern — the LLM decides which domain the query
        belongs to, and the agent then uses the appropriate tools.

        Parameters:
            temperature=0  →  Deterministic classification, no randomness
            max_tokens=10  →  We only need a single word response
        """
        # Include recent conversation for context on follow-up queries
        context_messages = []
        if self.messages:
            # Last 4 messages (2 turns) for context
            context_messages = self.messages[-4:]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,       # Deterministic: same query always gets same classification
            max_tokens=10,       # Only need one word: INVENTORY/PRICING/REVIEWS/CATALOG/GENERAL
            top_p=1.0,           # No nucleus sampling needed at temperature 0
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                *context_messages,
                {"role": "user", "content": query},
            ],
        )

        intent = response.choices[0].message.content.strip().upper()

        # Validate — fall back to GENERAL if classification is unexpected
        valid_intents = {"INVENTORY", "PRICING", "REVIEWS", "CATALOG", "GENERAL"}
        if intent not in valid_intents:
            intent = "GENERAL"

        return intent

    def process_query(self, query: str) -> dict:
        """Main entry point: classify intent → dispatch to specialized handler.

        The Router Pattern works as follows:
        1. LLM classifies intent into one of 5 categories
        2. Based on the classified intent, the query is dispatched to a
           specialised handler that has access ONLY to the tools relevant
           to that intent domain
        3. The handler uses OpenAI function calling with its scoped tools

        Returns:
            dict with keys: response, intent, tool_calls (list of {name, args, result})
        """
        # Step 1: Router — classify intent via LLM
        intent = self.classify_intent(query)

        # Step 2: Dispatch to the specialised handler based on classified intent
        # Each handler receives only the tools relevant to its domain
        if intent == "GENERAL":
            return {
                "response": self._handle_general(query),
                "intent": intent,
                "tool_calls": [],
            }
        elif intent == "INVENTORY":
            response, tool_calls = self._handle_inventory(query)
        elif intent == "PRICING":
            response, tool_calls = self._handle_pricing(query)
        elif intent == "REVIEWS":
            response, tool_calls = self._handle_reviews(query)
        elif intent == "CATALOG":
            response, tool_calls = self._handle_catalog(query)
        else:
            response, tool_calls = self._call_with_tools(query, TOOL_SCHEMAS)

        # Step 3: Update conversation memory
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "intent": intent,
            "tool_calls": tool_calls,
        }

    # ----- Specialised intent handlers -----
    # Each handler scopes the available tools to its domain so the LLM
    # only sees relevant functions, improving accuracy and demonstrating
    # proper router-to-handler dispatch.

    def _handle_inventory(self, query: str) -> tuple[str, list[dict]]:
        """Handle INVENTORY intent: stock levels, stockout risk, restock needs."""
        inventory_schemas = [s for s in TOOL_SCHEMAS
                            if s["function"]["name"] in ("get_inventory_health", "generate_restock_alert")]
        return self._call_with_tools(query, inventory_schemas)

    def _handle_pricing(self, query: str) -> tuple[str, list[dict]]:
        """Handle PRICING intent: margins, positioning, profitability."""
        pricing_schemas = [s for s in TOOL_SCHEMAS
                          if s["function"]["name"] in ("get_pricing_analysis", "get_category_performance")]
        return self._call_with_tools(query, pricing_schemas)

    def _handle_reviews(self, query: str) -> tuple[str, list[dict]]:
        """Handle REVIEWS intent: customer feedback, ratings, sentiment."""
        review_schemas = [s for s in TOOL_SCHEMAS
                         if s["function"]["name"] in ("get_review_insights",)]
        return self._call_with_tools(query, review_schemas)

    def _handle_catalog(self, query: str) -> tuple[str, list[dict]]:
        """Handle CATALOG intent: product search, category overviews."""
        catalog_schemas = [s for s in TOOL_SCHEMAS
                          if s["function"]["name"] in ("search_products", "get_category_performance")]
        return self._call_with_tools(query, catalog_schemas)

    def _handle_general(self, query: str) -> str:
        """Handle greetings and meta questions using LLM knowledge."""
        # Include conversation history for context
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(
                category_filter=self.category_filter
            )},
            *self.messages[-6:],  # Recent context
            {"role": "user", "content": query},
        ]

        # Temperature 0.5: slightly creative for conversational responses
        # max_tokens 500: enough for a helpful explanation
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,    # Warmer for conversational, friendly responses
            max_tokens=500,     # Sufficient for greetings and explanations
            messages=messages,
        )

        result = response.choices[0].message.content

        # Update memory
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": result})

        return result

    def _call_with_tools(self, query: str, scoped_schemas: list = None) -> tuple[str, list[dict]]:
        """Use OpenAI function calling to invoke the right tool(s) and
        generate a natural language response from the results.

        The LLM receives only the scoped tool schemas (filtered by the
        router's intent classification) so it picks from the correct
        domain-specific tools.

        Parameters:
            query          →  The user's question
            scoped_schemas →  Tool schemas scoped to the classified intent
            temperature=0.1  →  Low for consistent, data-driven responses
            max_tokens=1000  →  Enough for detailed analysis with formatting

        Returns:
            (response_text, tool_calls_log) where tool_calls_log is a list of
            {"name": ..., "args": ..., "result": ...} dicts for UI display.
        """
        if scoped_schemas is None:
            scoped_schemas = TOOL_SCHEMAS

        tool_calls_log = []

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(
                category_filter=self.category_filter
            )},
            *self.messages[-6:],  # Recent conversation for follow-ups
            {"role": "user", "content": query},
        ]

        # First call: LLM decides which tool(s) to invoke from the scoped set
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,     # Very low: data analysis should be consistent
            max_tokens=1000,     # Room for detailed formatted responses
            top_p=0.95,          # Slight nucleus sampling for natural phrasing
            messages=messages,
            tools=scoped_schemas,
            tool_choice="auto",  # LLM decides whether and which tool to call
        )

        assistant_message = response.choices[0].message

        # If the LLM wants to call tool(s), execute them
        if assistant_message.tool_calls:
            # Add the assistant's tool-calling message to the conversation
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # Execute the tool function
                if fn_name in TOOL_MAP:
                    tool_result = TOOL_MAP[fn_name](**fn_args)
                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                # Log for UI display
                tool_calls_log.append({
                    "name": fn_name,
                    "args": fn_args,
                    "result": tool_result,
                })

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id,
                })

            # Second call: LLM generates a natural language response from tool results
            # Temperature slightly higher here for readable, well-formatted output
            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,    # Balanced: factual but naturally phrased
                max_tokens=1000,    # Detailed response with formatting
                messages=messages,
            )

            return final_response.choices[0].message.content, tool_calls_log

        # If no tool was called, return the direct response
        return (assistant_message.content or "I couldn't process that query. Please try rephrasing."), tool_calls_log

    def generate_daily_briefing(self) -> str:
        """Generate the startup Daily Briefing by calling tools and
        formatting results through the LLM.

        Called automatically when the app launches or chat is cleared.
        """
        # Gather data for the briefing using our tools directly
        restock_data = json.loads(
            TOOL_MAP["generate_restock_alert"](threshold_days=7)
        )

        # Find the worst-rated product
        worst = products_df.loc[products_df["avg_rating"].idxmin()]
        worst_id = worst["product_id"]
        worst_review_data = json.loads(TOOL_MAP["get_review_insights"](worst_id))

        # Find the product with the lowest gross margin
        products_df_copy = products_df.copy()
        products_df_copy["margin"] = (
            (products_df_copy["price"] - products_df_copy["cost"])
            / products_df_copy["price"]
            * 100
        )
        lowest_margin = products_df_copy.loc[products_df_copy["margin"].idxmin()]
        margin_data = json.loads(
            TOOL_MAP["get_pricing_analysis"](lowest_margin["product_id"])
        )

        # Build context for the briefing LLM call
        briefing_context = json.dumps({
            "restock_alerts": restock_data,
            "worst_rated_product": worst_review_data,
            "lowest_margin_product": margin_data,
        }, indent=2)

        # Generate briefing via LLM
        # Temperature 0.2: mostly factual with slight natural language variation
        # max_tokens 800: enough for a comprehensive but concise briefing
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,    # Factual briefing, minimal creativity
            max_tokens=800,     # Concise daily briefing
            messages=[
                {"role": "system", "content": BRIEFING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate today's Daily Briefing using this data:\n\n{briefing_context}"},
            ],
        )

        return response.choices[0].message.content

    def clear_memory(self):
        """Reset conversation memory. Called when user clicks 'Clear Chat'."""
        self.messages.clear()

    def set_category_filter(self, category: str):
        """Update the active category filter."""
        self.category_filter = category
