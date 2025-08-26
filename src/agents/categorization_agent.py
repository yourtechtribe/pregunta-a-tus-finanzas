"""
Multi-Agent Categorization System using LangGraph - Simplified Version
Focused only on resolving "Other" categories through web search and LLM
BBVAExtractor handles all rule-based categorization
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from pathlib import Path
import uuid

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig


# ==================== State Definition ====================

class TransactionCategorizationState(TypedDict):
    """State for the categorization workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    transaction: Dict[str, Any]
    merchant_name: str
    amount: float
    date: str
    search_results: Optional[List[Dict[str, Any]]]
    categorization: Optional[Dict[str, Any]]
    confidence: float
    method_used: str
    final_category: Optional[str]
    memory_updated: bool


# ==================== Memory Store ====================

class MerchantMemory:
    """Persistent memory for merchant categorizations"""
    
    def __init__(self, cache_file: str = "data/merchant_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file if exists"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Could not save cache: {e}")
    
    def remember_merchant(self, name: str, info: Dict):
        """Store merchant information"""
        key = name.lower().strip()
        self.cache[key] = {
            "name": name,
            "category": info["category"],
            "confidence": info.get("confidence", 0.8),
            "business_type": info.get("business_type"),
            "location": info.get("location"),
            "last_updated": datetime.now().isoformat(),
            "frequency": self.cache.get(key, {}).get("frequency", 0) + 1
        }
        self._save_cache()
    
    def get_merchant(self, name: str) -> Optional[Dict]:
        """Retrieve merchant information"""
        key = name.lower().strip()
        return self.cache.get(key)
    
    def is_recent(self, merchant_info: Dict, days: int = 30) -> bool:
        """Check if cached info is recent enough"""
        if not merchant_info or "last_updated" not in merchant_info:
            return False
        last_updated = datetime.fromisoformat(merchant_info["last_updated"])
        age_days = (datetime.now() - last_updated).days
        return age_days < days


# ==================== Agent Nodes ====================

async def check_memory_node(state: TransactionCategorizationState, config: RunnableConfig) -> Dict:
    """
    Check if merchant exists in memory cache
    """
    merchant_name = state.get("merchant_name", "")
    
    # Initialize memory
    memory = MerchantMemory(config.get("configurable", {}).get("cache_file", "data/merchant_cache.json"))
    merchant_info = memory.get_merchant(merchant_name)
    
    if merchant_info and memory.is_recent(merchant_info):
        # Found in cache and recent
        return {
            "method_used": "memory_cache",
            "categorization": merchant_info,
            "confidence": merchant_info["confidence"],
            "final_category": merchant_info["category"],
            "memory_updated": False,
            "messages": [HumanMessage(content=f"Found in cache: {merchant_info['category']}")],
        }
    
    # Not in cache or outdated - need to research
    return {
        "method_used": "needs_research",
        "messages": [HumanMessage(content=f"Not in cache, researching: {merchant_name}")],
    }


async def research_node(state: TransactionCategorizationState, config: RunnableConfig) -> Dict:
    """
    Research unknown merchants using web search and LLM
    """
    merchant_name = state.get("merchant_name", "")
    amount = state.get("amount", 0)
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Try web search first
    search_results = []
    try:
        search_tool = TavilySearchResults(
            max_results=3,
            api_key=os.getenv("TAVILY_API_KEY")
        )
        
        # Search query optimized for Spanish/Catalan context
        # Try to get location information
        query = f'"{merchant_name}" Barcelona Catalonia Spain business location empresa donde'
        results = search_tool.invoke(query)
        
        if isinstance(results, list):
            search_results = results
        elif isinstance(results, dict):
            search_results = results.get("results", [])
    except Exception as e:
        print(f"Search error: {e}")
    
    # Format search results for LLM
    search_context = ""
    if search_results:
        for i, result in enumerate(search_results[:3], 1):
            if isinstance(result, dict):
                search_context += f"\nResult {i}:\n"
                search_context += f"Title: {result.get('title', '')}\n"
                search_context += f"Content: {result.get('content', '')[:200]}\n"
    
    # Analyze with LLM
    prompt = f"""
    You are a financial transaction categorizer for Spanish bank statements.
    
    MERCHANT: {merchant_name}
    AMOUNT: {amount}€
    
    WEB SEARCH RESULTS:
    {search_context if search_context else "No web results available"}
    
    Analyze the merchant name and amount. Common patterns:
    - Positive amounts are usually Income or refunds
    - "TRANSFERENCIA" can be Transfers or Internal Transfer
    - Numbers like "16307" are often parking meters (Transportation)
    - "TRASPASO" is usually Internal Transfer between own accounts
    - "CACENCA" is a gas station chain in Catalonia
    
    Categories (choose ONE):
    Income, Savings, Taxes, Transfers, Internal Transfer, Donations, Loan, ATM,
    Groceries, Food & Dining, Transportation, Shopping, Entertainment, Healthcare,
    Utilities, Services, Education, Tech & Software, Sports, Vending, Housing, Fees, Other
    
    Output ONLY JSON:
    {{
        "category": "exact category name",
        "confidence": 0.0-1.0,
        "business_type": "type of business",
        "location": "city/region if identifiable from name or search results (null if unknown)",
        "reasoning": "brief explanation"
    }}
    """
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Clean response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        analysis = json.loads(content.strip())
        
        # Validate category
        valid_categories = [
            "Income", "Savings", "Taxes", "Transfers", "Internal Transfer",
            "Donations", "Loan", "ATM", "Groceries", "Food & Dining",
            "Transportation", "Shopping", "Entertainment", "Healthcare",
            "Utilities", "Services", "Education", "Tech & Software",
            "Sports", "Vending", "Housing", "Fees", "Other"
        ]
        
        if analysis.get("category") not in valid_categories:
            analysis["category"] = "Other"
            analysis["confidence"] = 0.3
        
    except Exception as e:
        print(f"LLM error: {e}")
        # Fallback categorization
        analysis = {
            "category": "Other",
            "confidence": 0.2,
            "business_type": "Unknown",
            "reasoning": "Could not determine category"
        }
    
    # Save to memory if confidence is high enough
    if analysis["confidence"] > 0.6:
        memory = MerchantMemory(config.get("configurable", {}).get("cache_file", "data/merchant_cache.json"))
        memory.remember_merchant(merchant_name, analysis)
        memory_updated = True
    else:
        memory_updated = False
    
    return {
        "search_results": search_results,
        "categorization": analysis,
        "confidence": analysis["confidence"],
        "method_used": "web_research_llm",
        "final_category": analysis["category"],
        "memory_updated": memory_updated,
        "messages": [HumanMessage(content=f"Categorized as {analysis['category']} ({analysis['confidence']:.0%})")],
    }


async def validation_node(state: TransactionCategorizationState) -> Dict:
    """
    Validate and adjust confidence based on amount patterns
    """
    category = state.get("final_category", "Other")
    amount = state.get("amount", 0)
    confidence = state.get("confidence", 0.5)
    merchant_name = state.get("merchant_name", "").lower()
    
    # Simple validation rules
    adjustments = []
    
    # Income should be positive
    if category == "Income" and amount < 0:
        confidence *= 0.7
        adjustments.append("Income usually positive")
    
    # Certain categories typically negative
    if category in ["Savings", "Taxes", "Loan"] and amount > 0:
        confidence *= 0.8
        adjustments.append(f"{category} usually negative")
    
    # Internal transfers detection
    if "traspaso" in merchant_name and category != "Internal Transfer":
        category = "Internal Transfer"
        confidence = 0.9
        adjustments.append("Detected as internal transfer")
    
    return {
        "final_category": category,
        "confidence": min(confidence, 0.95),  # Cap at 95%
        "validation_adjustments": adjustments
    }


# ==================== Graph Construction ====================

def create_categorization_graph():
    """
    Create simplified LangGraph workflow for transaction categorization
    """
    workflow = StateGraph(TransactionCategorizationState)
    
    # Add nodes
    workflow.add_node("check_memory", check_memory_node)
    workflow.add_node("research", research_node)
    workflow.add_node("validation", validation_node)
    
    # Define routing
    def route_after_memory(state: TransactionCategorizationState) -> str:
        if state.get("method_used") == "memory_cache":
            return "validation"
        return "research"
    
    # Add edges
    workflow.add_edge(START, "check_memory")
    workflow.add_conditional_edges(
        "check_memory",
        route_after_memory,
        {
            "validation": "validation",
            "research": "research"
        }
    )
    workflow.add_edge("research", "validation")
    workflow.add_edge("validation", END)
    
    # Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==================== Main Integration ====================

class SimplifiedMultiAgentCategorizer:
    """
    Simplified multi-agent categorizer for "Other" transactions only
    """
    
    def __init__(self, cache_file: str = "data/merchant_cache.json"):
        self.graph = create_categorization_graph()
        self.cache_file = cache_file
        self.stats = {
            "total": 0,
            "memory_cache": 0,
            "web_research_llm": 0,
            "failed": 0
        }
    
    async def categorize_transaction(self, transaction: Dict) -> Dict:
        """
        Categorize a single transaction marked as "Other"
        """
        # Only process if category is "Other"
        if transaction.get("category") not in ["Other", None, ""]:
            return {
                "transaction": transaction,
                "category": transaction["category"],
                "confidence": 1.0,
                "method": "already_categorized"
            }
        
        # Prepare state
        initial_state = {
            "transaction": transaction,
            "merchant_name": transaction.get("description", ""),
            "amount": transaction.get("amount", 0),
            "date": transaction.get("date", ""),
            "messages": []
        }
        
        # Run the graph
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "cache_file": self.cache_file
            }
        }
        
        try:
            result = await self.graph.ainvoke(initial_state, config)
            
            # Update stats
            method = result.get("method_used", "unknown")
            self.stats[method] = self.stats.get(method, 0) + 1
            self.stats["total"] += 1
            
            return {
                "transaction": transaction,
                "category": result.get("final_category", "Other"),
                "confidence": result.get("confidence", 0),
                "method": method,
                "memory_updated": result.get("memory_updated", False)
            }
        
        except Exception as e:
            print(f"Error categorizing: {e}")
            self.stats["failed"] += 1
            self.stats["total"] += 1
            
            return {
                "transaction": transaction,
                "category": "Other",
                "confidence": 0,
                "method": "error",
                "error": str(e)
            }
    
    async def process_others_only(self, transactions: List[Dict]) -> List[Dict]:
        """
        Process only transactions marked as "Other"
        """
        # Filter only "Other" transactions
        others = [t for t in transactions if t.get("category") in ["Other", None, ""]]
        
        if not others:
            print("✅ No 'Other' transactions to process")
            return transactions
        
        print(f"🤖 Processing {len(others)} 'Other' transactions...")
        print("=" * 60)
        
        # Process each "Other" transaction
        for i, transaction in enumerate(others, 1):
            print(f"\n[{i}/{len(others)}] {transaction.get('description', 'Unknown')}")
            
            result = await self.categorize_transaction(transaction)
            
            # Update original transaction
            if result["category"] != "Other" and result["confidence"] > 0.5:
                transaction["category"] = result["category"]
                transaction["category_confidence"] = result["confidence"]
                transaction["category_method"] = f"multiagent_{result['method']}"
                
                print(f"  ✓ Categorized as: {result['category']} ({result['confidence']:.0%})")
                if result.get("memory_updated"):
                    print(f"  ✓ Saved to cache for future use")
            else:
                print(f"  ✗ Could not categorize (confidence: {result['confidence']:.0%})")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Processing Summary:")
        print(f"  Total processed: {self.stats['total']}")
        print(f"  From cache: {self.stats.get('memory_cache', 0)}")
        print(f"  Web research + LLM: {self.stats.get('web_research_llm', 0)}")
        print(f"  Failed: {self.stats.get('failed', 0)}")
        
        return transactions
    
    def process_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Synchronous wrapper for pipeline integration
        """
        return asyncio.run(self.process_others_only(transactions))


# ==================== Example Usage ====================

async def main():
    """Example usage"""
    test_transactions = [
        {
            "description": "TRANSFERENCIA A CUENTA AHORRO",
            "amount": -500,
            "date": "01/07/2025",
            "category": "Other"
        },
        {
            "description": "NOMINA JULIO",
            "amount": 2500,
            "date": "25/07/2025",
            "category": "Income"  # Already categorized, will skip
        },
        {
            "description": "NUEVO COMERCIO XYZ",
            "amount": -45.50,
            "date": "15/07/2025",
            "category": "Other"
        }
    ]
    
    categorizer = SimplifiedMultiAgentCategorizer()
    results = await categorizer.process_others_only(test_transactions)
    
    print("\n✅ Final Results:")
    for t in results:
        print(f"  {t['description']}: {t['category']}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        exit(1)
    
    asyncio.run(main())