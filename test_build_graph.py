#!/usr/bin/env python3
"""
Test LightRAG graph building with OpenAI API - Synchronous Version
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Load environment variables
load_dotenv()

def build_and_test_graph():
    """Build knowledge graph from extracted transactions"""
    
    print("="*60)
    print("🔧 LIGHTRAG GRAPH BUILDING TEST (SYNC)")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found in .env")
        return False
    print(f"✅ API key configured: {api_key[:20]}...")
    
    # Initialize LightRAG
    working_dir = Path("test_rag_knowledge")
    working_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Working directory: {working_dir}")
    
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    print("✅ LightRAG initialized")
    
    # Load extracted transactions
    with open("output/test_extraction.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    transactions = data["transactions"]
    print(f"\n📊 Processing {len(transactions)} transactions...")
    
    # Convert transactions to text chunks for RAG
    text_chunks = []
    for tx in transactions:
        chunk = (
            f"On {tx['date']}, there was a {'income' if tx['amount'] > 0 else 'expense'} "
            f"transaction: {tx['description']} for €{abs(tx['amount']):.2f} "
            f"in category {tx['category']}."
        )
        text_chunks.append(chunk)
    
    # Add summary information
    total_income = sum(tx['amount'] for tx in transactions if tx['amount'] > 0)
    total_expenses = sum(tx['amount'] for tx in transactions if tx['amount'] < 0)
    
    summary = (
        f"Financial summary: Total income was €{total_income:.2f}, "
        f"total expenses were €{abs(total_expenses):.2f}, "
        f"resulting in a balance of €{total_income + total_expenses:.2f}. "
        f"Main expense categories include Groceries, Housing, and Transportation."
    )
    text_chunks.append(summary)
    
    # Build the knowledge graph
    full_text = "\n".join(text_chunks)
    print(f"\n📝 Text size: {len(full_text)} characters")
    
    print("\n🔨 Building knowledge graph (this may take a moment)...")
    try:
        # Use synchronous insert method
        rag.insert(full_text)
        print("✅ Knowledge graph built successfully!")
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with queries
    print("\n" + "="*60)
    print("🔍 TESTING QUERIES")
    print("="*60)
    
    test_queries = [
        "What is my total income?",
        "What are my main expenses?",
        "Show me grocery transactions",
        "What is my financial balance?",
    ]
    
    for query in test_queries:
        print(f"\n📌 Query: {query}")
        print("-"*40)
        
        try:
            # Use synchronous query method
            response = rag.query(
                query,
                param=QueryParam(mode="naive")
            )
            
            if response:
                print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
            else:
                print("No response received")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Check graph files created
    print("\n" + "="*60)
    print("📂 GRAPH FILES CREATED")
    print("="*60)
    
    graph_files = list(working_dir.glob("*.json")) + list(working_dir.glob("*.graphml"))
    for file in graph_files[:5]:  # Show first 5 files
        size = file.stat().st_size / 1024  # KB
        print(f"  - {file.name}: {size:.1f} KB")
    
    if len(graph_files) > 5:
        print(f"  ... and {len(graph_files) - 5} more files")
    
    print("\n✅ Graph building test completed!")
    return True

if __name__ == "__main__":
    # Run synchronous function
    success = build_and_test_graph()
    
    if success:
        print("\n🎉 SUCCESS: LightRAG is working with your API keys!")
        print("You can now build more complex knowledge graphs with your financial data.")
    else:
        print("\n⚠️ Some issues were encountered. Check the errors above.")