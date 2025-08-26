#!/usr/bin/env python3
"""
Fixed LightRAG test script with proper async handling
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_lightrag_sync():
    """Test LightRAG with synchronous approach"""
    
    print("\n" + "="*60)
    print("🧪 TESTING LIGHTRAG (FIXED VERSION)")
    print("="*60)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('sk-test'):
        print("❌ OpenAI API key not configured")
        return
    print(f"✅ OpenAI API key configured")
    
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
        print("✅ LightRAG modules imported")
    except ImportError as e:
        print(f"❌ Error importing: {e}")
        return
    
    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./simple_rag_knowledge",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    print("✅ LightRAG initialized")
    
    # Check knowledge graph
    kg_path = Path("simple_rag_knowledge")
    if kg_path.exists():
        files = list(kg_path.glob("*.json"))
        print(f"✅ Knowledge graph found: {len(files)} files")
    else:
        print("⚠️  No knowledge graph found")
        return
    
    # Test queries synchronously
    test_queries = [
        "What are the main expense categories?",
        "Show transactions above 100 EUR",
        "What is the spending pattern?"
    ]
    
    print("\n" + "="*60)
    print("📊 RUNNING QUERIES")
    print("="*60)
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query_text}")
        print("-"*40)
        
        try:
            # Use synchronous query method
            result = rag.query(
                query_text,
                param=QueryParam(mode="naive")
            )
            
            if result:
                print(f"Response: {result[:200]}..." if len(result) > 200 else f"Response: {result}")
            else:
                print("No response received")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print("="*60)

if __name__ == "__main__":
    test_lightrag_sync()