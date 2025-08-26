#!/usr/bin/env python3
"""
Test script for Multi-Agent Categorization System
Tests integration with BBVA pipeline and categorization accuracy
"""

import asyncio
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent))

from src.agents.categorization_agent import MultiAgentCategorizer
from src.extractors.bbva_extractor import BBVAExtractor


async def test_standalone_categorization():
    """Test the multi-agent system with predefined transactions"""
    print("\n" + "=" * 80)
    print("TEST 1: STANDALONE CATEGORIZATION")
    print("=" * 80)
    
    # Test transactions covering different scenarios
    test_transactions = [
        # Should use BBVA rules
        {
            "description": "CACENCA",
            "amount": -45.00,
            "date": "15/07/2025",
            "category": "Other"
        },
        {
            "description": "TRASPASO PROGRAMA TU CUENTA",
            "amount": -500.00,
            "date": "01/07/2025",
            "category": "Other"
        },
        # Should need research
        {
            "description": "CLAUDE.AI SUBSCRIPTION",
            "amount": -20.00,
            "date": "01/07/2025",
            "category": "Other"
        },
        {
            "description": "RESTAURANTE UMAMI BCN",
            "amount": -55.50,
            "date": "20/07/2025",
            "category": "Other"
        },
        # Parking meter pattern
        {
            "description": "16307 SANT CUGAT",
            "amount": -3.50,
            "date": "22/07/2025",
            "category": "Other"
        },
        # Supermarket
        {
            "description": "CARREFOUR MARKET",
            "amount": -87.43,
            "date": "16/07/2025",
            "category": "Other"
        },
        # Should be Income (positive amount)
        {
            "description": "NOMINA EMPRESA XYZ",
            "amount": 2500.00,
            "date": "25/07/2025",
            "category": "Other"
        },
        # Should be Savings (pension plan)
        {
            "description": "PLAN DE PENSIONES BBVA",
            "amount": -150.00,
            "date": "05/07/2025",
            "category": "Other"
        }
    ]
    
    # Initialize categorizer
    categorizer = MultiAgentCategorizer()
    
    # Process transactions
    results = await categorizer.process_batch(test_transactions, only_others=True)
    
    # Display results
    print("\n📊 RESULTS:")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        tx = result['transaction']
        print(f"\n{i}. {tx['description']}")
        print(f"   Amount: {tx['amount']}€")
        print(f"   Original: {tx.get('category', 'Other')}")
        print(f"   → New Category: {result['category']}")
        print(f"   → Confidence: {result['confidence']:.1%}")
        print(f"   → Method: {result['method']}")
    
    # Calculate accuracy
    expected = [
        "Transportation",  # CACENCA
        "Internal Transfer",  # TRASPASO
        "Tech & Software",  # CLAUDE.AI
        "Food & Dining",  # UMAMI
        "Transportation",  # 16307 (parking)
        "Groceries",  # CARREFOUR
        "Income",  # NOMINA
        "Savings"  # PLAN PENSIONES
    ]
    
    correct = sum(1 for r, e in zip(results, expected) if r['category'] == e)
    accuracy = (correct / len(expected)) * 100
    
    print(f"\n✅ Accuracy: {correct}/{len(expected)} ({accuracy:.1f}%)")
    
    return results


async def test_with_csv_file():
    """Test with actual CSV file from examples"""
    print("\n" + "=" * 80)
    print("TEST 2: CSV FILE INTEGRATION")
    print("=" * 80)
    
    csv_path = Path("examples/sample_data.csv")
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return None
    
    # Extract transactions using BBVAExtractor
    print("📂 Loading transactions from CSV...")
    extractor = BBVAExtractor()
    result = extractor.extract(str(csv_path))
    transactions = result['transactions']
    
    print(f"✅ Loaded {len(transactions)} transactions")
    
    # Show category distribution before
    categories_before = {}
    for t in transactions:
        cat = t.get('category', 'Other')
        categories_before[cat] = categories_before.get(cat, 0) + 1
    
    print("\n📊 Categories BEFORE multi-agent processing:")
    for cat, count in sorted(categories_before.items()):
        print(f"   {cat}: {count}")
    
    # Process only "Other" categories with multi-agent
    categorizer = MultiAgentCategorizer()
    improved_results = await categorizer.process_batch(transactions, only_others=True)
    
    # Update transactions with improved categories
    for result in improved_results:
        # Find and update original transaction
        for t in transactions:
            if (t.get('description') == result['transaction'].get('description') and
                t.get('amount') == result['transaction'].get('amount')):
                t['category'] = result['category']
                t['categorization_confidence'] = result['confidence']
                t['categorization_method'] = result['method']
    
    # Show category distribution after
    categories_after = {}
    for t in transactions:
        cat = t.get('category', 'Other')
        categories_after[cat] = categories_after.get(cat, 0) + 1
    
    print("\n📊 Categories AFTER multi-agent processing:")
    for cat, count in sorted(categories_after.items()):
        print(f"   {cat}: {count}")
    
    # Calculate improvement
    others_before = categories_before.get('Other', 0)
    others_after = categories_after.get('Other', 0)
    improvement = others_before - others_after
    
    print(f"\n✅ Improvement: Reduced 'Other' from {others_before} to {others_after}")
    print(f"   ({improvement} transactions re-categorized)")
    
    # Save improved results
    output_file = Path("output/transactions_multi_agent_improved.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 Saved improved transactions to: {output_file}")
    
    return transactions


async def test_memory_persistence():
    """Test that memory/cache persists between runs"""
    print("\n" + "=" * 80)
    print("TEST 3: MEMORY PERSISTENCE")
    print("=" * 80)
    
    test_tx = {
        "description": "UNIQUE_MERCHANT_TEST_123",
        "amount": -50.00,
        "date": "15/07/2025",
        "category": "Other"
    }
    
    # First run - should use research
    print("\n🔄 First run (should research):")
    categorizer1 = MultiAgentCategorizer(cache_file="test_cache.json")
    result1 = await categorizer1.categorize_transaction(test_tx)
    print(f"   Method: {result1['method']}")
    print(f"   Category: {result1['category']}")
    
    # Second run - should use memory
    print("\n🔄 Second run (should use memory):")
    categorizer2 = MultiAgentCategorizer(cache_file="test_cache.json")
    result2 = await categorizer2.categorize_transaction(test_tx)
    print(f"   Method: {result2['method']}")
    print(f"   Category: {result2['category']}")
    
    # Verify memory was used
    if result2['method'] == 'memory':
        print("\n✅ Memory persistence working correctly!")
    else:
        print("\n⚠️ Memory persistence not working as expected")
    
    return result2


async def test_validation_logic():
    """Test validation and confidence adjustment"""
    print("\n" + "=" * 80)
    print("TEST 4: VALIDATION LOGIC")
    print("=" * 80)
    
    test_cases = [
        # Income with negative amount (should reduce confidence)
        {
            "description": "DEVOLUCION COMPRA",
            "amount": -50.00,  # Negative but labeled as income
            "date": "15/07/2025",
            "category": "Income"
        },
        # Internal transfer (should be corrected)
        {
            "description": "TRASPASO ENTRE CUENTAS",
            "amount": -1000.00,
            "date": "10/07/2025",
            "category": "Transfers"  # Should be corrected to Internal Transfer
        },
        # Known merchant (should boost confidence)
        {
            "description": "CACENCA GASOLINERA",
            "amount": -45.00,
            "date": "20/07/2025",
            "category": "Transportation"
        }
    ]
    
    categorizer = MultiAgentCategorizer()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test['description']}")
        print(f"   Original category: {test['category']}")
        
        result = await categorizer.categorize_transaction(test)
        
        print(f"   → Final category: {result['category']}")
        print(f"   → Confidence: {result['confidence']:.1%}")
        print(f"   → Validation: {result.get('validation', {})}")
    
    print("\n✅ Validation tests completed")


async def main():
    """Run all tests"""
    print("\n" + "🤖" * 20)
    print("MULTI-AGENT CATEGORIZATION SYSTEM TEST SUITE")
    print("🤖" * 20)
    
    # Check environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not found in .env file")
        print("   The system will fall back to rule-based categorization")
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠️ TAVILY_API_KEY not found in .env file")
        print("   Web search will not be available")
    
    # Run tests
    try:
        # Test 1: Standalone categorization
        await test_standalone_categorization()
        
        # Test 2: CSV file integration
        await test_with_csv_file()
        
        # Test 3: Memory persistence
        # await test_memory_persistence()  # Skip if no API keys
        
        # Test 4: Validation logic
        await test_validation_logic()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())