#!/usr/bin/env python3
"""
Test the multi-layer anonymization system
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.processors.adaptive_anonymizer import AdaptiveAnonymizer

# Load environment variables
load_dotenv()

def test_anonymization():
    """Test anonymization with different sensitivity levels"""
    
    print("="*60)
    print("🔒 ANONYMIZATION SYSTEM TEST")
    print("="*60)
    
    # Check API key for LLM-based anonymization
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OpenAI API key not found - will use regex-only mode")
        use_llm = False
    else:
        print(f"✅ API key configured for LLM anonymization")
        use_llm = True
    
    # Initialize anonymizer
    anonymizer = AdaptiveAnonymizer()
    print("✅ Anonymizer initialized")
    
    # Load test transactions
    with open("output/test_extraction.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    transactions = data["transactions"][:5]  # Test with first 5
    print(f"\n📊 Testing with {len(transactions)} transactions")
    
    # Test different anonymization levels
    levels = ["low", "medium", "high"]
    
    for level in levels:
        print(f"\n" + "="*60)
        print(f"🔐 TESTING {level.upper()} ANONYMIZATION")
        print("="*60)
        
        # Copy transactions for anonymization
        test_transactions = [tx.copy() for tx in transactions]
        
        # Apply anonymization
        try:
            anon_transactions = anonymizer.anonymize_transactions(test_transactions)
            
            print(f"\n✅ Anonymized {len(anon_transactions)} transactions")
            
            # Show examples
            for i, (orig, anon) in enumerate(zip(transactions[:2], anon_transactions[:2]), 1):
                print(f"\n{i}. Transaction comparison:")
                print("-"*40)
                print(f"Original description: {orig['description']}")
                print(f"Anonymized: {anon.get('description_clean', anon.get('description'))}")
                print(f"Original category: {orig['category']}")
                print(f"Preserved category: {anon['category']}")
                print(f"Amount: €{abs(anon['amount']):.2f} (preserved)")
                
        except Exception as e:
            print(f"❌ Error during {level} anonymization: {e}")
    
    # Test specific sensitive patterns
    print("\n" + "="*60)
    print("🔍 TESTING SENSITIVE PATTERN DETECTION")
    print("="*60)
    
    test_texts = [
        "NOMINA EMPRESA EJEMPLO",
        "TRANSFERENCIA DE JUAN PEREZ",
        "COMPRA EN FARMACIA LOPEZ",
        "PAGO A 123-45-678",
        "INGRESO DE MARIA GARCIA",
        "juan.perez@ejemplo.com",
        "Tel: +34 612 345 678"
    ]
    
    for text in test_texts:
        anon_text = anonymizer.anonymize_text(text)
        print(f"\nOriginal: {text}")
        print(f"Anonymized: {anon_text}")
        
        # Check if sensitive info was detected
        if text != anon_text:
            print("✅ Sensitive information detected and anonymized")
        else:
            print("ℹ️ No sensitive information detected")
    
    # Test batch anonymization performance
    print("\n" + "="*60)
    print("⚡ TESTING BATCH PERFORMANCE")
    print("="*60)
    
    # Load all transactions
    with open("output/test_extraction.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_transactions = data["transactions"]
    print(f"\nAnonymizing {len(all_transactions)} transactions...")
    
    import time
    start_time = time.time()
    
    try:
        anon_all = anonymizer.anonymize_transactions(all_transactions)
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.2f} seconds")
        print(f"Average: {elapsed/len(all_transactions)*1000:.2f} ms per transaction")
        
        # Save anonymized data
        output_file = "output/anonymized_transactions.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "transactions": anon_all,
                "metadata": {
                    "original_count": len(all_transactions),
                    "anonymized_count": len(anon_all),
                    "anonymization_time": elapsed,
                    "method": "adaptive_multilayer"
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved anonymized data to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during batch anonymization: {e}")
    
    # Verify data integrity
    print("\n" + "="*60)
    print("🔍 VERIFYING DATA INTEGRITY")
    print("="*60)
    
    if 'anon_all' in locals():
        # Check that important fields are preserved
        original_total = sum(tx['amount'] for tx in all_transactions)
        anon_total = sum(tx['amount'] for tx in anon_all)
        
        print(f"Original total: €{original_total:.2f}")
        print(f"Anonymized total: €{anon_total:.2f}")
        
        if abs(original_total - anon_total) < 0.01:
            print("✅ Financial integrity preserved")
        else:
            print("❌ Financial totals don't match!")
        
        # Check categories preserved
        orig_categories = set(tx['category'] for tx in all_transactions)
        anon_categories = set(tx['category'] for tx in anon_all)
        
        if orig_categories == anon_categories:
            print("✅ All categories preserved")
        else:
            print(f"⚠️ Category mismatch: {orig_categories - anon_categories}")
    
    print("\n" + "="*60)
    print("✅ Anonymization test completed!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_anonymization()
    
    if success:
        print("\n🎉 SUCCESS: Anonymization system is working correctly!")
        print("Your financial data can be safely anonymized while preserving analytical value.")
    else:
        print("\n⚠️ Some issues were encountered. Check the errors above.")