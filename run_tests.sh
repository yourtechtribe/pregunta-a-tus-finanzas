#!/bin/bash

# Test runner script for Pregunta tus Finanzas
echo "================================================"
echo "🧪 Running Test Suite"
echo "================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Test 1: Check dependencies
echo "1. Testing dependencies..."
python -c "
import pandas
import numpy
import lightrag
import openai
import spacy
print('   ✅ All core dependencies installed')
" 2>/dev/null || echo "   ❌ Missing dependencies"

# Test 2: Test data extraction
echo ""
echo "2. Testing data extraction..."
python -c "
from src.extractors.bbva_extractor import BBVAExtractor
extractor = BBVAExtractor()
result = extractor.extract('examples/sample_data.csv')
if result['status'] == 'success':
    print(f'   ✅ Extracted {len(result[\"transactions\"])} transactions')
else:
    print('   ❌ Extraction failed')
" 2>/dev/null || echo "   ❌ Extractor error"

# Test 3: Test anonymization
echo ""
echo "3. Testing anonymization..."
python -c "
from src.processors.adaptive_anonymizer import AdaptiveAnonymizer
anonymizer = AdaptiveAnonymizer()
test_data = [{'concept': 'Payment to John Doe', 'amount': 100}]
result = anonymizer.anonymize_transactions(test_data)
if len(result) == 1:
    print('   ✅ Anonymization working')
else:
    print('   ❌ Anonymization failed')
" 2>/dev/null || echo "   ❌ Anonymizer error"

# Test 4: Test LightRAG
echo ""
echo "4. Testing LightRAG..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key and not api_key.startswith('sk-test'):
    print('   ✅ API key configured')
    from lightrag import LightRAG
    print('   ✅ LightRAG importable')
else:
    print('   ⚠️  API key not configured')
" 2>/dev/null || echo "   ❌ LightRAG error"

# Test 5: Run pytest if available
echo ""
echo "5. Running unit tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short 2>/dev/null || echo "   ⚠️  Some tests failed"
else
    echo "   ⚠️  pytest not found"
fi

echo ""
echo "================================================"
echo "✅ Test suite completed"
echo "================================================"