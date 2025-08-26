#!/bin/bash

# Setup script for "Pregunta tus Finanzas"
# This script automates the initial setup process

echo "================================================"
echo "🚀 Pregunta tus Finanzas - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
else
    echo "✅ Python $python_version found"
fi

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo ""
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "🧠 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Create .env file if it doesn't exist
echo ""
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists. Skipping..."
else
    echo "📝 Creating .env file..."
    cat > .env << EOL
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-api-key-here

# Tavily API Configuration (optional)
TAVILY_API_KEY=tvly-your-api-key-here

# Environment
ENVIRONMENT=development
EOL
    echo "✅ .env file created"
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/embeddings
mkdir -p outputs
mkdir -p simple_rag_knowledge
echo "✅ Directories created"

# Run tests to verify installation
echo ""
echo "🧪 Running quick verification test..."
python -c "
import pandas
import numpy
import lightrag
import openai
import spacy
print('✅ All core modules imported successfully')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Installation verified"
else
    echo "⚠️  Some modules may not be properly installed"
fi

echo ""
echo "================================================"
echo "✅ Setup completed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Place your bank statements in data/raw/"
echo "3. Run: python scripts/run_pipeline.py"
echo "4. Build knowledge graph: python scripts/build_lightrag_graph.py"
echo "5. Query your data: python scripts/demo_queries.py"
echo ""
echo "For more information, check the README.md file"
echo "================================================"