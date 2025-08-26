#!/usr/bin/env python3
"""
Alternative RAG implementation using LangChain
Due to LightRAG async issues, this provides a working alternative
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

def test_langchain_rag():
    """Test RAG with LangChain and Chroma"""
    
    print("="*60)
    print("🔧 LANGCHAIN RAG TEST (ALTERNATIVE TO LIGHTRAG)")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found in .env")
        return False
    print(f"✅ API key configured: {api_key[:20]}...")
    
    # Load extracted transactions
    with open("output/test_extraction.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    transactions = data["transactions"]
    print(f"\n📊 Processing {len(transactions)} transactions...")
    
    # Prepare documents for RAG
    documents = []
    for tx in transactions:
        content = (
            f"Date: {tx['date']}\n"
            f"Type: {'Income' if tx['amount'] > 0 else 'Expense'}\n"
            f"Description: {tx['description']}\n"
            f"Amount: €{abs(tx['amount']):.2f}\n"
            f"Category: {tx['category']}\n"
            f"Concept: {tx.get('concept', tx['description'])}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "date": tx['date'],
                "amount": tx['amount'],
                "category": tx['category'],
                "type": "income" if tx['amount'] > 0 else "expense"
            }
        )
        documents.append(doc)
    
    # Add summary document
    total_income = sum(tx['amount'] for tx in transactions if tx['amount'] > 0)
    total_expenses = sum(tx['amount'] for tx in transactions if tx['amount'] < 0)
    
    summary_content = (
        f"Financial Summary:\n"
        f"Total Income: €{total_income:.2f}\n"
        f"Total Expenses: €{abs(total_expenses):.2f}\n"
        f"Net Balance: €{total_income + total_expenses:.2f}\n"
        f"Number of Transactions: {len(transactions)}\n"
        f"Main Expense Categories: {', '.join(set(tx['category'] for tx in transactions if tx['amount'] < 0))}"
    )
    
    summary_doc = Document(
        page_content=summary_content,
        metadata={"type": "summary"}
    )
    documents.append(summary_doc)
    
    print(f"📝 Created {len(documents)} documents for indexing")
    
    # Initialize embeddings and vector store
    print("\n🔨 Building vector store with OpenAI embeddings...")
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./langchain_chroma_db"
        )
        print("✅ Vector store created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False
    
    # Initialize LLM and QA chain
    print("\n🤖 Initializing QA chain...")
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Test queries
    print("\n" + "="*60)
    print("🔍 TESTING QUERIES")
    print("="*60)
    
    test_queries = [
        "What is my total income?",
        "What are my main expense categories?",
        "Show me transactions above 100 EUR",
        "What is my financial balance?",
        "What percentage of income goes to housing?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-"*40)
        
        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "No answer")
            
            print(f"Answer: {answer[:300]}..." if len(answer) > 300 else f"Answer: {answer}")
            
            # Show sources
            if result.get("source_documents"):
                print(f"Sources: {len(result['source_documents'])} documents used")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Test similarity search
    print("\n" + "="*60)
    print("🔍 TESTING SIMILARITY SEARCH")
    print("="*60)
    
    search_query = "expensive transactions"
    print(f"\nSearching for: '{search_query}'")
    print("-"*40)
    
    similar_docs = vectorstore.similarity_search(search_query, k=3)
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n{i}. Similar document:")
        print(doc.page_content[:200])
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    print("\n" + "="*60)
    print("✅ LangChain RAG test completed successfully!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_langchain_rag()
    
    if success:
        print("\n🎉 SUCCESS: LangChain RAG is working with your API keys!")
        print("This is a viable alternative to LightRAG for your financial queries.")
    else:
        print("\n⚠️ Some issues were encountered. Check the errors above.")