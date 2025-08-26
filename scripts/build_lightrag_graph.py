#!/usr/bin/env python3
"""
Build LightRAG knowledge graph from financial chunks.
This uses the REAL LightRAG library that generates AI-powered responses.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import time
import logging
import sys

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LightRAG components
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
    logger.info("✅ LightRAG modules imported successfully")
except ImportError as e:
    logger.error(f"❌ LightRAG import error: {e}")
    logger.error("Install with: pip install lightrag-hku")
    LIGHTRAG_AVAILABLE = False
    sys.exit(1)


class FinancialLightRAG:
    """
    Real LightRAG implementation for financial data.
    This uses GPT-4o-mini to extract entities, build knowledge graphs,
    and generate natural language responses.
    """
    
    def __init__(self, working_dir: str = "lightrag_financial_graph"):
        """
        Initialize LightRAG with AI-powered entity extraction and response generation.
        
        Args:
            working_dir: Directory where the knowledge graph will be stored
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.rag = None
        self.chunks = []
        self.initialized = False
        
        # Cost tracking
        self.total_cost = 0.0
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.15 / 1_000_000,   # $0.15 per 1M input tokens
                "output": 0.60 / 1_000_000   # $0.60 per 1M output tokens
            }
        }
        
        logger.info(f"📁 Working directory: {self.working_dir}")
    
    def initialize(self):
        """Initialize the real LightRAG system with GPT-4o-mini"""
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")
        
        logger.info("🔧 Initializing LightRAG with GPT-4o-mini...")
        
        # Initialize LightRAG with AI models
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            
            # LLM for entity extraction and response generation
            llm_model_func=gpt_4o_mini_complete,
            
            # Embeddings for semantic search
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8191,
                func=openai_embed
            )
        )
        
        logger.info("✅ LightRAG initialized with AI capabilities")
    
    async def initialize_storages(self):
        """Initialize LightRAG storage systems"""
        try:
            logger.info("📦 Initializing storage systems...")
            await self.rag.initialize_storages()
            self.initialized = True
            logger.info("✅ Storage systems initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize storages: {e}")
            return False
    
    def load_chunks(self, chunks_path: str):
        """
        Load chunks with embeddings from pregunta-tus-finanzas format.
        
        Args:
            chunks_path: Path to chunks_with_embeddings_lightrag.json
        """
        logger.info(f"📥 Loading chunks from: {chunks_path}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats
        if 'documents' in data:
            # LightRAG format
            self.chunks = data['documents']
        elif 'chunks' in data:
            # Standard format
            self.chunks = data['chunks']
        else:
            self.chunks = data if isinstance(data, list) else []
        
        logger.info(f"✅ Loaded {len(self.chunks)} chunks")
        
        # Analyze chunk types
        type_counts = {}
        for chunk in self.chunks:
            # Get chunk type from metadata or direct attribute
            if isinstance(chunk, dict):
                if 'metadata' in chunk:
                    chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
                else:
                    chunk_type = chunk.get('chunk_type', 'unknown')
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        logger.info("📊 Chunk distribution:")
        for chunk_type, count in sorted(type_counts.items()):
            logger.info(f"  - {chunk_type}: {count}")
    
    def prepare_chunk_for_lightrag(self, chunk: dict) -> str:
        """
        Prepare a chunk for LightRAG insertion.
        Format it to help the LLM extract entities and relationships.
        
        Args:
            chunk: Chunk dictionary with content and metadata
            
        Returns:
            Formatted text for LightRAG
        """
        # Get content
        if 'content' in chunk:
            content = chunk['content']
        elif 'prepared_text' in chunk:
            content = chunk['prepared_text']
        elif 'text' in chunk:
            content = chunk['text']
        else:
            return ""
        
        # Get metadata
        metadata = chunk.get('metadata', {})
        chunk_type = metadata.get('chunk_type', chunk.get('chunk_type', 'unknown'))
        
        # Build context-rich text for entity extraction
        parts = [f"[{chunk_type.upper()}]"]
        
        # Add relevant metadata to help entity extraction
        if metadata.get('category'):
            parts.append(f"Category: {metadata['category']}")
        
        if metadata.get('date_range'):
            date_range = metadata['date_range']
            if 'start' in date_range:
                parts.append(f"Period: {date_range['start']} to {date_range.get('end', 'present')}")
        
        if metadata.get('entity_info'):
            entity = metadata['entity_info'].get('entity')
            if entity:
                parts.append(f"Entity: {entity}")
        
        # Add the main content
        parts.append("\n" + content)
        
        # Add statistical summary if available
        if metadata.get('statistical_summary'):
            stats = metadata['statistical_summary']
            if 'total' in stats:
                parts.append(f"\nTotal Amount: €{stats['total']:.2f}")
            if 'count' in stats:
                parts.append(f"Transactions: {stats['count']}")
        
        return "\n".join(parts)
    
    async def build_knowledge_graph(self, max_chunks: int = None, batch_size: int = 5):
        """
        Build the knowledge graph using LightRAG's AI-powered entity extraction.
        
        Args:
            max_chunks: Maximum number of chunks to process (None = all)
            batch_size: Number of chunks to process in each batch
        """
        if not self.initialized:
            logger.error("❌ LightRAG not initialized! Call initialize_storages() first")
            return
        
        chunks_to_process = self.chunks[:max_chunks] if max_chunks else self.chunks
        total_chunks = len(chunks_to_process)
        
        logger.info(f"\n🏗️ Building AI-Powered Knowledge Graph...")
        logger.info(f"  - Processing {total_chunks} chunks")
        logger.info(f"  - GPT-4o-mini will extract entities and relationships")
        logger.info(f"  - Batch size: {batch_size}")
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks_to_process[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            logger.info(f"\n📦 Processing batch {batch_num}/{total_batches}")
            
            # Process each chunk in the batch
            batch_tasks = []
            for j, chunk in enumerate(batch):
                chunk_text = self.prepare_chunk_for_lightrag(chunk)
                
                if chunk_text:
                    # Create async task for insertion
                    task = self._insert_chunk(chunk_text, i + j)
                    batch_tasks.append(task)
            
            # Execute batch in parallel
            if batch_tasks:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Count successes and failures
                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                        logger.error(f"  ❌ Error: {result}")
                    else:
                        processed += 1
            
            # Estimate cost (rough approximation)
            batch_chars = sum(len(self.prepare_chunk_for_lightrag(c)) for c in batch)
            batch_tokens = batch_chars / 4
            batch_cost = batch_tokens * (
                self.pricing["gpt-4o-mini"]["input"] * 2 +  # Entity extraction
                self.pricing["gpt-4o-mini"]["output"] * 0.5  # Extracted entities
            )
            self.total_cost += batch_cost
            
            logger.info(f"  ✅ Batch completed - Processed: {len(batch)}, Cost: ${batch_cost:.4f}")
            
            # Brief pause between batches
            if i + batch_size < total_chunks:
                await asyncio.sleep(0.5)
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n✅ Knowledge Graph Building Complete!")
        logger.info(f"  - Successfully processed: {processed}/{total_chunks}")
        logger.info(f"  - Failed: {failed}")
        logger.info(f"  - Time elapsed: {elapsed:.1f}s")
        logger.info(f"  - Estimated total cost: ${self.total_cost:.4f}")
        
        # Check generated files
        self._check_graph_files()
    
    async def _insert_chunk(self, chunk_text: str, index: int):
        """Insert a single chunk into LightRAG"""
        try:
            await self.rag.ainsert(chunk_text)
            logger.info(f"  ✓ Chunk {index + 1} inserted")
            return True
        except Exception as e:
            logger.error(f"  ✗ Chunk {index + 1} failed: {e}")
            raise e
    
    def _check_graph_files(self):
        """Check what files were generated by LightRAG"""
        logger.info(f"\n📁 Checking generated graph files in {self.working_dir}...")
        
        expected_files = [
            "graph_storage.json",
            "vector_storage.json", 
            "entities.json",
            "relationships.json",
            "communities.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json"
        ]
        
        for filename in expected_files:
            file_path = self.working_dir / filename
            if file_path.exists():
                size = file_path.stat().st_size
                logger.info(f"  ✅ {filename}: {size:,} bytes")
                
                # Show sample of entities if available
                if filename == "entities.json" and size > 0:
                    with open(file_path, 'r') as f:
                        entities = json.load(f)
                        if entities:
                            logger.info(f"     Sample entities: {list(entities.keys())[:5]}")
            else:
                logger.info(f"  ⚠️ {filename}: Not found yet")
    
    async def query(self, question: str, mode: str = "hybrid"):
        """
        Query the knowledge graph using AI-generated responses.
        
        Args:
            question: Natural language question
            mode: Query mode - "naive", "local", "global", or "hybrid"
            
        Returns:
            AI-generated natural language response
        """
        if not self.initialized:
            logger.error("❌ LightRAG not initialized!")
            return None
        
        logger.info(f"\n🤖 AI Query: {question}")
        logger.info(f"  Mode: {mode}")
        
        try:
            # Create query parameters
            param = QueryParam(mode=mode)
            
            # Execute query - LightRAG will:
            # 1. Search the knowledge graph
            # 2. Retrieve relevant context
            # 3. Generate response using GPT-4o-mini
            start_time = time.time()
            response = await self.rag.aquery(question, param)
            elapsed = time.time() - start_time
            
            logger.info(f"  ✅ Response generated in {elapsed:.1f}s")
            
            # Estimate query cost
            query_cost = (len(question) + len(response)) / 4 * self.pricing["gpt-4o-mini"]["input"]
            logger.info(f"  💰 Estimated cost: ${query_cost:.6f}")
            
            return response
            
        except Exception as e:
            logger.error(f"  ❌ Query failed: {e}")
            return None
    
    async def test_ai_queries(self):
        """Test the knowledge graph with sample queries using AI generation"""
        
        logger.info("\n" + "="*60)
        logger.info("🧪 TESTING AI-POWERED QUERIES")
        logger.info("="*60)
        
        test_queries = [
            ("💰 GASTOS", "¿Cuánto gasté en total y cuáles fueron mis principales categorías de gasto?"),
            ("🛒 CATEGORÍAS", "Analiza mis gastos en Groceries y dame recomendaciones"),
            ("📈 PATRONES", "¿Qué patrones de gasto puedes identificar en mis transacciones?"),
            ("💡 AHORRO", "¿Dónde puedo ahorrar dinero basándote en mis gastos actuales?"),
            ("📊 RESUMEN", "Dame un resumen ejecutivo de mi situación financiera")
        ]
        
        results = []
        
        for category, query in test_queries:
            logger.info(f"\n{category}")
            logger.info(f"📝 Query: {query}")
            logger.info("-" * 50)
            
            # Test different modes
            for mode in ["hybrid"]:  # You can test other modes: "local", "global", "naive"
                response = await self.query(query, mode=mode)
                
                if response:
                    # Show preview of response
                    preview = response[:300] + "..." if len(response) > 300 else response
                    logger.info(f"\n🤖 AI Response ({mode} mode):")
                    logger.info(f"{preview}")
                    
                    results.append({
                        "category": category,
                        "query": query,
                        "mode": mode,
                        "response": response
                    })
                else:
                    logger.error(f"No response for mode: {mode}")
        
        # Save results
        output_file = self.working_dir / "ai_query_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        return results


async def main():
    """Main function to build and test the LightRAG knowledge graph"""
    
    # Paths
    chunks_path = "data/embeddings/chunks_with_embeddings_lightrag.json"
    working_dir = "lightrag_financial_graph"
    
    # Check if chunks exist
    if not Path(chunks_path).exists():
        logger.error(f"❌ Chunks file not found: {chunks_path}")
        logger.info("Run first: python3 scripts/generate_embeddings.py")
        return
    
    # Initialize LightRAG
    logger.info("\n" + "="*60)
    logger.info("🚀 BUILDING AI-POWERED FINANCIAL KNOWLEDGE GRAPH")
    logger.info("="*60)
    
    rag = FinancialLightRAG(working_dir=working_dir)
    rag.initialize()
    
    # Initialize storages
    if not await rag.initialize_storages():
        logger.error("Failed to initialize storages")
        return
    
    # Load chunks
    rag.load_chunks(chunks_path)
    
    # Estimate cost
    total_chars = sum(len(rag.prepare_chunk_for_lightrag(c)) for c in rag.chunks)
    estimated_cost = (total_chars / 4) * rag.pricing["gpt-4o-mini"]["input"] * 3
    logger.info(f"\n💰 Estimated cost for full graph: ${estimated_cost:.4f}")
    
    if estimated_cost > 0.50:
        logger.warning(f"⚠️ High estimated cost: ${estimated_cost:.4f}")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return
    
    # Build knowledge graph
    logger.info("\n🏗️ Building knowledge graph with AI entity extraction...")
    await rag.build_knowledge_graph(max_chunks=None, batch_size=5)
    
    # Test with AI queries
    logger.info("\n🧪 Testing AI-powered queries...")
    await rag.test_ai_queries()
    
    logger.info("\n" + "="*60)
    logger.info("✨ AI-POWERED KNOWLEDGE GRAPH COMPLETE!")
    logger.info("="*60)
    logger.info("\nThe system can now:")
    logger.info("  ✅ Answer questions using GPT-4o-mini")
    logger.info("  ✅ Generate natural language responses")
    logger.info("  ✅ Understand context and relationships")
    logger.info("  ✅ Provide financial insights and recommendations")
    logger.info(f"\nGraph location: {working_dir}/")
    logger.info("\nNext: Run python3 scripts/query_lightrag_ai.py for interactive queries")


if __name__ == "__main__":
    asyncio.run(main())