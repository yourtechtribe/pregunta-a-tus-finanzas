#!/usr/bin/env python3
"""
Generate embeddings for adaptive chunks using OpenAI API.
Simplified version for pregunta-tus-finanzas project.
Prepares data for LightRAG graph construction.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import time
import argparse
from glob import glob

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    """Generate embeddings for adaptive financial chunks"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "text-embedding-3-small"
        self.chunks = []
        self.total_cost = 0.0
        self.pricing = {
            "text-embedding-3-small": 0.00002  # per 1K tokens
        }
        
    def load_chunks(self, chunks_path: str):
        """Load chunks from JSON file or directory"""
        all_chunks = []
        
        # Handle directory of chunk files
        if os.path.isdir(chunks_path):
            print(f"📁 Loading chunks from directory: {chunks_path}")
            chunk_files = glob(os.path.join(chunks_path, "*.json"))
            
            for file_path in chunk_files:
                print(f"  Loading: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chunks' in data:
                        all_chunks.extend(data['chunks'])
                    else:
                        all_chunks.append(data)
        
        # Handle single file
        else:
            print(f"📄 Loading chunks from file: {chunks_path}")
            with open(chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'chunks' in data:
                    all_chunks = data['chunks']
                else:
                    all_chunks = [data]
        
        self.chunks = all_chunks
        print(f"✅ Loaded {len(self.chunks)} total chunks")
        
        # Count by type
        type_counts = {}
        for chunk in self.chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        print("\n📊 Chunk distribution by type:")
        for chunk_type, count in sorted(type_counts.items()):
            print(f"  - {chunk_type}: {count}")
    
    def prepare_chunk_text(self, chunk: Dict) -> str:
        """Prepare chunk text optimized for embedding based on type"""
        chunk_type = chunk.get('chunk_type', 'unknown')
        base_text = chunk.get('text', '')
        
        # Add type-specific context for better embeddings
        if chunk_type == 'narrative':
            # Narrative chunks are already well-formatted
            return base_text
        
        elif chunk_type == 'temporal':
            # Add temporal context
            temporal_info = chunk.get('temporal_info', {})
            period = temporal_info.get('period', '')
            if period:
                return f"[Período: {period}] {base_text}"
            return base_text
        
        elif chunk_type == 'categorical':
            # Emphasize category
            category = chunk.get('category', 'unknown')
            subcategory = chunk.get('subcategory', '')
            prefix = f"[{category}"
            if subcategory:
                prefix += f" - {subcategory}"
            prefix += "] "
            return prefix + base_text
        
        elif chunk_type == 'entity':
            # Emphasize entity/merchant
            entity_info = chunk.get('entity_info', {})
            entity = entity_info.get('entity', '')
            if entity:
                return f"[Comercio: {entity}] {base_text}"
            return base_text
        
        elif chunk_type == 'pattern':
            # Include pattern type
            pattern_info = chunk.get('pattern_info', {})
            pattern_type = pattern_info.get('pattern_type', '')
            if pattern_type:
                return f"[Patrón {pattern_type}] {base_text}"
            return base_text
        
        return base_text
    
    def estimate_cost(self):
        """Estimate the cost of generating embeddings"""
        total_chars = 0
        for chunk in self.chunks:
            text = self.prepare_chunk_text(chunk)
            total_chars += len(text)
        
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = total_chars / 4
        estimated_cost = (estimated_tokens / 1000) * self.pricing[self.model]
        
        print(f"\n💰 Cost Estimation:")
        print(f"  - Total chunks: {len(self.chunks)}")
        print(f"  - Total characters: {total_chars:,}")
        print(f"  - Estimated tokens: {estimated_tokens:,.0f}")
        print(f"  - Estimated cost: ${estimated_cost:.4f}")
        
        return estimated_cost
    
    def generate_embeddings(self, batch_size: int = 20):
        """Generate embeddings for all chunks"""
        print(f"\n🚀 Starting embedding generation...")
        print(f"  - Model: {self.model}")
        print(f"  - Batch size: {batch_size}")
        
        chunks_with_embeddings = []
        total_chunks = len(self.chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i:i+batch_size]
            batch_texts = [self.prepare_chunk_text(chunk) for chunk in batch]
            
            print(f"\n📦 Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
            print(f"  - Chunks {i+1} to {min(i+batch_size, total_chunks)}")
            
            try:
                # Generate embeddings for the batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                # Calculate cost for this batch
                batch_chars = sum(len(text) for text in batch_texts)
                batch_tokens = batch_chars / 4  # Rough estimate
                batch_cost = (batch_tokens / 1000) * self.pricing[self.model]
                self.total_cost += batch_cost
                
                # Add embeddings to chunks
                for j, chunk in enumerate(batch):
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = response.data[j].embedding
                    chunk_with_embedding['embedding_model'] = self.model
                    chunk_with_embedding['embedding_generated_at'] = datetime.now().isoformat()
                    chunk_with_embedding['prepared_text'] = batch_texts[j]  # Store prepared text
                    chunks_with_embeddings.append(chunk_with_embedding)
                
                print(f"  ✅ Batch completed (cost: ${batch_cost:.6f})")
                
                # Brief pause to respect rate limits
                if i + batch_size < total_chunks:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"  ❌ Error in batch: {e}")
                # Add chunks without embeddings
                for chunk in batch:
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding_error'] = str(e)
                    chunks_with_embeddings.append(chunk_with_embedding)
        
        print(f"\n✅ Embedding generation complete!")
        print(f"  - Total chunks processed: {len(chunks_with_embeddings)}")
        print(f"  - Successful embeddings: {sum(1 for c in chunks_with_embeddings if 'embedding' in c)}")
        print(f"  - Failed embeddings: {sum(1 for c in chunks_with_embeddings if 'embedding_error' in c)}")
        print(f"  - Total cost: ${self.total_cost:.6f}")
        
        return chunks_with_embeddings
    
    def save_embeddings(self, chunks_with_embeddings: List[Dict], output_path: str):
        """Save chunks with embeddings to JSON file - LightRAG ready format"""
        
        # Prepare metadata for LightRAG
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'embedding_model': self.model,
                'total_chunks': len(chunks_with_embeddings),
                'successful_embeddings': sum(1 for c in chunks_with_embeddings if 'embedding' in c),
                'total_cost': self.total_cost,
                'chunk_types': {},
                'lightrag_ready': True,  # Flag for LightRAG compatibility
                'embedding_dimension': 1536  # text-embedding-3-small dimension
            },
            'chunks': chunks_with_embeddings
        }
        
        # Count chunk types and prepare summary
        for chunk in chunks_with_embeddings:
            chunk_type = chunk.get('chunk_type', 'unknown')
            output_data['metadata']['chunk_types'][chunk_type] = \
                output_data['metadata']['chunk_types'].get(chunk_type, 0) + 1
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save main embeddings file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved embeddings to: {output_path}")
        
        # Also save a LightRAG-specific format for easier integration
        lightrag_path = output_path.replace('.json', '_lightrag.json')
        self.save_lightrag_format(chunks_with_embeddings, lightrag_path)
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"  - Total chunks with embeddings: {output_data['metadata']['successful_embeddings']}")
        print(f"  - Embedding dimension: {output_data['metadata']['embedding_dimension']}")
        
        print("\n🎯 Chunk types processed:")
        for chunk_type, count in sorted(output_data['metadata']['chunk_types'].items()):
            print(f"  - {chunk_type}: {count}")
        
        # Sample verification
        sample_chunks = [c for c in chunks_with_embeddings if 'embedding' in c][:3]
        if sample_chunks:
            print(f"\n✔️ Verification - Sample embedding dimensions: {[len(c['embedding']) for c in sample_chunks]}")
    
    def save_lightrag_format(self, chunks_with_embeddings: List[Dict], output_path: str):
        """Save in a format optimized for LightRAG graph construction"""
        
        # Create documents in LightRAG format
        documents = []
        
        for chunk in chunks_with_embeddings:
            if 'embedding' in chunk:
                doc = {
                    'id': chunk.get('chunk_id', ''),
                    'content': chunk.get('prepared_text', chunk.get('text', '')),
                    'metadata': {
                        'chunk_type': chunk.get('chunk_type', 'unknown'),
                        'category': chunk.get('category', ''),
                        'subcategory': chunk.get('subcategory', ''),
                        'transaction_ids': chunk.get('transaction_ids', []),
                        'date_range': chunk.get('date_range', {}),
                        'temporal_info': chunk.get('temporal_info', {}),
                        'entity_info': chunk.get('entity_info', {}),
                        'pattern_info': chunk.get('pattern_info', {}),
                        'statistical_summary': chunk.get('statistical_summary', {})
                    },
                    'embedding': chunk['embedding']
                }
                documents.append(doc)
        
        lightrag_data = {
            'version': '1.0',
            'model': self.model,
            'created_at': datetime.now().isoformat(),
            'documents': documents,
            'graph_ready': True
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lightrag_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved LightRAG format to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for adaptive financial chunks'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSON file or directory with chunks'
    )
    parser.add_argument(
        '--output',
        default='data/embeddings/chunks_with_embeddings.json',
        help='Output JSON file for embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for API calls (default: 20)'
    )
    parser.add_argument(
        '--skip-confirmation',
        action='store_true',
        help='Skip cost confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or export it:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Load chunks
    generator.load_chunks(args.input)
    
    if len(generator.chunks) == 0:
        print("❌ No chunks found to process")
        return
    
    # Estimate cost
    estimated_cost = generator.estimate_cost()
    
    # Ask for confirmation if cost is significant
    if not args.skip_confirmation and estimated_cost > 0.10:
        print(f"\n⚠️  Estimated cost is ${estimated_cost:.4f}")
        response = input("Do you want to proceed? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Generate embeddings
    chunks_with_embeddings = generator.generate_embeddings(batch_size=args.batch_size)
    
    # Save results
    generator.save_embeddings(chunks_with_embeddings, args.output)
    
    print("\n✨ Process complete!")
    print("\n📍 Next steps:")
    print("1. Build LightRAG graph: python3 scripts/build_lightrag_graph.py")
    print("2. Query your finances: python3 scripts/query_finances.py")
    print("\n💡 Tip: The _lightrag.json file is optimized for graph construction")


if __name__ == "__main__":
    main()