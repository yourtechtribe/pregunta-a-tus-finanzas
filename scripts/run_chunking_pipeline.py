#!/usr/bin/env python3
"""
Chunking Pipeline Integration
Runs the chunking strategy after categorization and anonymization
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.chunking_strategy import AdaptiveChunkGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_chunking_pipeline(
    input_file: str,
    output_dir: str = 'output_chunks',
    strategies: list = None,
    anonymized: bool = True
):
    """
    Run the chunking pipeline on processed transactions
    
    Args:
        input_file: Path to categorized/anonymized transactions JSON
        output_dir: Directory for output files
        strategies: List of chunking strategies to use
        anonymized: Whether the data has been anonymized
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{'='*60}")
        logger.info("🚀 INICIANDO PIPELINE DE CHUNKING")
        logger.info(f"{'='*60}")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Strategies: {strategies or 'all'}")
        logger.info(f"Anonymized: {anonymized}")
        
        # Load input data
        logger.info("\n📥 Cargando datos procesados...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine data format
        if isinstance(data, dict) and 'transactions' in data:
            transactions = data['transactions']
            metadata = data.get('metadata', {})
        elif isinstance(data, list):
            transactions = data
            metadata = {}
        else:
            transactions = data
            metadata = {}
        
        logger.info(f"  ✓ {len(transactions)} transacciones cargadas")
        
        # Initialize chunk generator
        logger.info("\n🔧 Inicializando generador de chunks...")
        generator = AdaptiveChunkGenerator(
            json_path=input_file,
            anonymized=anonymized,
            strategies=strategies
        )
        
        # Load and process data
        logger.info("\n📊 Procesando datos...")
        generator.load_data()
        
        # Generate chunks
        logger.info("\n🔨 Generando chunks...")
        generator.generate_chunks()
        
        # Save different chunk outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all chunks
        all_chunks_file = output_path / f'all_chunks_{timestamp}.json'
        generator.save_chunks(str(all_chunks_file))
        
        # Save chunks by type
        logger.info("\n📁 Organizando chunks por tipo...")
        chunks_by_type = {}
        for chunk in generator.chunks:
            chunk_type = chunk['chunk_type']
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(chunk)
        
        for chunk_type, chunks in chunks_by_type.items():
            type_file = output_path / f'{chunk_type}_chunks_{timestamp}.json'
            with open(type_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'chunk_type': chunk_type,
                        'count': len(chunks),
                        'created_at': datetime.now().isoformat()
                    },
                    'chunks': chunks
                }, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"  ✓ {chunk_type}: {len(chunks)} chunks")
        
        # Generate summary report
        logger.info("\n📝 Generando reporte...")
        report = {
            'pipeline_run': {
                'timestamp': datetime.now().isoformat(),
                'input_file': input_file,
                'output_dir': str(output_dir),
                'anonymized': anonymized,
                'strategies': strategies or 'all'
            },
            'input_statistics': {
                'total_transactions': len(transactions),
                'date_range': {
                    'start': min(t.get('date', '') for t in transactions),
                    'end': max(t.get('date', '') for t in transactions)
                } if transactions else None
            },
            'output_statistics': {
                'total_chunks': len(generator.chunks),
                'chunks_by_type': {k: len(v) for k, v in chunks_by_type.items()},
                'chunks_by_strategy': dict(generator.chunk_stats)
            },
            'files_generated': {
                'all_chunks': str(all_chunks_file),
                'by_type': {k: str(output_path / f'{k}_chunks_{timestamp}.json') 
                           for k in chunks_by_type.keys()}
            }
        }
        
        report_file = output_path / f'chunking_report_{timestamp}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("✅ PIPELINE DE CHUNKING COMPLETADO")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Resumen:")
        logger.info(f"  • Transacciones procesadas: {len(transactions)}")
        logger.info(f"  • Chunks generados: {len(generator.chunks)}")
        logger.info(f"  • Tipos de chunks: {len(chunks_by_type)}")
        logger.info(f"  • Archivos generados: {len(chunks_by_type) + 2}")
        logger.info(f"\n📁 Archivos en: {output_dir}/")
        logger.info(f"  • all_chunks_{timestamp}.json")
        logger.info(f"  • chunking_report_{timestamp}.json")
        for chunk_type in chunks_by_type.keys():
            logger.info(f"  • {chunk_type}_chunks_{timestamp}.json")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error en pipeline de chunking: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run chunking pipeline on processed transactions'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSON file with processed transactions'
    )
    parser.add_argument(
        '--output', '-o',
        default='output_chunks',
        help='Output directory for chunks'
    )
    parser.add_argument(
        '--strategies', '-s',
        nargs='+',
        choices=AdaptiveChunkGenerator.CHUNK_STRATEGIES,
        help='Chunking strategies to use (default: all)'
    )
    parser.add_argument(
        '--anonymized',
        action='store_true',
        help='Data has been anonymized'
    )
    parser.add_argument(
        '--pipeline-output',
        help='Use output from main pipeline (finds latest)'
    )
    
    args = parser.parse_args()
    
    # If pipeline output specified, find the latest files
    if args.pipeline_output:
        pipeline_dir = Path(args.pipeline_output)
        if pipeline_dir.exists():
            # Find latest anonymized or categorized file
            json_files = list(pipeline_dir.glob('transactions_*.json'))
            if json_files:
                # Prefer anonymized over categorized
                anon_files = [f for f in json_files if 'anonymized' in f.name]
                cat_files = [f for f in json_files if 'categorized' in f.name]
                
                if anon_files:
                    input_file = str(max(anon_files, key=lambda f: f.stat().st_mtime))
                    logger.info(f"Using anonymized file: {input_file}")
                    args.anonymized = True
                elif cat_files:
                    input_file = str(max(cat_files, key=lambda f: f.stat().st_mtime))
                    logger.info(f"Using categorized file: {input_file}")
                else:
                    input_file = str(max(json_files, key=lambda f: f.stat().st_mtime))
                    logger.info(f"Using latest JSON file: {input_file}")
                
                args.input = input_file
            else:
                logger.error(f"No transaction JSON files found in {pipeline_dir}")
                sys.exit(1)
        else:
            logger.error(f"Pipeline output directory not found: {pipeline_dir}")
            sys.exit(1)
    
    # Run pipeline
    run_chunking_pipeline(
        input_file=args.input,
        output_dir=args.output,
        strategies=args.strategies,
        anonymized=args.anonymized
    )


if __name__ == "__main__":
    main()