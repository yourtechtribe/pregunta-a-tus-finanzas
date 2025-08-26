#!/usr/bin/env python3
"""
Pipeline principal para procesar extractos bancarios con RAG
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import os

# Mapeo de categorías inglés -> español para el RAG
# Sincronizado con las categorías del BBVA extractor
CATEGORY_MAPPING = {
    # Alimentación
    'Groceries': 'alimentacion',
    'Food & Dining': 'restaurantes',
    
    # Transporte
    'Transportation': 'transporte',
    
    # Compras y entretenimiento
    'Shopping': 'compras',
    'Entertainment': 'ocio',
    'Sports': 'deportes',
    
    # Salud y hogar
    'Healthcare': 'salud',
    'Housing': 'vivienda',
    'Utilities': 'servicios_basicos',
    'Services': 'servicios',
    
    # Educación y tecnología
    'Education': 'educacion',
    'Tech & Software': 'tecnologia',
    
    # Finanzas
    'Income': 'ingresos',
    'Savings': 'ahorro',
    'Loan': 'prestamos',
    'Taxes': 'impuestos',
    'Fees': 'comisiones',
    
    # Transferencias
    'Transfers': 'transferencias',
    'Internal Transfer': 'transferencia_interna',
    
    # Otros
    'ATM': 'efectivo',
    'Donations': 'donaciones',
    'Vending': 'vending',
    'Other': 'otros'
}

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.extractors.bbva_extractor import BBVAExtractor
from src.processors.adaptive_anonymizer import AdaptiveAnonymizer
from src.rag.lightrag_implementation import PersonalFinanceLightRAG

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Procesar extracto bancario y construir sistema RAG'
    )
    parser.add_argument(
        '--bank', 
        type=str, 
        required=True,
        choices=['bbva'],  # Añadir más bancos cuando estén disponibles
        help='Banco del extracto'
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Ruta al archivo del extracto (CSV/Excel)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Carpeta de salida para resultados'
    )
    parser.add_argument(
        '--skip-anonymization',
        action='store_true',
        help='Saltar el paso de anonimización (solo para testing)'
    )
    parser.add_argument(
        '--categorization',
        type=str,
        choices=['rules', 'gpt5-nano', 'auto'],
        default='rules',
        help='Método de categorización: rules (local/gratis), gpt5-nano (LLM/coste), auto (intenta GPT5, fallback a rules)'
    )
    
    args = parser.parse_args()
    
    # Verificar que el archivo existe
    input_file = Path(args.file)
    if not input_file.exists():
        logger.error(f"El archivo {input_file} no existe")
        sys.exit(1)
    
    # Crear carpeta de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Extracción
        logger.info(f"📊 Extrayendo transacciones de {args.bank.upper()}...")
        logger.info(f"📂 Método de categorización: {args.categorization.upper()}")
        
        if args.bank == 'bbva':
            extractor = BBVAExtractor()
        
        # Configurar si usar GPT para categorización
        use_gpt = False
        if args.categorization in ['gpt5-nano', 'auto']:
            # Verificar si hay API key
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'sk-proj-test-key':
                use_gpt = True
                logger.info("🤖 Usando GPT-5-nano para categorización inteligente")
            else:
                if args.categorization == 'gpt5-nano':
                    logger.error("❌ OPENAI_API_KEY no configurada. Configure la API key en .env")
                    sys.exit(1)
                else:
                    logger.warning("⚠️ OPENAI_API_KEY no configurada. Usando reglas locales")
                    use_gpt = False
        
        # Extraer con el método apropiado
        if use_gpt:
            result = extractor.extract(str(input_file), use_gpt=True)
        else:
            result = extractor.extract(str(input_file), use_gpt=False)
        transactions = result.get('transactions', [])
        
        logger.info(f"✅ Extraídas {len(transactions)} transacciones")
        
        # Mostrar información de categorización
        if 'categorization_method' in result.get('metadata', {}):
            method = result['metadata']['categorization_method']
            logger.info(f"🎯 Método usado: {method}")
            
            if 'gpt' in method.lower():
                # Estimar coste (GPT-5-nano: $0.05/1M input, $0.40/1M output)
                # Aproximadamente 50 tokens por transacción
                tokens_estimate = len(transactions) * 50
                cost_estimate = (tokens_estimate / 1_000_000) * 0.05
                logger.info(f"💰 Coste estimado: ${cost_estimate:.4f} USD")
        
        # Mostrar estadísticas de categorización
        if transactions:
            categories = {}
            for t in transactions:
                cat = t.get('category', 'Other')
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info(f"📋 Categorías detectadas:")
            for cat, count in sorted(categories.items()):
                logger.info(f"   - {cat}: {count} transacciones")
        
        # Convertir formato si es necesario
        if transactions and 'Fecha' in transactions[0]:
            # Formato directo de BBVA
            converted_transactions = []
            for t in transactions:
                converted_transactions.append({
                    'date': t.get('Fecha', ''),
                    'description': t.get('Concepto', ''),
                    'amount': float(str(t.get('Importe', '0')).replace(',', '.').replace('EUR', '').strip()),
                    'category': CATEGORY_MAPPING.get(t.get('Categoría', 'Other'), 'otros'),
                    'balance': float(str(t.get('Saldo', '0')).replace(',', '.').replace('EUR', '').strip()) if t.get('Saldo') else 0
                })
            transactions = converted_transactions
        
        # Guardar transacciones originales
        with open(output_dir / 'transactions_raw.json', 'w', encoding='utf-8') as f:
            json.dump(transactions, f, ensure_ascii=False, indent=2, default=str)
        
        # 2. Anonimización
        if not args.skip_anonymization:
            logger.info("🔒 Anonimizando datos sensibles...")
            anonymizer = AdaptiveAnonymizer()
            
            for transaction in transactions:
                if 'description' in transaction:
                    result = anonymizer.anonymize_text(transaction['description'])
                    transaction['description'] = result.anonymized_text
                    transaction['_anonymization_metadata'] = {
                        'entities_found': len(result.entities_found),
                        'confidence': result.confidence,
                        'method': result.method_used
                    }
            
            logger.info("✅ Datos anonimizados exitosamente")
            
            # Guardar transacciones anonimizadas
            with open(output_dir / 'transactions_anonymized.json', 'w', encoding='utf-8') as f:
                json.dump(transactions, f, ensure_ascii=False, indent=2, default=str)
        
        # 3. Construcción del grafo RAG
        logger.info("🔨 Construyendo grafo de conocimiento con LightRAG...")
        
        # Configurar LightRAG
        rag_config = {
            'working_dir': str(output_dir / 'lightrag_db'),
            'chunk_token_size': 250,
            'llm_model_max_async': 8,
            'entity_extract_max_gleaning': 1,
            'llm_model_name': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
            'embedding_model_name': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        }
        
        # Por ahora usar implementación simplificada sin LightRAG completo
        rag = PersonalFinanceLightRAG()
        
        # Guardar transacciones para RAG
        rag_data = []
        for i, t in enumerate(transactions):
            rag_data.append({
                'id': f'tx_{i}',
                'date': t.get('date', '2025-07-01'),
                'metadata': {
                    'description_original': t.get('description', ''),
                    'amount': t.get('amount', 0),
                    'category': CATEGORY_MAPPING.get(t.get('category', 'Other'), 'otros'),
                    'transaction_type': 'gasto' if t.get('amount', 0) < 0 else 'ingreso',
                    'is_recurring': False,
                    'month': 7,
                    'year': 2025,
                    'is_weekend': False
                }
            })
        
        # Guardar JSON temporal para RAG
        rag_json_path = output_dir / 'transactions_for_rag.json'
        with open(rag_json_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)
        
        # Cargar en RAG
        rag.load_transactions(str(rag_json_path))
        
        logger.info("✅ Grafo de conocimiento construido")
        
        # 4. Ejemplo de consultas
        logger.info("\n📝 Ejecutando consultas de ejemplo...")
        
        example_queries = [
            "¿Cuál fue mi gasto total este mes?",
            "¿Cuánto gasté en alimentación?",
            "¿Cuáles son mis gastos recurrentes?"
        ]
        
        results = []
        for query in example_queries:
            logger.info(f"\nConsulta: {query}")
            response = rag.query(query)
            if isinstance(response, dict):
                response = response.get('answer', str(response))
            logger.info(f"Respuesta: {response[:200]}...")  # Primeros 200 caracteres
            results.append({
                'query': query,
                'response': response
            })
        
        # Guardar resultados
        with open(output_dir / 'query_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Pipeline completado. Resultados en: {output_dir}")
        logger.info(f"   - Transacciones crudas: transactions_raw.json")
        logger.info(f"   - Transacciones anonimizadas: transactions_anonymized.json")
        logger.info(f"   - Resultados de consultas: query_results.json")
        logger.info(f"   - Base de datos LightRAG: lightrag_db/")
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()