#!/usr/bin/env python3
"""
Enhanced Pipeline with Multi-Agent System for OTHER category refinement
Flujo correcto:
1. BBVAExtractor categoriza primero (reglas o GPT-5-nano)
2. Solo las categorías "OTHER" pasan al sistema multi-agente
3. El multi-agente es un paso de refinamiento, no la primera opción
"""

import argparse
import json
import logging
import asyncio
from pathlib import Path
import sys
import os
from datetime import datetime

# Mapeo de categorías inglés -> español para el RAG
CATEGORY_MAPPING = {
    'Groceries': 'alimentacion',
    'Food & Dining': 'restaurantes',
    'Transportation': 'transporte',
    'Shopping': 'compras',
    'Entertainment': 'ocio',
    'Sports': 'deportes',
    'Healthcare': 'salud',
    'Housing': 'vivienda',
    'Utilities': 'servicios_basicos',
    'Services': 'servicios',
    'Education': 'educacion',
    'Tech & Software': 'tecnologia',
    'Income': 'ingresos',
    'Savings': 'ahorro',
    'Loan': 'prestamos',
    'Taxes': 'impuestos',
    'Fees': 'comisiones',
    'Transfers': 'transferencias',
    'Internal Transfer': 'transferencia_interna',
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
from src.agents.categorization_agent import SimplifiedMultiAgentCategorizer as MultiAgentCategorizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def refine_others_with_agents(transactions):
    """
    Refina las transacciones marcadas como 'Other' usando el sistema multi-agente
    
    Args:
        transactions: Lista de transacciones del BBVAExtractor
    
    Returns:
        Lista de transacciones con categorías refinadas
    """
    # Contar cuántas transacciones son "Other"
    others_count = sum(1 for t in transactions if t.get('category') == 'Other')
    
    if others_count == 0:
        logger.info("✅ No hay transacciones 'Other' para refinar")
        return transactions
    
    logger.info(f"🤖 Encontradas {others_count} transacciones 'Other' para refinar con multi-agente")
    
    # Inicializar el sistema multi-agente
    categorizer = MultiAgentCategorizer(cache_file="data/merchant_cache.json")
    
    # Procesar solo las transacciones "Other"
    improved_results = await categorizer.process_batch(transactions, only_others=True)
    
    # Actualizar las transacciones originales con las categorías mejoradas
    for result in improved_results:
        # Encontrar y actualizar la transacción original
        for t in transactions:
            if (t.get('description') == result['transaction'].get('description') and
                t.get('amount') == result['transaction'].get('amount')):
                # Solo actualizar si la nueva categoría es mejor que "Other"
                if result['category'] != 'Other' and result['confidence'] > 0.5:
                    t['category'] = result['category']
                    t['category_confidence'] = result['confidence']
                    t['category_method'] = f"multi_agent_{result['method']}"
                    logger.debug(f"  ✓ '{t['description']}' -> {result['category']} ({result['confidence']:.0%})")
    
    # Contar mejoras
    new_others_count = sum(1 for t in transactions if t.get('category') == 'Other')
    improved = others_count - new_others_count
    
    logger.info(f"✅ Multi-agente refinó {improved}/{others_count} transacciones 'Other'")
    
    return transactions


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline mejorado con sistema multi-agente para refinar categorías'
    )
    parser.add_argument(
        '--bank', 
        type=str, 
        required=True,
        choices=['bbva'],
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
        help='Método de categorización inicial: rules (local/gratis), gpt5-nano (LLM/coste), auto (intenta GPT5, fallback a rules)'
    )
    parser.add_argument(
        '--use-agents',
        action='store_true',
        default=True,
        help='Usar sistema multi-agente para refinar categorías "Other" (activado por defecto)'
    )
    parser.add_argument(
        '--no-agents',
        action='store_true',
        help='Desactivar el sistema multi-agente'
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
    
    # Determinar si usar agentes
    use_agents = not args.no_agents
    
    try:
        # ========== FASE 1: EXTRACCIÓN Y CATEGORIZACIÓN INICIAL ==========
        logger.info(f"📊 FASE 1: Extrayendo transacciones de {args.bank.upper()}...")
        logger.info(f"📂 Método de categorización inicial: {args.categorization.upper()}")
        
        if args.bank == 'bbva':
            extractor = BBVAExtractor()
        
        # Configurar si usar GPT para categorización inicial
        use_gpt = False
        if args.categorization in ['gpt5-nano', 'auto']:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'sk-proj-test-key':
                use_gpt = True
                logger.info("🤖 Usando GPT-5-nano para categorización inicial")
            else:
                if args.categorization == 'gpt5-nano':
                    logger.error("❌ OPENAI_API_KEY no configurada")
                    sys.exit(1)
                else:
                    logger.warning("⚠️ OPENAI_API_KEY no configurada. Usando reglas locales")
                    use_gpt = False
        
        # Extraer con el método apropiado
        result = extractor.extract(str(input_file), use_gpt=use_gpt)
        transactions = result.get('transactions', [])
        
        logger.info(f"✅ Extraídas {len(transactions)} transacciones")
        
        # Mostrar estadísticas de categorización inicial
        categories_initial = {}
        for t in transactions:
            cat = t.get('category', 'Other')
            categories_initial[cat] = categories_initial.get(cat, 0) + 1
        
        logger.info(f"📋 Categorías detectadas (inicial):")
        for cat, count in sorted(categories_initial.items()):
            percentage = (count / len(transactions)) * 100
            logger.info(f"   - {cat}: {count} ({percentage:.1f}%)")
        
        others_initial = categories_initial.get('Other', 0)
        
        # ========== FASE 2: REFINAMIENTO CON MULTI-AGENTE (OPCIONAL) ==========
        if use_agents and others_initial > 0:
            logger.info(f"\n🔄 FASE 2: Refinamiento con sistema multi-agente...")
            logger.info(f"   {others_initial} transacciones 'Other' serán analizadas")
            
            # Ejecutar refinamiento asíncrono
            transactions = asyncio.run(refine_others_with_agents(transactions))
            
            # Mostrar estadísticas después del refinamiento
            categories_refined = {}
            for t in transactions:
                cat = t.get('category', 'Other')
                categories_refined[cat] = categories_refined.get(cat, 0) + 1
            
            logger.info(f"\n📋 Categorías después del refinamiento:")
            for cat, count in sorted(categories_refined.items()):
                percentage = (count / len(transactions)) * 100
                change = count - categories_initial.get(cat, 0)
                change_str = f" ({change:+d})" if change != 0 else ""
                logger.info(f"   - {cat}: {count} ({percentage:.1f}%){change_str}")
            
            others_refined = categories_refined.get('Other', 0)
            improvement_percentage = ((others_initial - others_refined) / others_initial * 100) if others_initial > 0 else 0
            logger.info(f"\n✅ Mejora: 'Other' reducido de {others_initial} a {others_refined} ({improvement_percentage:.1f}% de mejora)")
        else:
            if not use_agents:
                logger.info("ℹ️ Sistema multi-agente desactivado")
            elif others_initial == 0:
                logger.info("✅ No hay transacciones 'Other' para refinar")
        
        # Guardar transacciones procesadas
        with open(output_dir / 'transactions_categorized.json', 'w', encoding='utf-8') as f:
            json.dump(transactions, f, ensure_ascii=False, indent=2, default=str)
        
        # ========== FASE 3: ANONIMIZACIÓN (OPCIONAL) ==========
        if not args.skip_anonymization:
            logger.info("\n🔒 FASE 3: Anonimizando datos sensibles...")
            anonymizer = AdaptiveAnonymizer(enable_llm=False)  # Solo usar reglas por ahora
            
            anonymized_count = 0
            entities_total = 0
            
            for transaction in transactions:
                if 'description' in transaction:
                    original = transaction['description']
                    # Detectar entidades primero
                    entities, confidence = anonymizer.process(original, use_adaptive=False)
                    
                    if entities:
                        # Anonimizar el texto
                        anonymized = anonymizer.anonymize_text(original, method="mask")
                        transaction['description_anonymized'] = anonymized
                        transaction['_anonymization_metadata'] = {
                            'original': original,
                            'entities_found': len(entities),
                            'entities': [{'type': e['entity_type'], 'text': e['text']} for e in entities],
                            'confidence': confidence
                        }
                        anonymized_count += 1
                        entities_total += len(entities)
                    else:
                        # No hay datos sensibles, mantener original
                        transaction['description_anonymized'] = original
            
            logger.info(f"✅ Anonimizadas {anonymized_count} transacciones con {entities_total} entidades sensibles")
            
            # Guardar transacciones anonimizadas
            with open(output_dir / 'transactions_anonymized.json', 'w', encoding='utf-8') as f:
                json.dump(transactions, f, ensure_ascii=False, indent=2, default=str)
        
        # ========== FASE 4: CONSTRUCCIÓN DEL GRAFO RAG ==========
        logger.info("\n🔨 FASE 4: Construyendo grafo de conocimiento con LightRAG...")
        
        rag = PersonalFinanceLightRAG()
        
        # Preparar datos para RAG
        rag_data = []
        for i, t in enumerate(transactions):
            # Skip internal transfers for spending analysis
            if t.get('category') == 'Internal Transfer':
                continue
                
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
                    'is_weekend': False,
                    'confidence': t.get('category_confidence', 1.0),
                    'method': t.get('category_method', 'rules')
                }
            })
        
        # Guardar JSON para RAG
        rag_json_path = output_dir / 'transactions_for_rag.json'
        with open(rag_json_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)
        
        # Cargar en RAG
        rag.load_transactions(str(rag_json_path))
        
        logger.info("✅ Grafo de conocimiento construido")
        
        # ========== FASE 5: CONSULTAS DE EJEMPLO ==========
        logger.info("\n📝 FASE 5: Ejecutando consultas de ejemplo...")
        
        example_queries = [
            "¿Cuál fue mi gasto total este mes?",
            "¿Cuánto gasté en alimentación?",
            "¿Cuáles son mis gastos recurrentes?",
            "¿En qué categoría gasto más dinero?"
        ]
        
        results = []
        for query in example_queries:
            logger.info(f"\nConsulta: {query}")
            response = rag.query(query)
            if isinstance(response, dict):
                response = response.get('answer', str(response))
            logger.info(f"Respuesta: {response[:200]}...")
            results.append({
                'query': query,
                'response': response
            })
        
        # Guardar resultados
        with open(output_dir / 'query_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # ========== RESUMEN FINAL ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ PIPELINE COMPLETADO CON ÉXITO")
        logger.info(f"{'='*60}")
        logger.info(f"📁 Resultados guardados en: {output_dir}")
        logger.info(f"   - transactions_categorized.json: Transacciones categorizadas")
        if not args.skip_anonymization:
            logger.info(f"   - transactions_anonymized.json: Transacciones anonimizadas")
        logger.info(f"   - transactions_for_rag.json: Datos para RAG")
        logger.info(f"   - query_results.json: Resultados de consultas")
        
        # Estadísticas finales
        logger.info(f"\n📊 ESTADÍSTICAS FINALES:")
        logger.info(f"   - Total transacciones: {len(transactions)}")
        logger.info(f"   - Categorías únicas: {len(set(t.get('category', 'Other') for t in transactions))}")
        
        # Calcular accuracy si no hay "Other"
        final_others = sum(1 for t in transactions if t.get('category') == 'Other')
        accuracy = ((len(transactions) - final_others) / len(transactions)) * 100
        logger.info(f"   - Precisión de categorización: {accuracy:.1f}%")
        
        if use_agents and others_initial > 0:
            logger.info(f"   - Mejora por multi-agente: {others_initial - final_others} transacciones")
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()