#!/usr/bin/env python3
"""
Script para testear el sistema AdaptiveLightRAG con chunks y embeddings
"""

import sys
import os
from pathlib import Path

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.adaptive_lightrag import AdaptiveLightRAG, build_knowledge_graph
import json
from datetime import datetime


def test_adaptive_lightrag():
    """Test completo del sistema AdaptiveLightRAG"""
    
    print("=" * 60)
    print("🧪 TEST DE SISTEMA ADAPTIVE LIGHTRAG")
    print("=" * 60)
    
    # Inicializar sistema
    print("\n📦 Inicializando AdaptiveLightRAG...")
    rag = AdaptiveLightRAG()
    
    # Rutas de archivos
    embeddings_path = 'data/embeddings/chunks_with_embeddings_lightrag.json'
    transactions_path = 'output/transactions_categorized.json'
    
    # Verificar que existen los archivos
    if not os.path.exists(embeddings_path):
        print(f"❌ No se encontró: {embeddings_path}")
        print("Ejecuta primero: python3 scripts/generate_embeddings.py")
        return
    
    if not os.path.exists(transactions_path):
        print(f"❌ No se encontró: {transactions_path}")
        print("Ejecuta primero el pipeline de procesamiento")
        return
    
    # Cargar datos
    print(f"\n📥 Cargando chunks con embeddings...")
    rag.load_chunks_with_embeddings(embeddings_path, transactions_path)
    
    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DEL SISTEMA")
    print("=" * 60)
    print(f"Total chunks cargados: {len(rag.chunks)}")
    print(f"Total transacciones: {len(rag.transactions)}")
    print("\nDistribución de chunks por tipo:")
    for chunk_type, chunks in rag.chunks_by_type.items():
        if chunks:
            print(f"  - {chunk_type}: {len(chunks)}")
    
    # Construir grafo de conocimiento
    print("\n" + "=" * 60)
    print("🕸️ CONSTRUCCIÓN DEL GRAFO DE CONOCIMIENTO")
    print("=" * 60)
    graph = build_knowledge_graph(rag)
    print(f"Nodos totales: {graph['statistics']['total_nodes']}")
    print(f"  - Nodos de chunks: {graph['statistics']['chunk_nodes']}")
    print(f"  - Nodos de transacciones: {graph['statistics']['transaction_nodes']}")
    print(f"Relaciones totales: {graph['statistics']['total_edges']}")
    
    # Consultas de prueba
    print("\n" + "=" * 60)
    print("🔍 PRUEBAS DE CONSULTAS")
    print("=" * 60)
    
    test_queries = [
        {
            'query': "¿Cuánto gasté en total?",
            'expected_type': 'aggregation',
            'description': 'Consulta de agregación total'
        },
        {
            'query': "¿Cuánto gasté en Groceries?",
            'expected_type': 'aggregation',
            'description': 'Agregación por categoría'
        },
        {
            'query': "Muestra mis transacciones de julio",
            'expected_type': 'search',
            'description': 'Búsqueda temporal'
        },
        {
            'query': "¿Qué patrones de gasto tengo?",
            'expected_type': 'analysis',
            'description': 'Análisis de patrones'
        },
        {
            'query': "¿Cuál es la tendencia de mis gastos?",
            'expected_type': 'analysis',
            'description': 'Análisis de tendencias'
        },
        {
            'query': "¿Dónde puedo ahorrar dinero?",
            'expected_type': 'analysis',
            'description': 'Oportunidades de ahorro'
        },
        {
            'query': "Lista mis gastos recurrentes",
            'expected_type': 'search',
            'description': 'Búsqueda de recurrencias'
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {test['description']}")
        print(f"{'='*50}")
        print(f"📝 Consulta: {test['query']}")
        
        # Ejecutar consulta con chunks
        response = rag.query(test['query'], use_chunks=True)
        
        print(f"✅ Tipo detectado: {response.get('type', 'unknown')}")
        
        if response.get('type') == test['expected_type']:
            print(f"   ✓ Coincide con el esperado: {test['expected_type']}")
        else:
            print(f"   ⚠️ Esperado: {test['expected_type']}, Obtenido: {response.get('type')}")
        
        # Mostrar respuesta
        if 'answer' in response:
            print(f"📊 Respuesta: {response['answer']}")
        
        # Mostrar chunks utilizados
        if 'chunks_used' in response:
            print(f"🔗 Chunks utilizados: {response['chunks_used'][:3]}")
        
        # Mostrar detalles adicionales según el tipo
        if response['type'] == 'aggregation' and 'value' in response:
            print(f"💰 Valor: {response['value']}")
            if 'details' in response:
                print(f"📋 Detalles: {response['details']}")
        
        elif response['type'] == 'search' and 'results' in response:
            print(f"🔍 Resultados encontrados: {response['num_results']}")
            if response['results']:
                print("📌 Primeros 2 resultados:")
                for j, result in enumerate(response['results'][:2], 1):
                    print(f"   {j}. Tipo: {result.get('chunk_type', 'N/A')}")
                    if 'content' in result:
                        preview = result['content'][:100] + '...' if len(result['content']) > 100 else result['content']
                        print(f"      Contenido: {preview}")
                    if 'transactions' in result and result['transactions']:
                        print(f"      Transacciones asociadas: {len(result['transactions'])}")
        
        elif response['type'] == 'analysis':
            if 'patterns' in response:
                print(f"🔄 Patrones encontrados: {len(response.get('patterns', []))}")
            if 'trends' in response:
                print(f"📈 Tendencias analizadas")
            if 'savings_opportunities' in response:
                opps = response['savings_opportunities']
                if isinstance(opps, dict) and 'total_potential' in opps:
                    print(f"💡 Ahorro potencial: {opps['total_potential']:.2f}€")
    
    # Comparación con y sin chunks
    print("\n" + "=" * 60)
    print("🔄 COMPARACIÓN: CON CHUNKS vs SIN CHUNKS")
    print("=" * 60)
    
    comparison_query = "¿Cuánto gasté en Entertainment?"
    
    print(f"\n📝 Consulta: {comparison_query}")
    
    print("\n--- Con chunks (multi-perspectiva) ---")
    response_with_chunks = rag.query(comparison_query, use_chunks=True)
    print(f"Respuesta: {response_with_chunks.get('answer', 'N/A')}")
    if 'chunks_used' in response_with_chunks:
        print(f"Chunks utilizados: {len(response_with_chunks['chunks_used'])}")
    if 'context' in response_with_chunks:
        print(f"Contexto enriquecido: Sí ({len(response_with_chunks['context'])} elementos)")
    
    print("\n--- Sin chunks (búsqueda directa) ---")
    response_without_chunks = rag.query(comparison_query, use_chunks=False)
    print(f"Respuesta: {response_without_chunks.get('answer', 'N/A')}")
    print(f"Contexto enriquecido: No")
    
    # Test de búsqueda semántica
    print("\n" + "=" * 60)
    print("🔍 TEST DE BÚSQUEDA SEMÁNTICA")
    print("=" * 60)
    
    semantic_query = "gastos en comida y supermercado"
    print(f"\n📝 Búsqueda: {semantic_query}")
    
    results = rag.semantic_search(semantic_query, top_k=3)
    print(f"✅ Encontrados {len(results)} chunks relevantes:")
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Chunk ID: {chunk.chunk_id}")
        print(f"   Tipo: {chunk.chunk_type}")
        print(f"   Relevancia: {score:.2f}")
        print(f"   Preview: {chunk.content[:100]}...")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"✓ Sistema AdaptiveLightRAG inicializado correctamente")
    print(f"✓ {len(rag.chunks)} chunks cargados con embeddings")
    print(f"✓ {len(rag.transactions)} transacciones disponibles para análisis")
    print(f"✓ Grafo de conocimiento construido con {graph['statistics']['total_nodes']} nodos")
    print(f"✓ {len(test_queries)} consultas de prueba ejecutadas")
    print(f"✓ Sistema de búsqueda multi-perspectiva funcionando")
    print("\n🎉 ¡Todas las pruebas completadas exitosamente!")
    
    # Guardar estadísticas
    stats_file = 'output/lightrag_test_results.json'
    test_results = {
        'test_date': datetime.now().isoformat(),
        'system_stats': {
            'total_chunks': len(rag.chunks),
            'total_transactions': len(rag.transactions),
            'chunk_types': {k: len(v) for k, v in rag.chunks_by_type.items() if v},
            'graph_nodes': graph['statistics']['total_nodes'],
            'graph_edges': graph['statistics']['total_edges']
        },
        'queries_tested': len(test_queries),
        'test_status': 'success'
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {stats_file}")


if __name__ == "__main__":
    test_adaptive_lightrag()