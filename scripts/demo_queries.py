#!/usr/bin/env python3
"""
Demo rápida del sistema de consultas financieras
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.lightrag_simple import SimpleLightRAG
import json


def run_demo():
    """Ejecuta una demo con consultas seleccionadas"""
    
    print("\n" + "="*70)
    print("🎯 DEMO - PREGUNTA A TUS FINANZAS")
    print("="*70)
    
    # Inicializar sistema
    print("\n📦 Inicializando sistema...")
    rag = SimpleLightRAG()
    
    # Cargar datos
    embeddings_path = 'data/embeddings/chunks_with_embeddings_lightrag.json'
    transactions_path = 'output/transactions_categorized.json'
    
    rag.load_chunks_with_embeddings(embeddings_path, transactions_path)
    print(f"✅ Sistema listo: {len(rag.chunks)} chunks, {len(rag.transactions)} transacciones")
    
    # Consultas de demostración
    demo_queries = [
        ("💰 AGREGACIÓN", "¿Cuánto gasté en Groceries?"),
        ("📊 ANÁLISIS", "¿Qué patrones de gasto tengo?"),
        ("🔍 BÚSQUEDA", "Muestra mis gastos de julio"),
        ("💡 AHORRO", "¿Dónde puedo ahorrar dinero?"),
        ("📈 TENDENCIA", "¿Cuál es la tendencia de mis gastos?")
    ]
    
    print("\n" + "="*70)
    print("🔍 EJECUTANDO CONSULTAS DE DEMOSTRACIÓN")
    print("="*70)
    
    results = []
    
    for category, query in demo_queries:
        print(f"\n{category}")
        print(f"📝 Consulta: {query}")
        print("-"*50)
        
        # Ejecutar consulta
        response = rag.query(query, use_chunks=True)
        
        # Mostrar resultados
        print(f"✅ Tipo detectado: {response.get('type', 'unknown')}")
        
        if 'answer' in response:
            print(f"📊 Respuesta: {response['answer']}")
        
        # Detalles específicos según tipo
        if response['type'] == 'aggregation' and 'value' in response:
            if isinstance(response['value'], dict):
                for k, v in response['value'].items():
                    print(f"   {k}: €{v:.2f}")
            else:
                print(f"   Valor: €{response['value']:.2f}")
            
            if 'details' in response:
                details = response['details']
                print(f"   Transacciones: {details.get('num_transactions', 0)}")
                if details.get('category'):
                    print(f"   Categoría: {details['category']}")
        
        elif response['type'] == 'search' and 'results' in response:
            print(f"   Resultados encontrados: {len(response['results'])}")
            if response['results']:
                # Mostrar primer resultado
                first = response['results'][0]
                if 'content' in first:
                    preview = first['content'][:100] + '...'
                    print(f"   Ejemplo: {preview}")
        
        elif response['type'] == 'analysis':
            if 'patterns' in response and isinstance(response['patterns'], dict):
                patterns = response['patterns']
                if 'top_categorias' in patterns:
                    print("   Top categorías de gasto:")
                    for cat, amount in list(patterns['top_categorias'].items())[:3]:
                        print(f"     • {cat}: €{amount:.2f}")
                if 'gasto_recurrente_total' in patterns:
                    print(f"   Gastos recurrentes: €{patterns['gasto_recurrente_total']:.2f}")
            
            if 'savings_opportunities' in response:
                savings = response['savings_opportunities']
                if isinstance(savings, dict) and 'total_potential' in savings:
                    print(f"   💡 Potencial de ahorro: €{savings['total_potential']:.2f}/mes")
            
            if 'trends' in response and isinstance(response['trends'], dict):
                trends = response['trends']
                if 'tendencia' in trends:
                    print(f"   📈 Tendencia: {trends['tendencia']}")
                if 'cambio_porcentual' in trends:
                    print(f"   Cambio: {trends['cambio_porcentual']:.1f}%")
        
        # Chunks utilizados
        if 'chunks_used' in response:
            print(f"   🔗 Chunks utilizados: {len(response['chunks_used'])}")
            for chunk_id in response['chunks_used'][:2]:
                print(f"      - {chunk_id}")
        
        # Guardar resultado
        results.append({
            'category': category,
            'query': query,
            'response_type': response.get('type'),
            'answer': response.get('answer', 'N/A')
        })
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ RESUMEN DE LA DEMO")
    print("="*70)
    
    # Estadísticas financieras
    if rag.df is not None:
        total_gastos = abs(rag.df[rag.df['amount'] < 0]['amount'].sum())
        total_ingresos = rag.df[rag.df['amount'] > 0]['amount'].sum()
        balance = total_ingresos - total_gastos
        
        print(f"\n💰 Resumen Financiero:")
        print(f"   Total gastos: €{total_gastos:.2f}")
        print(f"   Total ingresos: €{total_ingresos:.2f}")
        print(f"   Balance: €{balance:.2f}")
        
        # Top categorías
        gastos_df = rag.df[rag.df['amount'] < 0].copy()
        gastos_df['amount_abs'] = gastos_df['amount'].abs()
        top_cats = gastos_df.groupby('category')['amount_abs'].sum().nlargest(3)
        
        print(f"\n📊 Top 3 Categorías de Gasto:")
        for i, (cat, amount) in enumerate(top_cats.items(), 1):
            percentage = (amount / total_gastos) * 100
            print(f"   {i}. {cat}: €{amount:.2f} ({percentage:.1f}%)")
    
    print(f"\n🎯 Consultas ejecutadas: {len(demo_queries)}")
    print(f"📦 Chunks en el sistema: {len(rag.chunks)}")
    print(f"💾 Transacciones analizadas: {len(rag.transactions)}")
    
    # Guardar resultados
    output_file = 'output/demo_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Resultados guardados en: {output_file}")
    
    print("\n" + "="*70)
    print("🎉 ¡Demo completada exitosamente!")
    print("\n💡 Para usar el modo interactivo ejecuta:")
    print("   python3 scripts/query_finances.py")
    print("="*70)


if __name__ == "__main__":
    run_demo()