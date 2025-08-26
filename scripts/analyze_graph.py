#!/usr/bin/env python3
"""
Analiza el grafo de conocimiento generado por LightRAG
Muestra entidades, relaciones y estadísticas
"""

import json
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import re

def analyze_lightrag_graph(working_dir: str = "simple_rag_knowledge"):
    """
    Analiza el grafo de conocimiento financiero
    """
    working_dir = Path(working_dir)
    
    print("\n" + "="*60)
    print("📊 ANÁLISIS DEL GRAFO DE CONOCIMIENTO FINANCIERO")
    print("="*60)
    
    # Cargar el grafo GraphML
    graphml_path = working_dir / "graph_chunk_entity_relation.graphml"
    if graphml_path.exists():
        G = nx.read_graphml(str(graphml_path))
        print(f"\n✅ Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    else:
        print(f"❌ No se encontró el grafo en {graphml_path}")
        return
    
    # Analizar nodos
    print("\n🔍 ANÁLISIS DE NODOS (ENTIDADES)")
    print("-"*40)
    
    # Clasificar nodos por patrones
    entities_by_type = {
        'fechas': [],
        'montos': [],
        'categorías': [],
        'comercios': [],
        'períodos': [],
        'otros': []
    }
    
    for node in G.nodes():
        node_lower = node.lower()
        
        # Detectar tipos
        if re.search(r'\d{4}', node) or 'july' in node_lower or 'week' in node_lower:
            entities_by_type['fechas'].append(node)
        elif '€' in node or re.search(r'\d+\.\d+', node):
            entities_by_type['montos'].append(node)
        elif any(cat in node_lower for cat in ['groceries', 'housing', 'transport', 'entertainment', 'food', 'shopping']):
            entities_by_type['categorías'].append(node)
        elif any(merchant in node_lower for merchant in ['transfer', 'supermercado', 'amazon', 'netflix', 'alquiler', 'farmacia']):
            entities_by_type['comercios'].append(node)
        elif 'financial' in node_lower or 'summary' in node_lower or 'analysis' in node_lower:
            entities_by_type['períodos'].append(node)
        else:
            entities_by_type['otros'].append(node)
    
    # Mostrar entidades por tipo
    for tipo, entidades in entities_by_type.items():
        if entidades:
            print(f"\n📌 {tipo.upper()} ({len(entidades)} entidades):")
            for entity in sorted(entidades)[:10]:  # Mostrar máximo 10
                degree = G.degree(entity)
                print(f"  • {entity} ({degree} conexiones)")
            if len(entidades) > 10:
                print(f"  ... y {len(entidades)-10} más")
    
    # Nodos más importantes (por grado)
    print("\n⭐ TOP 10 NODOS MÁS CONECTADOS:")
    print("-"*40)
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for i, (node, degree) in enumerate(top_nodes, 1):
        # Obtener vecinos
        neighbors = list(G.neighbors(node))
        neighbor_preview = ", ".join(neighbors[:3])
        if len(neighbors) > 3:
            neighbor_preview += f"... ({len(neighbors)-3} más)"
        
        print(f"{i}. {node}")
        print(f"   Conexiones: {degree}")
        print(f"   Conectado con: {neighbor_preview}")
    
    # Analizar relaciones
    print("\n🔗 ANÁLISIS DE RELACIONES")
    print("-"*40)
    
    # Contar tipos de relaciones si hay atributos
    edge_labels = []
    for u, v, data in G.edges(data=True):
        if 'label' in data:
            edge_labels.append(data['label'])
        elif 'relationship' in data:
            edge_labels.append(data['relationship'])
    
    if edge_labels:
        label_counts = Counter(edge_labels)
        print("\nTipos de relaciones más comunes:")
        for label, count in label_counts.most_common(10):
            if label:
                print(f"  • {label}: {count} ocurrencias")
    
    # Componentes del grafo
    print("\n🌐 ESTRUCTURA DEL GRAFO")
    print("-"*40)
    
    # Componentes conectados
    if G.is_directed():
        num_components = nx.number_weakly_connected_components(G)
    else:
        num_components = nx.number_connected_components(G)
    
    print(f"Componentes conectados: {num_components}")
    
    # Densidad del grafo
    density = nx.density(G)
    print(f"Densidad del grafo: {density:.4f}")
    
    # Coeficiente de clustering
    if G.number_of_nodes() > 0:
        avg_clustering = nx.average_clustering(G)
        print(f"Coeficiente de clustering promedio: {avg_clustering:.4f}")
    
    # Detectar comunidades (si el grafo es suficientemente grande)
    if G.number_of_nodes() > 10:
        print("\n👥 DETECCIÓN DE COMUNIDADES")
        print("-"*40)
        
        # Convertir a grafo no dirigido si es necesario
        G_undirected = G.to_undirected() if G.is_directed() else G
        
        # Detectar comunidades usando el algoritmo de Louvain
        try:
            import community.community_louvain as community_louvain
            communities = community_louvain.best_partition(G_undirected)
            
            # Agrupar nodos por comunidad
            community_groups = defaultdict(list)
            for node, comm_id in communities.items():
                community_groups[comm_id].append(node)
            
            print(f"Comunidades detectadas: {len(community_groups)}")
            
            for comm_id, nodes in sorted(community_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                print(f"\nComunidad {comm_id} ({len(nodes)} nodos):")
                sample_nodes = nodes[:5]
                for node in sample_nodes:
                    print(f"  • {node}")
                if len(nodes) > 5:
                    print(f"  ... y {len(nodes)-5} más")
        except ImportError:
            print("Instala python-louvain para detectar comunidades: pip install python-louvain")
    
    # Métricas financieras extraídas
    print("\n💰 INFORMACIÓN FINANCIERA EXTRAÍDA")
    print("-"*40)
    
    # Buscar montos en el grafo
    montos_encontrados = []
    for node in G.nodes():
        # Buscar patrones de montos en euros
        monto_match = re.search(r'€([\d,]+\.?\d*)', node)
        if monto_match:
            try:
                monto = float(monto_match.group(1).replace(',', ''))
                montos_encontrados.append((node, monto))
            except:
                pass
    
    if montos_encontrados:
        montos_encontrados.sort(key=lambda x: x[1], reverse=True)
        print("\nMontos identificados:")
        total = 0
        for nombre, monto in montos_encontrados[:10]:
            print(f"  • {nombre}: €{monto:.2f}")
            total += monto
        if len(montos_encontrados) > 10:
            print(f"  ... y {len(montos_encontrados)-10} más")
        print(f"\nTotal visible: €{total:.2f}")
    
    # Resumen final
    print("\n📈 RESUMEN EJECUTIVO")
    print("="*60)
    print(f"• Total de entidades: {G.number_of_nodes()}")
    print(f"• Total de relaciones: {G.number_of_edges()}")
    print(f"• Nodo más conectado: {top_nodes[0][0]} ({top_nodes[0][1]} conexiones)")
    print(f"• Densidad del grafo: {density:.2%}")
    
    categorias_encontradas = len(entities_by_type['categorías'])
    comercios_encontrados = len(entities_by_type['comercios'])
    
    if categorias_encontradas > 0:
        print(f"• Categorías de gasto: {categorias_encontradas}")
    if comercios_encontrados > 0:
        print(f"• Comercios/Merchants: {comercios_encontrados}")
    
    print("\n💡 INSIGHTS")
    print("-"*40)
    
    # Insights basados en el análisis
    if 'July 2025' in degree_dict and degree_dict['July 2025'] > 5:
        print("• El período 'July 2025' es central en el grafo (muchas transacciones)")
    
    if 'Groceries' in degree_dict:
        print(f"• 'Groceries' está conectado con {degree_dict['Groceries']} entidades")
    
    if any('€' in node for node in G.nodes()):
        print("• Se identificaron múltiples montos monetarios en el grafo")
    
    if density < 0.1:
        print("• El grafo es disperso, sugiriendo diversidad en las transacciones")
    elif density > 0.3:
        print("• El grafo es denso, indicando muchas interrelaciones")
    
    # Guardar análisis en JSON
    analysis_results = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'density': density,
        'top_nodes': [(node, degree) for node, degree in top_nodes],
        'entities_by_type': {k: v[:10] for k, v in entities_by_type.items()},
        'financial_amounts': montos_encontrados[:10]
    }
    
    output_file = working_dir / "graph_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Análisis guardado en: {output_file}")


if __name__ == "__main__":
    analyze_lightrag_graph()