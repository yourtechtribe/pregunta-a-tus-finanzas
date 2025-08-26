#!/usr/bin/env python3
"""
Visualización interactiva del grafo de conocimiento financiero generado por LightRAG
"""

import json
import networkx as nx
from pyvis.network import Network
import webbrowser
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import re

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FinancialGraphVisualizer:
    """
    Visualizador del grafo de conocimiento financiero
    """
    
    def __init__(self, working_dir: str = "simple_rag_knowledge"):
        """
        Inicializa el visualizador
        
        Args:
            working_dir: Directorio donde está el grafo de LightRAG
        """
        self.working_dir = Path(working_dir)
        self.graph = None
        self.entities = {}
        self.relations = {}
        self.chunks = {}
        
        # Configuración de colores por tipo de entidad
        self.color_map = {
            # Entidades financieras
            "MERCHANT": "#FF6B6B",      # Rojo coral - Comercios
            "CATEGORY": "#4ECDC4",      # Turquesa - Categorías
            "AMOUNT": "#95E77E",        # Verde lima - Montos
            "DATE": "#FFD93D",          # Amarillo - Fechas
            "PERSON": "#6C5CE7",        # Púrpura - Personas
            "ORGANIZATION": "#A8E6CF",  # Verde pastel - Organizaciones
            "LOCATION": "#FFB6C1",      # Rosa - Ubicaciones
            "TRANSACTION": "#FFA500",   # Naranja - Transacciones
            "PATTERN": "#87CEEB",       # Azul cielo - Patrones
            # Nodos de chunk
            "CHUNK": "#808080",         # Gris - Chunks
            "default": "#C0C0C0"        # Plata - Por defecto
        }
        
        # Configuración de tamaños
        self.size_map = {
            "MERCHANT": 25,
            "CATEGORY": 30,
            "AMOUNT": 20,
            "PERSON": 35,
            "CHUNK": 15,
            "default": 20
        }
    
    def load_graph_data(self) -> bool:
        """
        Carga los datos del grafo desde los archivos de LightRAG
        
        Returns:
            True si se cargó correctamente
        """
        logger.info(f"📂 Cargando datos del grafo desde: {self.working_dir}")
        
        # Verificar que existe el directorio
        if not self.working_dir.exists():
            logger.error(f"❌ No se encontró el directorio: {self.working_dir}")
            return False
        
        # Intentar cargar el grafo GraphML si existe
        graphml_path = self.working_dir / "graph_chunk_entity_relation.graphml"
        if graphml_path.exists():
            try:
                self.graph = nx.read_graphml(str(graphml_path))
                logger.info(f"✅ Grafo GraphML cargado: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas")
                return True
            except Exception as e:
                logger.warning(f"⚠️ No se pudo cargar GraphML: {e}")
        
        # Si no hay GraphML, construir desde los archivos JSON
        return self.build_graph_from_json()
    
    def build_graph_from_json(self) -> bool:
        """
        Construye el grafo desde los archivos JSON de LightRAG
        
        Returns:
            True si se construyó correctamente
        """
        logger.info("🔨 Construyendo grafo desde archivos JSON...")
        
        self.graph = nx.Graph()
        
        # Cargar entidades
        entities_file = self.working_dir / "kv_store_full_entities.json"
        if entities_file.exists():
            with open(entities_file, 'r', encoding='utf-8') as f:
                self.entities = json.load(f)
            logger.info(f"  Entidades cargadas: {len(self.entities)}")
        
        # Cargar relaciones
        relations_file = self.working_dir / "kv_store_full_relations.json"
        if relations_file.exists():
            with open(relations_file, 'r', encoding='utf-8') as f:
                self.relations = json.load(f)
            logger.info(f"  Relaciones cargadas: {len(self.relations)}")
        
        # Cargar chunks
        chunks_file = self.working_dir / "kv_store_text_chunks.json"
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            logger.info(f"  Chunks cargados: {len(self.chunks)}")
        
        # Procesar entidades y añadir nodos
        self._process_entities()
        
        # Procesar relaciones y añadir aristas
        self._process_relations()
        
        # Procesar chunks si hay
        self._process_chunks()
        
        logger.info(f"✅ Grafo construido: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas")
        
        return self.graph.number_of_nodes() > 0
    
    def _process_entities(self):
        """Procesa las entidades y las añade como nodos al grafo"""
        for doc_id, doc_data in self.entities.items():
            if isinstance(doc_data, str):
                # Es contenido JSON como string
                try:
                    entities_data = json.loads(doc_data)
                    if 'entities' in entities_data:
                        for entity in entities_data['entities']:
                            self._add_entity_node(entity)
                except:
                    pass
            elif isinstance(doc_data, dict) and 'entities' in doc_data:
                for entity in doc_data['entities']:
                    self._add_entity_node(entity)
    
    def _add_entity_node(self, entity: Dict):
        """Añade un nodo de entidad al grafo"""
        entity_name = entity.get('entity_name', '')
        entity_type = entity.get('entity_type', 'default').upper()
        
        if entity_name:
            # Determinar el tipo financiero
            financial_type = self._classify_financial_entity(entity_name, entity_type)
            
            self.graph.add_node(
                entity_name,
                type=financial_type,
                original_type=entity_type,
                description=entity.get('description', ''),
                importance=entity.get('importance_score', 0.5)
            )
    
    def _classify_financial_entity(self, name: str, entity_type: str) -> str:
        """
        Clasifica una entidad en categorías financieras
        
        Args:
            name: Nombre de la entidad
            entity_type: Tipo original de la entidad
            
        Returns:
            Tipo financiero clasificado
        """
        name_lower = name.lower()
        
        # Patrones de clasificación
        if any(word in name_lower for word in ['€', 'euro', 'gasto', 'ingreso', 'total']):
            return "AMOUNT"
        elif any(word in name_lower for word in ['julio', 'agosto', 'semana', 'mes', '2025', 'fecha']):
            return "DATE"
        elif any(word in name_lower for word in ['groceries', 'housing', 'transport', 'entertainment', 'food']):
            return "CATEGORY"
        elif any(word in name_lower for word in ['transfer', 'supermercado', 'amazon', 'netflix', 'alquiler', 'mercadona']):
            return "MERCHANT"
        elif any(word in name_lower for word in ['recurrente', 'patrón', 'suscripción', 'mensual']):
            return "PATTERN"
        elif entity_type == "PERSON":
            return "PERSON"
        elif entity_type == "ORGANIZATION":
            return "ORGANIZATION"
        else:
            return entity_type.upper() if entity_type else "default"
    
    def _process_relations(self):
        """Procesa las relaciones y las añade como aristas al grafo"""
        for doc_id, doc_data in self.relations.items():
            if isinstance(doc_data, str):
                try:
                    relations_data = json.loads(doc_data)
                    if 'relationships' in relations_data:
                        for relation in relations_data['relationships']:
                            self._add_relation_edge(relation)
                except:
                    pass
            elif isinstance(doc_data, dict) and 'relationships' in doc_data:
                for relation in doc_data['relationships']:
                    self._add_relation_edge(relation)
    
    def _add_relation_edge(self, relation: Dict):
        """Añade una arista de relación al grafo"""
        source = relation.get('source_entity', '')
        target = relation.get('target_entity', '')
        relationship = relation.get('relationship', '')
        
        if source and target:
            # Asegurarse de que los nodos existen
            if source not in self.graph:
                self.graph.add_node(source, type="default")
            if target not in self.graph:
                self.graph.add_node(target, type="default")
            
            # Añadir la arista
            self.graph.add_edge(
                source,
                target,
                relationship=relationship,
                strength=relation.get('relationship_strength', 0.5)
            )
    
    def _process_chunks(self):
        """Procesa los chunks y los conecta con las entidades"""
        chunk_count = 0
        for chunk_id, chunk_data in self.chunks.items():
            if isinstance(chunk_data, str):
                # Es el contenido del chunk
                chunk_node_id = f"chunk_{chunk_count}"
                self.graph.add_node(
                    chunk_node_id,
                    type="CHUNK",
                    content=chunk_data[:100] + "..." if len(chunk_data) > 100 else chunk_data
                )
                chunk_count += 1
    
    def create_visualization(self, output_path: str = "outputs/financial_graph.html"):
        """
        Crea la visualización interactiva del grafo
        
        Args:
            output_path: Ruta donde guardar el HTML
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.error("❌ No hay grafo para visualizar")
            return
        
        logger.info(f"🎨 Creando visualización con {self.graph.number_of_nodes()} nodos...")
        
        # Crear red de visualización
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#1a1a1a",
            font_color="white",
            notebook=False,
            cdn_resources='in_line'
        )
        
        # Configurar física del grafo
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 14,
              "face": "Arial",
              "color": "white"
            },
            "borderWidth": 2,
            "shadow": true
          },
          "edges": {
            "color": {
              "inherit": true,
              "opacity": 0.7
            },
            "smooth": {
              "type": "continuous"
            },
            "width": 2,
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
          }
        }
        """)
        
        # Añadir nodos
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'default')
            
            # Preparar tooltip
            tooltip_parts = [f"<b>{node}</b>"]
            tooltip_parts.append(f"Tipo: {node_type}")
            if attrs.get('description'):
                tooltip_parts.append(f"Descripción: {attrs['description'][:100]}")
            if attrs.get('importance'):
                tooltip_parts.append(f"Importancia: {attrs['importance']:.2f}")
            
            # Determinar tamaño basado en importancia y tipo
            base_size = self.size_map.get(node_type, self.size_map['default'])
            importance = attrs.get('importance', 0.5)
            node_size = base_size * (0.5 + importance)
            
            # Añadir nodo a la visualización
            net.add_node(
                node,
                label=self._truncate_label(node),
                title="<br>".join(tooltip_parts),
                color=self.color_map.get(node_type, self.color_map['default']),
                size=node_size,
                shape="dot" if node_type != "CHUNK" else "square",
                borderWidth=3 if importance > 0.7 else 2
            )
        
        # Añadir aristas
        for source, target, attrs in self.graph.edges(data=True):
            relationship = attrs.get('relationship', '')
            strength = attrs.get('strength', 0.5)
            
            # Tooltip para la arista
            edge_tooltip = f"{source} → {target}"
            if relationship:
                edge_tooltip += f"<br>Relación: {relationship}"
            
            net.add_edge(
                source,
                target,
                title=edge_tooltip,
                width=1 + strength * 3,  # Grosor basado en fuerza
                physics=True
            )
        
        # Crear directorio de salida si no existe
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar visualización
        net.save_graph(str(output_path))
        logger.info(f"✅ Visualización guardada en: {output_path}")
        
        # Generar estadísticas
        self._print_statistics()
        
        return str(output_path)
    
    def _truncate_label(self, label: str, max_length: int = 20) -> str:
        """Trunca etiquetas largas para mejor visualización"""
        if len(label) > max_length:
            return label[:max_length-3] + "..."
        return label
    
    def _print_statistics(self):
        """Imprime estadísticas del grafo"""
        logger.info("\n📊 ESTADÍSTICAS DEL GRAFO:")
        logger.info(f"  Total nodos: {self.graph.number_of_nodes()}")
        logger.info(f"  Total aristas: {self.graph.number_of_edges()}")
        
        # Contar por tipo
        type_counts = {}
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'default')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        logger.info("\n  Distribución por tipo:")
        for node_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {node_type}: {count}")
        
        # Nodos más conectados
        if self.graph.number_of_nodes() > 0:
            degree_dict = dict(self.graph.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            
            logger.info("\n  Top 5 nodos más conectados:")
            for node, degree in top_nodes:
                node_type = self.graph.nodes[node].get('type', 'default')
                logger.info(f"    {node} ({node_type}): {degree} conexiones")
    
    def open_in_browser(self, html_path: str):
        """Abre la visualización en el navegador"""
        try:
            full_path = os.path.abspath(html_path)
            webbrowser.open(f'file://{full_path}')
            logger.info("🌐 Abriendo visualización en el navegador...")
        except Exception as e:
            logger.error(f"❌ Error al abrir el navegador: {e}")


def main():
    """Función principal"""
    logger.info("\n" + "="*60)
    logger.info("🌐 VISUALIZACIÓN DEL GRAFO DE CONOCIMIENTO FINANCIERO")
    logger.info("="*60)
    
    # Crear visualizador
    visualizer = FinancialGraphVisualizer(working_dir="simple_rag_knowledge")
    
    # Cargar datos del grafo
    if not visualizer.load_graph_data():
        logger.error("❌ No se pudo cargar el grafo")
        logger.info("Asegúrate de haber ejecutado: python3 scripts/lightrag_simple.py")
        return
    
    # Crear visualización
    output_path = visualizer.create_visualization("outputs/financial_knowledge_graph.html")
    
    if output_path:
        # Abrir en navegador
        visualizer.open_in_browser(output_path)
        
        logger.info("\n✨ Visualización completada!")
        logger.info(f"📁 Archivo HTML: {output_path}")
        logger.info("\n💡 Tips para la visualización:")
        logger.info("  • 🖱️ Arrastra los nodos para reorganizar")
        logger.info("  • 🔍 Usa la rueda del ratón para zoom")
        logger.info("  • 👆 Haz clic en un nodo para ver detalles")
        logger.info("  • 🎨 Los colores indican tipos de entidades:")
        logger.info("    - Rojo: Comercios")
        logger.info("    - Turquesa: Categorías")
        logger.info("    - Verde: Montos")
        logger.info("    - Amarillo: Fechas")
        logger.info("    - Púrpura: Personas")


if __name__ == "__main__":
    main()