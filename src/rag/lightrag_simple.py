#!/usr/bin/env python3
"""
Implementación simplificada de LightRAG para el proyecto Pregunta-tus-Finanzas
Basada en la documentación oficial de HKUDS/LightRAG
"""

import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar LightRAG
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
    logger.info("✅ LightRAG importado correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando LightRAG: {e}")
    logger.error("Instala con: pip install lightrag-hku")
    LIGHTRAG_AVAILABLE = False
    exit(1)


class SimpleFinancialRAG:
    """
    Implementación simplificada de LightRAG para datos financieros
    """
    
    def __init__(self, working_dir: str = "rag_financial_knowledge"):
        """
        Inicializa el sistema RAG financiero
        
        Args:
            working_dir: Directorio donde se guardará el grafo de conocimiento
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.rag = None
        self.initialized = False
        
        logger.info(f"📁 Directorio de trabajo: {self.working_dir}")
        
    async def initialize(self):
        """
        Inicializa LightRAG con la configuración correcta según la documentación
        """
        # Verificar API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OPENAI_API_KEY no encontrada en variables de entorno")
        
        logger.info("🔧 Inicializando LightRAG...")
        
        try:
            # Inicializar LightRAG con configuración básica
            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=gpt_4o_mini_complete,
                embedding_func=EmbeddingFunc(
                    embedding_dim=1536,
                    max_token_size=8191,
                    func=openai_embed
                )
            )
            
            # Inicializar storages (CRÍTICO según la documentación)
            logger.info("📦 Inicializando storages...")
            await self.rag.initialize_storages()
            
            # Intentar inicializar pipeline status si está disponible
            try:
                from lightrag.kg.shared_storage import initialize_pipeline_status
                await initialize_pipeline_status()
                logger.info("✅ Pipeline status inicializado")
            except ImportError:
                logger.warning("⚠️ initialize_pipeline_status no disponible (versión antigua de LightRAG)")
            
            self.initialized = True
            logger.info("✅ LightRAG inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando LightRAG: {e}")
            raise
    
    async def insert_financial_data(self, chunks_path: str):
        """
        Inserta los chunks financieros en LightRAG
        
        Args:
            chunks_path: Ruta al archivo JSON con los chunks
        """
        if not self.initialized:
            raise RuntimeError("LightRAG no está inicializado. Llama a initialize() primero")
        
        logger.info(f"📥 Cargando chunks desde: {chunks_path}")
        
        # Cargar chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Obtener lista de chunks según el formato
        if 'documents' in data:
            chunks = data['documents']
        elif 'chunks' in data:
            chunks = data['chunks']
        else:
            chunks = []
        
        logger.info(f"📊 Procesando {len(chunks)} chunks")
        
        # Insertar cada chunk como un documento de texto enriquecido
        inserted = 0
        failed = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Preparar el texto del chunk
                text = self._prepare_chunk_text(chunk)
                
                if text:
                    # Insertar en LightRAG (usa ainsert para async)
                    await self.rag.ainsert(text)
                    inserted += 1
                    logger.info(f"  ✓ Chunk {i+1}/{len(chunks)} insertado")
                else:
                    logger.warning(f"  ⚠️ Chunk {i+1} vacío, saltando")
                    
            except Exception as e:
                failed += 1
                logger.error(f"  ✗ Error en chunk {i+1}: {e}")
        
        logger.info(f"\n✅ Inserción completada:")
        logger.info(f"  - Insertados: {inserted}")
        logger.info(f"  - Fallidos: {failed}")
        
        return inserted, failed
    
    def _prepare_chunk_text(self, chunk: Dict) -> str:
        """
        Prepara el texto del chunk para inserción en LightRAG
        Formatea la información para que el LLM pueda extraer entidades correctamente
        
        Args:
            chunk: Diccionario con los datos del chunk
            
        Returns:
            Texto formateado para LightRAG
        """
        # Obtener el contenido principal
        if 'content' in chunk:
            content = chunk['content']
        elif 'prepared_text' in chunk:
            content = chunk['prepared_text']
        elif 'text' in chunk:
            content = chunk['text']
        else:
            return ""
        
        # Obtener metadata
        metadata = chunk.get('metadata', {})
        chunk_type = metadata.get('chunk_type', chunk.get('chunk_type', 'transaction'))
        
        # Construir texto enriquecido para mejor extracción de entidades
        parts = []
        
        # Añadir tipo de chunk como contexto
        parts.append(f"[TIPO: {chunk_type}]")
        
        # Añadir información temporal si existe
        if 'date_range' in metadata:
            date_range = metadata['date_range']
            if 'start' in date_range:
                parts.append(f"Período: {date_range['start']} - {date_range.get('end', 'actual')}")
        
        # Añadir categoría si existe
        if metadata.get('category'):
            parts.append(f"Categoría: {metadata['category']}")
        
        # Añadir el contenido principal
        parts.append(content)
        
        # Añadir resumen estadístico si existe
        if 'statistical_summary' in metadata:
            stats = metadata['statistical_summary']
            if 'total' in stats:
                parts.append(f"Total: €{stats['total']:.2f}")
            if 'count' in stats:
                parts.append(f"Número de transacciones: {stats['count']}")
            if 'average' in stats:
                parts.append(f"Promedio: €{stats['average']:.2f}")
        
        return "\n".join(parts)
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """
        Realiza una consulta al sistema RAG
        
        Args:
            question: Pregunta en lenguaje natural
            mode: Modo de búsqueda ("naive", "local", "global", "hybrid", "mix")
            
        Returns:
            Respuesta generada por el LLM
        """
        if not self.initialized:
            raise RuntimeError("LightRAG no está inicializado")
        
        logger.info(f"\n🔍 Consulta: {question}")
        logger.info(f"   Modo: {mode}")
        
        try:
            # Configurar parámetros de consulta
            param = QueryParam(mode=mode)
            
            # Realizar consulta
            response = await self.rag.aquery(question, param)
            
            if response:
                logger.info("✅ Respuesta generada")
                return response
            else:
                logger.warning("⚠️ Sin respuesta")
                return "No pude generar una respuesta para esa pregunta."
                
        except Exception as e:
            logger.error(f"❌ Error en consulta: {e}")
            return f"Error procesando la consulta: {str(e)}"
    
    async def batch_query(self, queries: List[str], mode: str = "hybrid") -> List[Dict]:
        """
        Realiza múltiples consultas en batch
        
        Args:
            queries: Lista de preguntas
            mode: Modo de búsqueda
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for query in queries:
            response = await self.query(query, mode)
            results.append({
                "query": query,
                "response": response,
                "mode": mode
            })
        
        return results


async def main():
    """
    Función principal para demostrar el uso de LightRAG
    """
    logger.info("\n" + "="*60)
    logger.info("💰 SISTEMA RAG FINANCIERO - IMPLEMENTACIÓN SIMPLE")
    logger.info("="*60)
    
    # Crear instancia del RAG
    rag = SimpleFinancialRAG(working_dir="simple_rag_knowledge")
    
    # Inicializar
    logger.info("\n1️⃣ INICIALIZACIÓN")
    await rag.initialize()
    
    # Insertar datos
    logger.info("\n2️⃣ INSERCIÓN DE DATOS")
    chunks_path = "data/embeddings/chunks_with_embeddings_lightrag.json"
    
    if Path(chunks_path).exists():
        inserted, failed = await rag.insert_financial_data(chunks_path)
    else:
        logger.error(f"❌ No se encontró: {chunks_path}")
        logger.info("Ejecuta primero: python3 scripts/generate_embeddings.py")
        return
    
    # Realizar consultas de prueba
    logger.info("\n3️⃣ CONSULTAS DE PRUEBA")
    
    test_queries = [
        "¿Cuál es mi situación financiera general?",
        "¿En qué categorías gasto más dinero?",
        "¿Cuánto gasté en Groceries?",
        "¿Tengo gastos recurrentes o suscripciones?",
        "¿Cuál fue mi mayor gasto del mes?",
        "Dame recomendaciones para ahorrar dinero"
    ]
    
    logger.info("\n" + "-"*60)
    logger.info("PROBANDO DIFERENTES MODOS DE CONSULTA")
    logger.info("-"*60)
    
    # Probar modo hybrid (recomendado)
    for i, query in enumerate(test_queries[:3], 1):
        logger.info(f"\n📝 Consulta {i}: {query}")
        response = await rag.query(query, mode="hybrid")
        
        # Mostrar respuesta (truncada si es muy larga)
        if len(response) > 300:
            preview = response[:300] + "..."
        else:
            preview = response
        
        logger.info(f"🤖 Respuesta: {preview}")
    
    # Comparar modos
    logger.info("\n" + "-"*60)
    logger.info("COMPARACIÓN DE MODOS")
    logger.info("-"*60)
    
    comparison_query = "¿Cuáles son mis principales gastos?"
    
    for mode in ["naive", "local", "hybrid"]:
        logger.info(f"\n🔍 Modo: {mode}")
        response = await rag.query(comparison_query, mode=mode)
        
        if "[no-context]" in response:
            logger.info("   ⚠️ Sin contexto suficiente")
        else:
            preview = response[:200] + "..." if len(response) > 200 else response
            logger.info(f"   → {preview}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ DEMO COMPLETADA")
    logger.info("="*60)
    logger.info("\nEl sistema RAG está listo para:")
    logger.info("  • Responder preguntas sobre tus finanzas")
    logger.info("  • Generar análisis con IA")
    logger.info("  • Proporcionar recomendaciones personalizadas")
    logger.info("\n💡 Nota: Con más datos, las respuestas serán más precisas")


if __name__ == "__main__":
    # Ejecutar función asíncrona
    asyncio.run(main())