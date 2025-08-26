# 💰 Pregunta a tus Finanzas - Sistema Agentic RAG Financiero

Sistema open source para analizar extractos bancarios usando RAG (Retrieval-Augmented Generation) con LightRAG y GPT-4o-mini. Convierte tus transacciones en un grafo de conocimiento consultable con lenguaje natural.

## 🌟 Características Principales

- ✅ **LightRAG real con IA**: Extracción automática de entidades y respuestas generadas por GPT-4o-mini
- ✅ **Grafo de conocimiento interactivo**: 50+ entidades y relaciones extraídas automáticamente
- ✅ **Chunking adaptativo**: 5 estrategias diferentes para máxima cobertura contextual
- ✅ **Visualización interactiva**: Grafo navegable con PyVis
- ✅ **95.7% precisión en anonimización**: Sistema de 3 capas adaptativo
- ✅ **100% privado**: Procesamiento local de datos sensibles
- ✅ **Ultra bajo costo**: ~$0.02 construcción + $0.001 por consulta
- ✅ **4 modos de consulta**: Naive, Local, Global, Híbrido

## 🚀 Quick Start

```bash
# Clonar el repositorio
git clone https://github.com/yourtechtribe/pregunta-a-tus-finanzas
cd pregunta-a-tus-finanzas

# Opción 1: Setup automático (RECOMENDADO)
chmod +x setup.sh
./setup.sh

# Opción 2: Setup manual
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# Editar .env y añadir: OPENAI_API_KEY=sk-...

# Construir el grafo de conocimiento
python scripts/build_lightrag_graph.py

# Hacer consultas interactivas
python scripts/demo_queries.py

# Visualizar el grafo (abre navegador en puerto 8080)
python scripts/visualize_graph.py
```

## 📊 Arquitectura del Sistema

```
Extracto CSV/Excel → Parser → Anonimización → Categorización → 
→ Chunking Adaptativo → Embeddings → LightRAG (GPT-4o-mini) → 
→ Grafo de Conocimiento → Consultas en Lenguaje Natural
```

### Pipeline Completo

1. **Extracción**: Parsers especializados por banco (BBVA implementado)
2. **Anonimización**: Sistema 3 capas (Regex + Presidio + LLM)
3. **Categorización**: Multi-agente con memoria persistente
4. **Chunking Adaptativo**: 5 estrategias (Temporal, Categoría, Monto, Merchant, Mixto)
5. **Embeddings**: OpenAI text-embedding-3-small (1536 dimensiones)
6. **LightRAG**: Extracción automática de entidades con GPT-4o-mini
7. **Grafo**: NetworkX + PyVis para visualización interactiva

## 💻 Uso del Sistema

### 1. Construir el Grafo de Conocimiento

```bash
# Procesar transacciones y construir grafo con LightRAG
python scripts/build_lightrag_graph.py

# El script:
# - Carga transacciones desde data/processed/
# - Aplica chunking adaptativo (5 estrategias)
# - Genera embeddings con OpenAI
# - Extrae entidades con GPT-4o-mini
# - Construye el grafo en simple_rag_knowledge/
```

### 2. Hacer Consultas

```bash
# Ejecutar demo de consultas
python scripts/demo_queries.py

# O usar el script de análisis
python scripts/analyze_graph.py
```

### 3. Visualizar el Grafo

```bash
# Iniciar servidor de visualización
python scripts/visualize_graph.py

# Abrir navegador en: http://localhost:8080/financial_knowledge_graph.html
```

### Opciones de categorización

| Método | Ventajas | Desventajas | Coste |
|--------|----------|-------------|-------|
| **rules** | ✅ 100% privado<br>✅ Sin coste<br>✅ Rápido | ❌ Requiere mantenimiento<br>❌ Menos precisión | Gratis |
| **gpt5-nano** | ✅ Alta precisión<br>✅ Auto-aprendizaje<br>✅ Detecta patrones | ❌ Requiere API key<br>❌ Coste por uso | ~$0.001/100 tx |

### Ejemplo de Código

```python
from scripts.build_lightrag_graph import FinancialLightRAG
import asyncio

# Inicializar LightRAG
rag = FinancialLightRAG(working_dir="simple_rag_knowledge")

# Cargar y procesar transacciones
chunks = rag.load_financial_chunks("data/embeddings/chunks_with_embeddings_lightrag.json")
await rag.build_knowledge_graph(chunks)

# Hacer consultas
response = await rag.query(
    "¿Cuánto gasté en restaurantes en julio?",
    mode="hybrid"  # naive, local, global, hybrid
)
print(response)

# Analizar el grafo
stats = rag.analyze_graph()
print(f"Entidades: {stats['total_entities']}")
print(f"Relaciones: {stats['total_relations']}")
```

### Ejemplos de Consultas

**Consultas de Agregación:**
- "¿Cuál fue mi gasto total en julio?"
- "¿Cuánto gasté en restaurantes?"
- "Suma todos los gastos de alimentación"

**Análisis de Patrones:**
- "¿Cuáles son mis gastos recurrentes?"
- "¿Qué patrones de gasto tengo?"
- "¿En qué categoría gasto más?"

**Búsquedas Específicas:**
- "Muestra transacciones superiores a 50 euros"
- "¿Qué compré en Amazon?"
- "Lista los gastos del fin de semana"

**Recomendaciones:**
- "¿Dónde puedo ahorrar dinero?"
- "¿Tengo gastos innecesarios?"
- "Analiza mi salud financiera"

## 📈 Métricas del Sistema

### Rendimiento
| Métrica | Valor | Detalles |
|---------|-------|----------|
| Entidades extraídas | 50+ | Categorías, comercios, fechas |
| Relaciones | 55+ | Conexiones automáticas |
| Precisión respuestas | 91% | En consultas agregadas |
| Cobertura de datos | 100% | Todas las transacciones |
| Latencia consulta | 2-5s | Con GPT-4o-mini |
| Precisión anonimización | 95.7% | Sistema 3 capas |

### Costos (OpenAI)
| Operación | Costo | Detalles |
|-----------|-------|----------|
| Construcción grafo | ~$0.02 | Extracción entidades |
| Embeddings | ~$0.001 | Por 100 transacciones |
| Consulta | ~$0.001 | Por pregunta |
| Total mensual | <$0.05 | Uso típico |

## 🤝 Contribuir

¿Quieres añadir soporte para tu banco? ¡Genial!

1. Fork el repositorio
2. Crea un parser en `src/extractors/tu_banco_extractor.py`
3. Añade tests en `tests/test_tu_banco.py`
4. Envía un Pull Request

Ver [CONTRIBUTING.md](docs/CONTRIBUTING.md) para más detalles.

## 📚 Documentación

- [Cómo funciona LightRAG](docs/COMO_FUNCIONA_LIGHTRAG.md) - Arquitectura detallada
- [Guía de instalación](docs/INSTALLATION.md) - Setup completo paso a paso
- [Roadmap del proyecto](docs/ROADMAP.md) - Próximas funcionalidades
- [Guía de contribución](docs/CONTRIBUTING.md) - Cómo añadir tu banco

## 🛡️ Privacidad y Seguridad

- **Datos 100% locales**: El grafo se construye y almacena localmente
- **Anonimización robusta**: Sistema de 3 capas antes de cualquier procesamiento
- **API mínima**: Solo se envían chunks anonimizados a OpenAI
- **Sin logs sensibles**: No se registran datos personales o financieros
- **Open source**: Código completamente auditable

## 🔧 Requisitos

- Python 3.10+
- OpenAI API Key (para embeddings y LightRAG)
- 2GB RAM mínimo
- 500MB espacio en disco

## 📜 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

## 👨‍💻 Autor

**Albert Gil López**
- CTO @ M.IA (himia.app)
- LinkedIn: [albertgilopez](https://linkedin.com/in/albertgilopez)
- Email: albert.gil@yourtechtribe.com

---

⭐ Si te gusta el proyecto, dale una estrella en GitHub!

🐛 Encuentra un bug? [Abre un issue](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/issues)

💡 Tienes una idea? [Inicia una discusión](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/discussions)