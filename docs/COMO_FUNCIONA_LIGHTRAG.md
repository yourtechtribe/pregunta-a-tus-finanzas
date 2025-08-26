# 🧠 Cómo Funciona LightRAG en Pregunta a tus Finanzas

## 📚 Índice
1. [Concepto General](#concepto-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Pipeline de Procesamiento](#pipeline-de-procesamiento)
4. [Construcción del Grafo de Conocimiento](#construcción-del-grafo-de-conocimiento)
5. [Sistema de Consultas](#sistema-de-consultas)
6. [Visualización Interactiva](#visualización-interactiva)
7. [Ejemplos Prácticos](#ejemplos-prácticos)

---

## 🎯 Concepto General

Este proyecto implementa **LightRAG real** (Universidad de Hong Kong) con GPT-4o-mini para crear un grafo de conocimiento financiero consultable con lenguaje natural.

### ¿Qué es LightRAG?

LightRAG es un framework de RAG (Retrieval-Augmented Generation) que:
1. **Extrae entidades automáticamente** usando IA
2. **Construye un grafo de conocimiento** con relaciones
3. **Genera respuestas en lenguaje natural** basadas en el grafo

```
Transacciones → Chunking → LightRAG (GPT-4o-mini) → Grafo → Consultas en español
```

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                  SISTEMA RAG FINANCIERO                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  📥 ENTRADA DE DATOS                                    │
│  ┌──────────────────┐                                   │
│  │ Extractos CSV     │                                   │
│  │ (BBVA, etc.)      │                                   │
│  └────────┬──────────┘                                   │
│           ↓                                              │
│  📝 PROCESAMIENTO                                       │
│  ┌──────────────────┐     ┌──────────────────┐        │
│  │ Parser Bancario   │ →   │  Anonimización   │        │
│  │                   │     │  (3 capas)       │        │
│  └──────────────────┘     └────────┬──────────┘        │
│                                     ↓                   │
│  🔄 CHUNKING ADAPTATIVO                                 │
│  ┌──────────────────────────────────────────┐          │
│  │ • Temporal (por fechas)                   │          │
│  │ • Categorías (tipo de gasto)              │          │
│  │ • Montos (rangos de gasto)                │          │
│  │ • Merchant (por comercio)                 │          │
│  │ • Mixto (combinaciones)                   │          │
│  └────────────────┬─────────────────────────┘          │
│                   ↓                                     │
│  🤖 LIGHTRAG (GPT-4o-mini)                             │
│  ┌──────────────────────────────────────────┐          │
│  │ • Extracción de entidades                 │          │
│  │ • Identificación de relaciones            │          │
│  │ • Construcción del grafo                  │          │
│  │ • Indexación para búsqueda                │          │
│  └────────────────┬─────────────────────────┘          │
│                   ↓                                     │
│  📊 GRAFO DE CONOCIMIENTO                              │
│  ┌──────────────────────────────────────────┐          │
│  │ 50+ Entidades | 55+ Relaciones            │          │
│  │ NetworkX + PyVis para visualización       │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  🔍 MODOS DE CONSULTA                                   │
│  ┌──────────────────────────────────────────┐          │
│  │ • Naive: Búsqueda directa en chunks       │          │
│  │ • Local: Exploración del vecindario       │          │
│  │ • Global: Vista completa del grafo        │          │
│  │ • Hybrid: Combinación inteligente         │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline de Procesamiento

### 1️⃣ **Extracción y Anonimización**

```python
# Parseo del extracto bancario
extractor = BBVAExtractor()
transactions = extractor.extract("extracto.csv")

# Anonimización de 3 capas
anonymizer = AdaptiveAnonymizer()
safe_transactions = anonymizer.process(transactions)
# → Remueve nombres, cuentas, referencias personales
```

### 2️⃣ **Chunking Adaptativo**

El sistema crea 5 tipos de chunks para máxima cobertura contextual:

```python
chunking_strategies = {
    'temporal': "Agrupa por semanas/meses",
    'category': "Agrupa por tipo de gasto",
    'amount': "Agrupa por rangos de monto",
    'merchant': "Agrupa por comercio",
    'mixed': "Combinaciones inteligentes"
}

# Ejemplo de chunk generado:
{
    "chunk_id": "temporal_week_27_2025",
    "content": "Semana 27 de 2025: 5 transacciones totalizando €590.39. 
                Categorías: Internal Transfer (€500), Groceries (€77.40), 
                Entertainment (€12.99)...",
    "metadata": {
        "period": "2025-W27",
        "transaction_count": 5,
        "total_amount": -590.39
    }
}
```

### 3️⃣ **Generación de Embeddings**

```python
# OpenAI text-embedding-3-small
embeddings = openai_embed(chunks)
# → Vectores de 1536 dimensiones para cada chunk
```

---

## 🕸️ Construcción del Grafo de Conocimiento

### LightRAG con GPT-4o-mini

```python
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete

# Inicializar LightRAG
rag = LightRAG(
    working_dir="simple_rag_knowledge",
    llm_func=gpt_4o_mini_complete,
    embed_func=openai_embed
)

# Construir grafo (extracción automática de entidades)
await rag.build_knowledge_graph(chunks)
```

### Entidades Extraídas Automáticamente

LightRAG identifica y extrae:

- **📅 Temporales**: "Julio 2025", "Semana 27", "1 de julio"
- **🏪 Comercios**: "Mercadona", "Netflix", "Amazon"
- **📂 Categorías**: "Groceries", "Entertainment", "Housing"
- **💰 Conceptos financieros**: "Gasto", "Transferencia", "Suscripción"
- **🔄 Patrones**: "Gasto recurrente", "Pago mensual"

### Relaciones Identificadas

```
[Netflix] --TIPO_DE--> [Suscripción]
[Netflix] --CATEGORIA--> [Entertainment]
[Julio 2025] --CONTIENE--> [5 transacciones]
[Mercadona] --CATEGORIA--> [Groceries]
[Housing] --REPRESENTA--> [50.7% del gasto]
```

---

## 🔍 Sistema de Consultas

### Modos de Búsqueda

```python
# 1. NAIVE - Búsqueda directa
response = await rag.query(
    "¿Cuánto gasté en comida?",
    mode="naive"
)
# Busca directamente en los chunks

# 2. LOCAL - Exploración de vecindario
response = await rag.query(
    "¿Qué gastos tuve en julio?",
    mode="local"  
)
# Explora entidades relacionadas con "julio"

# 3. GLOBAL - Vista completa
response = await rag.query(
    "¿Cuáles son mis patrones de gasto?",
    mode="global"
)
# Analiza todo el grafo

# 4. HYBRID - Combinación inteligente
response = await rag.query(
    "Analiza mi salud financiera",
    mode="hybrid"
)
# Combina local + global para mejor respuesta
```

### Proceso de Respuesta

1. **Comprensión**: GPT-4o-mini entiende la pregunta en español
2. **Búsqueda**: Identifica entidades relevantes en el grafo
3. **Agregación**: Recopila información de múltiples fuentes
4. **Generación**: Crea una respuesta en lenguaje natural

---

## 📊 Visualización Interactiva

### Grafo con PyVis

```python
from pyvis.network import Network

# Crear visualización interactiva
net = Network(height="750px", width="100%")

# Añadir nodos con colores por tipo
for entity in entities:
    net.add_node(
        entity['name'],
        color=get_color_by_type(entity['type']),
        size=calculate_importance(entity)
    )

# Añadir relaciones
for relation in relations:
    net.add_edge(
        relation['source'],
        relation['target'],
        title=relation['type']
    )

# Generar HTML interactivo
net.show("financial_knowledge_graph.html")
```

### Características de la Visualización

- 🎨 **Colores por tipo**: Categorías, fechas, comercios
- 📏 **Tamaño por importancia**: Nodos más grandes = más relevantes
- 🔗 **Relaciones etiquetadas**: Muestra el tipo de conexión
- 🖱️ **Interactivo**: Zoom, pan, selección de nodos
- 📱 **Responsive**: Funciona en móvil y desktop

---

## 📊 Ejemplos Prácticos

### Ejemplo 1: Consulta de Agregación

```python
# PREGUNTA
"¿Cuánto gasté en supermercados en julio?"

# PROCESO INTERNO
1. Identifica entidades: ["supermercados", "julio"]
2. Mapea a categoría: "Groceries"
3. Busca en grafo: Julio → contiene → Mercadona
4. Agrega montos: €77.40 + €58.09

# RESPUESTA
"En julio gastaste €135.49 en supermercados, con 2 transacciones 
en Mercadona los días 2 y 4 de julio."
```

### Ejemplo 2: Análisis de Patrones

```python
# PREGUNTA
"¿Qué patrones de gasto tengo?"

# PROCESO INTERNO
1. Modo global activado
2. Analiza todas las relaciones
3. Identifica clusters de gasto
4. Detecta recurrencias

# RESPUESTA
"He identificado los siguientes patrones:
- Suscripción mensual a Netflix (€12.99)
- Gastos en supermercados cada 2-3 días
- Housing representa el 50.7% de tus gastos
- Mayor concentración de gastos a inicio de mes"
```

### Ejemplo 3: Recomendaciones

```python
# PREGUNTA
"¿Dónde puedo ahorrar?"

# PROCESO INTERNO
1. Analiza categorías por monto
2. Identifica gastos no esenciales
3. Busca patrones modificables

# RESPUESTA
"Oportunidades de ahorro identificadas:
- Entertainment (€12.99): Revisar suscripciones
- Groceries (€135.49): Representa el 13.5% del gasto
- Considera la transferencia de €500 como ahorro
Total potencial de optimización: ~€25-50/mes"
```

---

## 🎯 Ventajas del Sistema

### 1. **Extracción Automática de Entidades**
- No requiere definir entidades manualmente
- GPT-4o-mini identifica automáticamente conceptos relevantes
- Se adapta a diferentes tipos de datos financieros

### 2. **Respuestas en Lenguaje Natural**
- Comprende preguntas en español coloquial
- Genera respuestas contextualizadas
- Incluye detalles relevantes automáticamente

### 3. **Grafo de Conocimiento Persistente**
- Se construye una vez y se reutiliza
- Actualización incremental con nuevos datos
- Búsquedas rápidas sin reconstruir

### 4. **Múltiples Perspectivas**
- 4 modos de consulta para diferentes necesidades
- Chunking adaptativo para máxima cobertura
- Visualización interactiva para exploración

---

## 💰 Costos y Rendimiento

### Costos de OpenAI

| Operación | Tokens | Costo |
|-----------|--------|-------|
| Construcción inicial | ~10,000 | ~$0.02 |
| Embeddings (100 tx) | ~2,000 | ~$0.001 |
| Consulta típica | ~500 | ~$0.001 |
| **Total mensual** | - | **<$0.05** |

### Rendimiento

| Métrica | Valor |
|---------|-------|
| Construcción grafo | 30-60 segundos |
| Consulta naive | 1-2 segundos |
| Consulta hybrid | 3-5 segundos |
| Precisión respuestas | >90% |

---

## 🚀 Resumen

El sistema combina lo mejor de tres mundos:

1. **LightRAG**: Framework robusto de RAG con grafos
2. **GPT-4o-mini**: IA para extracción y generación
3. **Visualización**: PyVis para exploración interactiva

El resultado es un sistema que entiende tus finanzas y puede responder preguntas complejas en lenguaje natural, manteniendo tus datos privados y con un costo mínimo.

---

## 💡 Tips de Uso

1. **Para mejores resultados**: Usa modo "hybrid" para preguntas complejas
2. **Para rapidez**: Usa modo "naive" para búsquedas simples
3. **Para exploración**: Usa la visualización interactiva
4. **Para análisis**: Combina múltiples consultas

El sistema está diseñado para ser tu asistente financiero personal, convirtiendo datos bancarios crudos en insights accionables.