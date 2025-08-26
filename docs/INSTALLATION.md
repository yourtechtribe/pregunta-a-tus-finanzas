# 📦 Guía de Instalación - Pregunta a tus Finanzas

## 📋 Requisitos Previos

### Sistema
- **Python**: 3.10 o superior
- **RAM**: 2GB mínimo (4GB recomendado)
- **Disco**: 500MB libres
- **SO**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)

### Cuentas y APIs
- **OpenAI API Key**: Requerida para LightRAG y embeddings
  - Obtener en: https://platform.openai.com/api-keys
  - Costo estimado: <$0.05/mes para uso típico

## 🚀 Instalación Rápida

### 1. Clonar el Repositorio

```bash
git clone https://github.com/yourtechtribe/pregunta-a-tus-finanzas.git
cd pregunta-a-tus-finanzas
```

### 2. Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tu editor favorito
nano .env  # o vim, code, notepad, etc.
```

Contenido del `.env`:
```env
# OpenAI API Key (requerida)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx

# Opcional: Configuración avanzada
LIGHTRAG_WORKING_DIR=simple_rag_knowledge
LIGHTRAG_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 5. Verificar Instalación

```bash
python scripts/test_installation.py
```

Si todo está correcto, verás:
```
✅ Python version: 3.10.12
✅ LightRAG installed
✅ OpenAI API key configured
✅ All dependencies installed
✅ System ready!
```

## 📚 Instalación Detallada

### Dependencias Principales

```bash
# Core
pip install lightrag-hku>=0.1.0  # Framework RAG
pip install openai>=1.0.0        # API de OpenAI
pip install python-dotenv         # Variables de entorno

# Procesamiento de datos
pip install pandas>=2.0.0        # Manejo de datos
pip install numpy>=1.24.0        # Cálculos numéricos

# Visualización
pip install pyvis>=0.3.0         # Grafos interactivos
pip install networkx>=3.0        # Análisis de grafos

# Anonimización
pip install presidio-analyzer    # Detección de PII
pip install presidio-anonymizer  # Anonimización

# Extras
pip install colorama             # Output con colores
pip install tqdm                 # Barras de progreso
```

### Instalación Manual de LightRAG

Si `pip install lightrag-hku` falla:

```bash
# Opción 1: Desde GitHub
pip install git+https://github.com/HKUDS/LightRAG.git

# Opción 2: Manual
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
cd ..
```

## 🏃 Primer Uso

### 1. Preparar tus Datos

Coloca tu extracto bancario en `data/raw/`:
```bash
cp tu_extracto.csv data/raw/extracto_bbva.csv
```

Formato esperado (BBVA):
```csv
Fecha,Concepto,Movimiento,Saldo
01/07/2025,TRANSFERENCIA,-500.00,1500.00
02/07/2025,MERCADONA,-77.40,1422.60
```

### 2. Procesar y Construir el Grafo

```bash
# Procesar extracto bancario
python scripts/run_pipeline.py --bank bbva --file data/raw/extracto_bbva.csv

# Construir grafo de conocimiento con LightRAG
python scripts/build_lightrag_graph.py
```

### 3. Hacer tu Primera Consulta

```bash
# Interfaz interactiva de consultas
python scripts/demo_queries.py
```

Ejemplos de preguntas:
- "¿Cuánto gasté en julio?"
- "¿Cuáles son mis gastos en supermercados?"
- "¿Qué patrones de gasto tengo?"

### 4. Visualizar el Grafo

```bash
# Iniciar servidor de visualización
python scripts/visualize_graph.py

# Abrir en navegador
# http://localhost:8080/financial_knowledge_graph.html
```

## 🔧 Configuración Avanzada

### Modelos de OpenAI

En `.env` puedes configurar:

```env
# Modelo para LightRAG (extracción de entidades)
LIGHTRAG_MODEL=gpt-4o-mini  # Más económico
# LIGHTRAG_MODEL=gpt-4o     # Más preciso

# Modelo de embeddings
EMBEDDING_MODEL=text-embedding-3-small  # Económico (1536 dim)
# EMBEDDING_MODEL=text-embedding-3-large # Preciso (3072 dim)
```

### Directorio de Trabajo

```env
# Dónde se almacena el grafo
LIGHTRAG_WORKING_DIR=simple_rag_knowledge

# Para múltiples usuarios/proyectos
# LIGHTRAG_WORKING_DIR=graphs/usuario1
# LIGHTRAG_WORKING_DIR=graphs/proyecto_2025
```

### Configuración de Chunking

Editar `scripts/chunking_strategy.py`:

```python
CHUNK_CONFIG = {
    'temporal_window': 7,      # Días por chunk temporal
    'category_min_txs': 2,     # Mín transacciones por categoría
    'amount_ranges': [0, 50, 100, 500, 1000],  # Rangos de monto
    'max_chunk_size': 2000     # Tokens máximos por chunk
}
```

## 🐛 Solución de Problemas

### Error: "No module named 'lightrag'"

```bash
# Reinstalar LightRAG
pip uninstall lightrag-hku
pip install --no-cache-dir lightrag-hku
```

### Error: "Invalid API key"

```bash
# Verificar tu API key
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows

# Probar la API
python -c "import openai; openai.api_key='tu-key'; print(openai.models.list())"
```

### Error: "Insufficient quota"

- Verifica tu saldo en: https://platform.openai.com/usage
- Añade créditos si es necesario
- Considera usar `gpt-4o-mini` (más económico)

### El grafo no se visualiza

```bash
# Verificar que PyVis está instalado
pip install --upgrade pyvis

# Verificar el puerto
lsof -i :8080  # Linux/Mac
netstat -an | findstr :8080  # Windows

# Usar otro puerto
python scripts/visualize_graph.py --port 8081
```

## 🚀 Optimización de Rendimiento

### Para Datasets Grandes (>1000 transacciones)

1. **Chunking por lotes**:
```python
# En build_lightrag_graph.py
BATCH_SIZE = 100  # Procesar de 100 en 100
```

2. **Caché de embeddings**:
```python
# Reutilizar embeddings existentes
USE_CACHE = True
CACHE_DIR = "data/embeddings/cache"
```

3. **Modo incremental**:
```python
# Añadir nuevas transacciones sin reconstruir
rag.update_graph(new_transactions)
```

### Reducir Costos de API

1. **Usar modelos económicos**:
```env
LIGHTRAG_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

2. **Limitar chunks**:
```python
MAX_CHUNKS_PER_QUERY = 5
```

3. **Caché de consultas**:
```python
ENABLE_QUERY_CACHE = True
```

## 📁 Estructura de Archivos Generada

Después de la instalación y primer uso:

```
pregunta-tus-finanzas/
├── simple_rag_knowledge/        # Grafo de LightRAG
│   ├── kv_store_*.json          # Almacén clave-valor
│   ├── vdb_*.json               # Base vectorial
│   └── graph_analysis.json      # Análisis del grafo
├── data/
│   ├── raw/                     # Extractos originales
│   ├── processed/               # Datos procesados
│   └── embeddings/              # Embeddings generados
├── outputs/
│   └── financial_knowledge_graph.html  # Visualización
└── .env                         # Configuración
```

## ✅ Verificación Final

Ejecuta el script de verificación completo:

```bash
python scripts/verify_setup.py
```

Output esperado:
```
🔍 Verificando instalación...
✅ Python 3.10+ detectado
✅ Todas las dependencias instaladas
✅ OpenAI API key configurada
✅ Directorio de trabajo creado
✅ Datos de prueba disponibles
✅ LightRAG inicializado correctamente
✅ Visualización funcionando

🎉 ¡Sistema listo para usar!

Próximos pasos:
1. python scripts/build_lightrag_graph.py  # Construir grafo
2. python scripts/demo_queries.py          # Hacer consultas
3. python scripts/visualize_graph.py       # Visualizar
```

## 📞 Soporte

Si encuentras problemas:

1. Revisa los [Issues](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/issues) existentes
2. Busca en [Discussions](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/discussions)
3. Abre un nuevo issue con:
   - Versión de Python
   - Sistema operativo
   - Error completo
   - Pasos para reproducir

## 🎯 Siguiente Paso

¡Felicidades! Ya tienes el sistema instalado. Ahora puedes:

1. Procesar tu extracto bancario real
2. Construir tu grafo de conocimiento personal
3. Hacer consultas sobre tus finanzas
4. Explorar el grafo interactivo

Consulta el [README.md](../README.md) para ejemplos de uso.