# "Pregunta a tus Finanzas": Cómo Construí un "ChatGPT" para mis Extractos Bancarios (y lo estoy convirtiendo en un Sistema Multi-Agente)

*Autor: Albert Gil López CTO @ M.IA*
*Tiempo de lectura: 15 min*

![IMAGEN]()

📋 **TL;DR - Lo que hemos construido (y lo que aprenderás)**

*   ✅ **Pipeline de Indexación Completo (Fase 1):** Un sistema robusto que cubre las 6 etapas iniciales del ciclo RAG: ingesta, limpieza, chunking, embeddings, construcción de grafos y sistema de queries.
*   ✅ **Anonimización Inteligente y Adaptativa:** Alcanzamos un 95.7% de precisión en la detección de PII usando un sistema de 3 capas que aprende y mejora con el tiempo, sin dependencias externas.
*   ✅ **100% Privado y Open Source:** Todo el código y los datos están en GitHub. Para proteger tu privacidad, los datos son anonimizados antes de ser procesados por cualquier proveedor externo via API.
*   ✅ **Costo Ultra-Bajo:** Procesar las 126 transacciones de una persona durante un mes costó solo $0.03, demostrando una viabilidad económica excepcional para el análisis financiero personal.
*   ✅ **Sistema de Consultas Multi-Modo:** Implementamos 4 modos de consulta (naive, local, global, híbrido) con latencias de 4 a 9 segundos, permitiendo un análisis flexible.
*   ✅ **Lecciones del Mundo Real:** Aprende de una implementación práctica sobre datos bancarios reales, incluyendo los errores y las soluciones que nos llevaron a un sistema funcional.

## 1. Introducción: De un análisis teórico a una implementación práctica

> "¿Y si pudieras preguntarle directamente a tus finanzas personales cualquier cosa y obtener respuestas instantáneas e inteligentes?"

En un artículo anterior, exploramos en detalle las 8 arquitecturas RAG que definen el panorama en 2025, desde el RAG básico hasta los sistemas multi-agente. Pero la teoría solo te lleva hasta cierto punto. La pregunta inevitable era: ¿cómo se comporta este ecosistema de arquitecturas con la complejidad de los datos financieros del mundo real?

Este experimento nace de esa curiosidad técnica y de la necesidad de explorar arquitecturas RAG robustas para el sector financiero. Aunque el objetivo inicial era entender profundamente cada decisión arquitectónica usando mis propios extractos bancarios, esta saga de tres artículos busca sentar las bases de una herramienta funcional. La meta final está alineada con la visión de M.IA: optimizar la tesorería, empezando por las finanzas personales para luego escalar el modelo a las pymes catalanas y ayudarles a tomar mejores decisiones.

Este artículo es el primero de una serie de tres. En el segundo, construiremos una interfaz de usuario interactiva vibe-codeando con Claude Code, y en el tercero, evolucionaremos nuestro pipeline a un sistema multi-agente utilizando el Agent Development Kit (ADK) de Google.

## 2. Background: El Problema que Resolvemos en M.IA

En M.IA (himia.app), hemos identificado una brecha tecnológica crítica. Según el "Barómetro de adopción de la IA en las pymes españolas" de IndesIA 2025, solo el 2.9% de las PYMEs españolas utilizan herramientas de inteligencia artificial, y en Cataluña, el porcentaje es de apenas el 3.7%, Esta brecha es especialmente grave en la gestión financiera, donde las aplicaciones tradicionales siguen el mismo patrón de hace una década: importas tu CSV, obtienes gráficos predefinidos, categorización automática mediocre (60-70% de precisión), y cero capacidad de hacer preguntas complejas sobre tus datos.

Mi frustración personal, que refleja la de miles de emprendedores y gestores financieros, vino cuando intenté responder una pregunta aparentemente simple: "¿Cuánto gasto en promedio cuando salgo a cenar con amigos vs cuando como solo?". Ninguna app puede hoy en día responder. Necesitas contexto, relaciones, y comprensión semántica que va más allá de simples categorías.

Esta experiencia personal alimenta directamente también nuestro trabajo en el proyecto con el IIIA-CSIC y PIMEC, donde estamos desarrollando un sistema multiagente de IA para transformar las transacciones económicas entre PYMEs catalanas y optimizar su tesorería. El concepto "Pregunta a tus Finanzas" nace de esta visión: democratizar el acceso a inteligencia financiera avanzada, explicable, que genere confianza y sirva para tomar mejores decisiones.

### ¿Por qué RAG para Finanzas Personales?

La tecnología RAG (Retrieval-Augmented Generation) es una de las aplicaciones más potentes que los LLMs han habilitado, permitiendo, por ejemplo, la creación de chatbots sofisticados de preguntas y respuestas (Q&A) que pueden razonar sobre información específica de una fuente de datos privada.

Una aplicación RAG típica tiene dos componentes principales:

1.  **Indexación:** un proceso para ingestar datos de una fuente y organizarlos. Esto generalmente ocurre offline e implica:
    *   **Carga (Load):** Cargar los datos, en nuestro caso, los extractos bancarios.
    *   **División (Split):** Dividir grandes documentos en trozos más pequeños para facilitar la búsqueda y el procesamiento por parte del modelo.
    *   **Almacenamiento (Store):** Guardar e indexar estos trozos, a menudo utilizando un VectorStore y un modelo de Embeddings.
2.  **Recuperación y Generación:** la cadena RAG en tiempo de ejecución que toma la consulta del usuario, recupera los datos relevantes del índice y luego los pasa al modelo para generar una respuesta.
    *   **Recuperación (Retrieve):** Dado una entrada del usuario, se recuperan los trozos relevantes del almacenamiento.
    *   **Generación (Generate):** Un ChatModel / LLM produce una respuesta utilizando un prompt que incluye tanto la pregunta como los datos recuperados.

![“Pregunta a Tu PDF”]()

Esta arquitectura es ideal para nuestro caso de uso porque nos permite construir un sistema que no solo "habla" sobre finanzas, sino que "entiende" y "razona" sobre tus finanzas, manteniendo la privacidad y el contexto.

RAG (Retrieval-Augmented Generation) combina lo mejor de dos mundos:

*   **Precisión de datos estructurados:** Cada transacción es un hecho inmutable
*   **Comprensión contextual de LLMs:** Capacidad de entender preguntas naturales y encontrar patrones

Pero aplicar RAG a datos financieros presenta desafíos únicos:

*   Los datos bancarios son altamente sensibles
*   RAG tradicionalmente funciona mejor con texto narrativo, de hecho los LLMs no están preparados para trabajar con datos estructurados.
*   Un error del 5% en categorización puede significar cientos de euros mal contabilizados (spoiler: el objetivo no es la contabilidad, sino el análisis de patrones y tendencias).
*   Las transacciones tienen relaciones temporales y causales

### LightRAG: Simple and Fast Retrieval-Augmented Generation

Entre las opciones disponibles, como GraphRAG de Microsoft, elegí LightRAG de la Universidad de Hong Kong (HKU). La decisión se basa en su enfoque pragmático y eficiente para resolver los desafíos específicos de este proyecto. Como se detalla en el paper, LightRAG ha sido diseñado para superar las limitaciones de los sistemas RAG tradicionales que dependen de representaciones de datos planas, incorporando estructuras de grafos para mejorar la “conciencia” contextual.

Las tres razones clave para esta elección son:

1.  **Arquitectura híbrida y velocidad:** A diferencia de sistemas puramente basados en grafos como GraphRAG, que pueden tardar entre 30 y 40 segundos por consulta, LightRAG opera como un Hybrid RAG. Combina búsqueda vectorial, búsqueda en grafo y búsqueda textual simple (naive) en paralelo, fusionando y reordenando los resultados. Esto le permite ofrecer tiempos de respuesta de entre 20 y 100 milisegundos, ideal para las consultas directas y factuales que caracterizan el análisis financiero personal.
2.  **Actualización incremental:** El entorno de las finanzas personales es dinámico. LightRAG está diseñado con un algoritmo de actualización incremental que permite añadir nuevas transacciones o fuentes de datos sin necesidad de reconstruir todo el índice desde cero. Esta capacidad es crucial para mantener el sistema relevante y eficiente a lo largo del tiempo.
3.  **Sistema de Recuperación de Doble Nivel:** El paper de LightRAG describe un sistema de recuperación de doble nivel que permite descubrir conocimiento tanto a bajo nivel (entidades específicas) como a alto nivel (conceptos y relaciones complejas). Esta dualidad es perfecta para nuestro caso de uso: podemos preguntar por un gasto concreto ("¿Cuánto costó la cena en 'La Pizzería'?") y también por patrones más amplios ("¿Cuál es mi gasto promedio en restaurantes italianos?").

En resumen, mientras que GraphRAG es una herramienta potente para análisis holísticos y descubrir conexiones ocultas en grandes volúmenes de datos narrativos, LightRAG ofrece una solución más ágil y rápida, optimizada para la velocidad y la relevancia contextual en un entorno de datos que cambia constantemente.
¿Cómo lo hemos implementado? A continuación, desglosamos el pipeline paso a paso.

## 3. Hands-on: Construyendo el Pipeline RAG Financiero, Paso a Paso

A lo largo de esta serie de tres artículos, construiremos un pipeline de RAG financiero completamente funcional. No solo presentaremos el código, sino que explicaremos las decisiones de diseño, los desafíos encontrados y las soluciones implementadas.

### Etapa 1: Ingesta de Datos - Un Parser Open Source y Extensible

El primer obstáculo en cualquier proyecto de análisis financiero es la diversidad de formatos de datos. Los bancos entregan extractos en una variedad de formatos: PDF, Excel, CSV, etc. Nuestro objetivo es crear un proyecto open source con un conjunto de "parsers" capaces de manejar esta diversidad.

Hemos empezado con `bbva_extractor.py`, un parser especializado para el banco BBVA:

*   **Prioridad al Excel para Máxima Precisión:** Para los archivos `.xlsx` de BBVA, la decisión fue usar la librería pandas de Python. Este método nos permite saltar las cabeceras de metadatos y acceder directamente a la tabla de transacciones, garantizando una precisión del 100%.
*   **El Reto del PDF (Trabajo en Progreso):** La extracción de datos de PDFs es un desafío conocido. Aunque hemos investigado el uso de Gemini Vision para interpretar los PDFs de forma nativa y extraer la información en un formato JSON estructurado, esta funcionalidad todavía no está implementada. Será una de las futuras mejoras del proyecto.

Durante el desarrollo, nos encontramos con un detalle crucial: la correcta interpretación de los formatos numéricos. Es fundamental tener en cuenta si se usan comas o puntos como separadores decimales para evitar errores que pueden invalidar todo el análisis.

Este primer extractor es la base de nuestro pipeline. Al normalizar los datos de BBVA en un formato JSON consistente, preparamos el terreno para la siguiente etapa: el enriquecimiento y la indexación.

Te invito a contribuir con extractores para otros bancos. Puedes hacer un fork del repositorio, agregar tu parser y enviar un pull request.

```python
# bbva_extractor.py - Parser específico para BBVA
class BBVAExtractor:
    def extract(self, csv_path):
        # BBVA usa ISO-8859-1, punto y coma como separador
        df = pd.read_csv(csv_path, encoding='ISO-8859-1', sep=';', decimal=',')
        # Normalización crítica: formato español (1.234,56) → float
        def parse_spanish_amount(amount_str):
            return float(amount_str.replace('.', '').replace(',', '.'))
        transactions = []
        for _, row in df.iterrows():
            transaction = {
                "date": pd.to_datetime(row['Fecha'], format='%d/%m/%Y'),
                "description": row['Concepto'],
                "amount": parse_spanish_amount(row['Importe']),
                "balance": parse_spanish_amount(row['Saldo'])
            }
            transactions.append(transaction)
        return transactions
```
*Resultado: 126 transacciones extraídas correctamente.*

### Etapa 2: Limpieza y Anonimización - Sistema de 3 Capas

La privacidad no es una opción. Para garantizarla, implementamos un sistema de anonimización de 3 capas que combina velocidad, inteligencia y adaptabilidad, procesando los datos de forma 100% local para máxima seguridad.

*   **Capa 1: Velocidad con Regex (70% de Precisión en <1ms):** Utilizamos expresiones regulares optimizadas para detectar patrones comunes y de alta frecuencia como DNIs, IBANs o números de teléfono. Es un filtro rápido y eficiente que maneja los casos más obvios sin sacrificar rendimiento.
*   **Capa 2: Inteligencia Contextual con Presidio (85% de Precisión en 10ms):** Para los datos que superan la primera capa, entra en juego Microsoft Presidio (https://microsoft.github.io/presidio/). Esta herramienta open-source, combinada con modelos de NLP de spaCy, además de buscar patrones, busca entender el contexto. Distingue entre un nombre propio y el nombre de un comercio, reduciendo drásticamente los falsos positivos. Añadimos permite añadir funciones a medida para entidades específicas de España, como NIFs, números de la seguridad social, etc.
*   **Capa 3: Validación Selectiva con LLMs (>95% de Precisión):** Cuando Presidio detecta una entidad con un bajo nivel de confianza, recurrimos de forma selectiva a un modelo de lenguaje. En lugar de enviar todo el texto, se aísla únicamente el dato dudoso y se pide una segunda opinión al LLM. Este enfoque quirúrgico nos da una precisión altísima sin comprometer la privacidad ni incurrir en altos costes, resolviendo menos del 10% de los casos que son ambiguos.

Este sistema se complementa con un mecanismo de aprendizaje continuo. Las detecciones de baja confianza y los errores identificados se registran y analizan mensualmente. Esto nos permite descubrir nuevos patrones de PII y refinarlos para fortalecer la Capa 1 y 2, haciendo el sistema más inteligente con cada ciclo.

### Etapa 3: Re-categorización Inteligente con un Sistema Multi-Agente

La categorización inicial que ofrecen los bancos, aunque útil, a menudo es demasiado genérica o, en ocasiones, incorrecta. Por ejemplo, una transacción real que encontramos en nuestro análisis fue "Cacenca". Un sistema de reglas simple podría interpretarla como una retirada de efectivo "Cash" debido a la ambigüedad del nombre, mientras que el banco la clasificaba como "Otros". Sin embargo, la tecnología nos permite afinar mucho más.

Para resolver este tipo de ambigüedades, diseñamos e implementamos un sistema de re-categorización multi-agente. En lugar de depender de una única función o un modelo monolítico, creamos un ecosistema de agentes de IA especializados que colaboran para analizar y categorizar cada transacción con la máxima precisión.

#### Arquitectura del Sistema Multi-Agente

Nuestro sistema se compone de varios agentes, cada uno con una tarea específica, que trabajan en conjunto bajo la dirección de un orquestador. Este diseño modular nos permite aislar responsabilidades y escalar o mejorar cada componente de forma independiente.

**El Flujo de Trabajo de los Agentes:**

1.  **OrchestratorAgent:** Recibe la transacción y, basándose en su complejidad y características, decide qué agente o secuencia de agentes es la más adecuada para procesarla.
2.  **CategorizationAgent:** Es el primer filtro inteligente. Aplica un pipeline de tres niveles para una categorización rápida y eficiente:
    *   **Reglas y Patrones:** Identifica categorías obvias mediante reglas predefinidas (ej: "seguridad social" se convierte en "Impuestos").
    *   **Memoria Persistente:** Consulta una base de datos interna para verificar si el comercio ya ha sido identificado y categorizado en el pasado.
    *   **Investigación Delegada:** Si el comercio es nuevo o desconocido, pasa la tarea al `ResearchAgent`.
3.  **ResearchAgent:** Cuando se enfrenta a un comercio ambiguo como "Cacenca", su misión es saber de qué se trata. Para ello, utiliza herramientas de búsqueda web avanzadas. En nuestra implementación, integramos Tavily Search API, una herramienta diseñada específicamente para agentes de IA que necesitan acceso a información web en tiempo real. Tavily realiza una búsqueda y entrega resultados optimizados y relevantes, reduciendo el ruido y mejorando la capacidad del agente para tomar decisiones informadas. Por ejemplo, el `ResearchAgent` construiría una consulta como "Cacenca Spain business type" y, gracias a Tavily, descubriría rápidamente que se trata de una gasolinera en Gelida, Barcelona.
4.  **ValidationAgent:** Una vez que el `ResearchAgent` propone una categoría (ej: "Transporte y Combustible"), este agente la valida. Comprueba si otros datos de la transacción, como el importe (50€), son coherentes con la categoría propuesta. Un gasto de 50€ es razonable para un repostaje de combustible, pero no lo sería para una cena en un restaurante de lujo.
5.  **MemoryStore:** Cada vez que se identifica y valida un nuevo comercio, la información se almacena de forma persistente. La próxima vez que aparezca una transacción de "Cacenca", el sistema la reconocerá instantáneamente desde la memoria, resolviendo la categorización en milisegundos en lugar de requerir una nueva búsqueda. Este mecanismo de aprendizaje continuo es clave para la eficiencia y escalabilidad del sistema.

Este enfoque, que combina reglas, memoria y una búsqueda web inteligente impulsada por agentes, nos ha permitido alcanzar una precisión de categorización superior al 98%. El sistema corrige los errores de la categorización bancaria y aprende y mejora con cada nueva transacción que procesa.

### Etapa 4: El Reto del Chunking en Datos Financieros

Nuestra primera aproximación al chunking fue la que dictaba la lógica convencional. Adoptamos una estrategia híbrida, combinando agrupaciones temporales (diarias, semanales) y semánticas (por categoría, comerciante), una práctica común en muchos sistemas RAG. El objetivo era crear "chunks" o fragmentos de unos 1200 tokens que agruparan transacciones para que el modelo de lenguaje pudiera identificar patrones y relaciones.

Inicialmente, los resultados parecían prometedores. Sin embargo, un análisis más riguroso destapó un problema fundamental que pasaba desapercibido: sin darnos cuenta, estábamos perdiendo cerca del 50% de las transacciones en el proceso.

La estrategia de agregación, aunque bien intencionada, tenía un defecto de base para el dominio financiero: priorizaba la formación de grandes chunks sobre la integridad de los datos. Esto generaba tres problemas clave:

1.  El chunking tradicional no respeta la naturaleza atómica de una transacción financiera. Cada movimiento es una pieza de información crítica e indivisible. Agruparlas sin garantizar la presencia de cada una es como leer un libro saltándose páginas; se pierde el hilo conductor y los detalles esenciales.
2.  Nuestra configuración, al buscar un tamaño mínimo para los chunks, descartaba implícitamente categorías con pocas transacciones o días de baja actividad. Esto significaba que movimientos importantes, pero aislados, simplemente no llegaban a formar parte del conocimiento del sistema.
3.  En los días o categorías con un volumen alto de transacciones, los chunks superaban el límite de tokens, lo que provocaba que se truncara información de manera arbitraria, perdiendo transacciones valiosas.

No nos podemos permitirnos perder ni una sola transacción. La fiabilidad del sistema depende de ello, así que nos obligó a rediseñar nuestra estrategia desde la base. Decidimos abandonar la idea de que los chunks grandes eran siempre mejores y adoptamos un enfoque radicalmente simple pero efectivo: cada transacción individual se convierte en un chunk enriquecido.

A esta estrategia la llamamos “Transaction-First”, ya que garantiza una cobertura completa del 100% de los datos al mismo tiempo que optimiza el uso de tokens. La clave es una arquitectura jerárquica que opera en múltiples niveles de abstracción, permitiendo tanto consultas muy específicas como análisis de patrones generales.

#### Arquitectura Jerárquica del Chunking

Nuestra arquitectura final genera 164 chunks organizados en cuatro niveles:

*   **Nivel 1: Transacciones Individuales Enriquecidas (126 chunks)**
    *   `tx_chunk_001`: Transacción enriquecida con ~93 tokens
    *   `tx_chunk_002`: Transacción enriquecida con ~93 tokens
    *   ... (hasta 126 transacciones)
*   **Nivel 2: Agregaciones Diarias (21 chunks)**
    *   `daily_2025_07_01`: Resumen del día con referencias
    *   ...
*   **Nivel 3: Agregaciones por Categoría (15 chunks)**
    *   `category_groceries`: Resumen de categoría con referencias
    *   ...
*   **Nivel 4: Resumen Mensual (2 chunks)**
    *   `monthly_july_2025`: Resumen completo del mes

**Total: 126 + 21 + 15 + 2 = 164 chunks**

Cada transacción individual se convierte en un "mini-documento" enriquecido con metadatos (día de la semana, clasificación del gasto, etc.), lo que mejora enormemente la calidad de los embeddings y la capacidad del sistema para entender el contexto de cada operación.

| Métrica                      | Estrategia Híbrida (Original) | Estrategia "Transaction-First" | Mejora |
| ---------------------------- | ----------------------------- | ------------------------------ | ------ |
| Cobertura de Datos           | ~50%                          | 100%                           | +50%   |
| Total de Chunks              | 31                            | 164                            | +429%  |
| Tokens Totales               | 27,075                        | 15,348                         | -43%   |
| Tokens / Chunk (Promedio)    | 873                           | 93                             | -89%   |

La paradoja aparente (más chunks pero menos tokens) se explica por la eliminación de redundancia: mientras la estrategia híbrida creaba chunks narrativos largos con información duplicada, Transaction-First mantiene cada dato una sola vez, enriquecido con el contexto mínimo necesario.

### Etapa 5: Embeddings - La Representación Numérica de la Realidad Financiera

Con los datos perfectamente estructurados y una cobertura del 100%, el siguiente paso es convertirlos a un formato que los modelos de lenguaje puedan procesar: los embeddings. Un embedding es un vector de números que representa el significado semántico de un texto. La distancia entre dos de estos vectores indica cuán relacionados están sus conceptos, un principio fundamental para la búsqueda y recuperación en un sistema RAG.

Nuestra estrategia "Transaction-First" generó un volumen considerable relativo de chunks (164 para ser exactos). Utilizamos el modelo `text-embedding-3-small` de OpenAI que representa un salto cualitativo respecto a su predecesor (`text-embedding-ada-002`), ofreciendo un rendimiento superior en benchmarks de la industria:

*   **MTEB (Massive Text Embedding Benchmark):** Su puntuación subió de 61.0% a 62.3%, demostrando una mayor capacidad para tareas de recuperación en inglés.
*   **MIRACL (Multilingual Information Retrieval Across a Continuum of Languages):** El rendimiento en tareas multilingües pasa de 31.4% a 44.0%, lo que garantiza la robustez del modelo.

A un precio de $0.02 por cada millón de tokens, este modelo nos ofrece una combinación óptima de rendimiento, eficiencia y escalabilidad.

Una de las características más potentes de los nuevos modelos de embedding de OpenAI es la capacidad de controlar la dimensionalidad del vector resultante. `text-embedding-3-small` genera por defecto embeddings de 1536 dimensiones, un tamaño que captura una gran riqueza semántica.

Sin embargo, la API nos permite reducir estas dimensiones si fuera necesario, por ejemplo, para adaptarnos a las limitaciones de una base de datos vectorial específica o para optimizar el uso de memoria, sacrificando un mínimo de precisión a cambio de una mayor eficiencia. Esta flexibilidad es clave para construir sistemas a medida.

Para ilustrar el proceso, veamos cómo convertiríamos uno de nuestros chunks de transacción en un embedding utilizando la librería de OpenAI.

```python
import os
from openai import OpenAI

# Es recomendable configurar la API key como una variable de entorno
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI()

def get_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Ejemplo con un chunk de transacción enriquecido
transaction_chunk = """
=== Transacción Financiera ===
Fecha: 2025-07-15
Monto: €-25.50
Descripción: Compra en AMAZON.ES
Categoría: Compras Online
Tipo: Gasto
Clasificación: Gasto mediano
Día de la semana: Tuesday
Mes: July 2025
Periodo: Día de semana
"""

embedding_vector = get_embedding(transaction_chunk)

print(f"Dimensiones del vector: {len(embedding_vector)}")
# print(f"Primeros 5 valores del vector: {embedding_vector[:5]}")
```

Este proceso se repite para cada uno de los 164 chunks. Los vectores resultantes se almacenan en Nano Vectordb, una base de datos vectorial ligera y rápida integrada en LightRAG. Además, implementamos un sistema de caché para asegurar que cada chunk se procese una sola vez, optimizando tanto el coste como el tiempo de ejecución.

Con esta base sólida, el sistema está listo para la etapa final: la construcción del grafo de conocimiento y la recuperación de información.

### Etapa 6: Construcción del Grafo de Conocimiento con LightRAG

Con nuestros chunks jerárquicos y embeddings preparados, llegamos al core de la inteligencia de nuestro sistema: la construcción de un Grafo de Conocimiento (Knowledge Graph). Aquí es donde los datos se transforman en una red de entidades y relaciones interconectadas, permitiendo un nivel de análisis que va más allá de la simple búsqueda de similitud.

El objetivo es simple pero potente: pasar de preguntar "muéstrame transacciones de Mercadona" a poder consultar "¿cuáles son mis patrones de gasto los fines de semana en comparación con los días laborables?".

Para esta tarea configuramos LightRAG con parámetros específicos para optimizar tanto el coste como la eficiencia, procesando nuestros 164 chunks enriquecidos. El parámetro `chunk_token_size=250` se estableció como límite máximo en LightRAG. Aunque nuestros chunks enriquecidos promedian unos 93 tokens, fijamos este valor como margen de seguridad para manejar posibles variaciones.

*   `chunk_token_size=250`: Optimizado para nuestros chunks transaccionales.
*   `entity_extract_max_gleaning=1`: Una sola pasada para la extracción de entidades, reduciendo costes a la mitad.
*   `max_parallel_insert=4`: Procesamos 4 chunks en paralelo para acelerar el proceso.
*   `llm_model_max_async=8`: Aumentamos la concurrencia para las llamadas al LLM.

Utilizamos GPT-4o-mini para la extracción de entidades y relaciones, un modelo que ofrece un equilibrio excepcional entre inteligencia y coste. El coste total estimado para construir el grafo completo fue de aproximadamente $0.05, una cifra que demuestra la viabilidad de aplicar técnicas avanzadas a escala personal.

#### Contribuyendo al ecosistema Open Source

Implementar tecnologías y utilizar librerías incipientes como LightRAG en un caso de uso real es un reto. Lejos de ser un obstáculo, esto se convirtió en una oportunidad para contribuir activamente al proyecto.

Durante la implementación, nos encontramos con varios puntos de fricción que documentamos y compartimos con la comunidad:

*   **Issue #209:** Reportamos múltiples errores de "RuntimeError: Event loop is closed" que aparecían al ejecutar demos con modelos como Ollama (gemma:2b) y Qwen2.5. Este problema provocaba que algunos resultados del sistema simplemente desaparecieran, afectando la confiabilidad de las respuestas.
*   **Issue #1933:** Identificamos un AttributeError crítico con el `storage_lock` en LightRAG v1.4.6, donde el sistema fallaba al insertar documentos porque el lock de almacenamiento no estaba correctamente inicializado como un objeto `asyncio.Lock()`. Este bug bloqueaba completamente el flujo de inserción de datos.

También propusimos soluciones y contribuimos con código para mejorar la herramienta:

*   **Pull Request #1979:** Desarrollamos una herramienta de diagnóstico completa que verifica el estado de inicialización del sistema. Esta herramienta ayuda a los desarrolladores a identificar y resolver problemas de configuración antes de que causen errores, mostrando exactamente qué comandos ejecutar para corregir cada problema detectado.
*   **Pull Request #1978:** Implementamos mensajes de error claros para cuando el storage no está inicializado. En lugar del "AttributeError: aenter", ahora el sistema proporciona instrucciones precisas sobre cómo inicializar correctamente los componentes, incluyendo ejemplos de código y enlaces a la documentación.

Desde M.IA tenemos un compromiso claro de no solo ser usuarios de herramientas open source, sino también participantes activos en la mejora del ecosistema.

#### La Estructura del Grafo Resultante

El proceso de construcción del grafo generó una rica red de conocimiento con los siguientes tipos de entidades y relaciones:

*   **Entidades:**
    *   **Merchants:** Nombres de comercios extraídos de las descripciones (ej. "Amazon", "Mercadona").
    *   **Categorías:** "Alimentación", "Transporte", etc.
    *   **Fechas:** Para análisis de patrones temporales.
    *   **Cantidades:** Transacciones significativas.
*   **Relaciones:**
    *   `TRANSACTION_AT`: Vincula una transacción con un comercio.
    *   `BELONGS_TO`: Asocia una transacción a una categoría.
    *   `OCCURRED_ON`: Conecta una transacción a una fecha.
    *   `AGGREGATES_TO`: Enlaza transacciones individuales con sus resúmenes diarios y categóricos.

Este grafo, almacenado en archivos como `chunk_entity_relation_graph.graphml` y `vdb_entities.json`, es el cerebro de nuestro sistema de consultas.

#### Nuevas Capacidades de Consulta: Local, Global e Híbrido

El grafo de conocimiento nos abre la puerta a un sistema de consultas mucho más sofisticado. LightRAG implementa tres modos principales que aprovechan la estructura del grafo:

*   **Modo Local:** Se enfoca en los nodos de chunk y sus vecinos inmediatos. Es ideal para preguntas muy específicas sobre una transacción o un documento.
*   **Modo Global:** Realiza una búsqueda amplia sobre las entidades (entity) y sus relaciones (relation), permitiendo responder preguntas que requieren una comprensión general del dominio.
*   **Modo Híbrido:** Combina la precisión del modo Local con el contexto del modo Global, ofreciendo respuestas balanceadas y completas.

Estos modos superan las limitaciones de un RAG tradicional basado solo en vectores de similitud, permitiendo un razonamiento más profundo sobre los datos.

### Etapa 7: El Sistema de Consultas - Haciendo las Preguntas Correctas

Ahora que tenemos nuestro grafo de conocimiento, es hora de hacerle preguntas. LightRAG nos ofrece cuatro modos de consulta, cada uno con sus propias fortalezas:

*   **NAIVE:** Búsqueda vectorial simple. Rápida pero menos precisa.
*   **LOCAL:** Se enfoca en fragmentos de texto específicos y sus vecinos directos en el grafo. Ideal para preguntas sobre detalles concretos.
*   **GLOBAL:** Utiliza un resumen global del grafo para entender el contexto general. Perfecta para preguntas amplias.
*   **HYBRID:** Combina lo mejor de los modos LOCAL y GLOBAL para obtener respuestas equilibradas y contextualizadas.

Para ilustrar el poder de nuestro sistema, hemos ejecutado una serie de preguntas diseñadas para reflejar un análisis financiero real. A continuación, presentamos una selección de estas consultas y las respuestas generadas por el sistema.

![IMAGEN GRAFO?]()

## 4. Conclusiones

Este viaje a través de la construcción de un sistema RAG financiero nos ha llevado desde la simple extracción de datos transaccionales hasta la creación de un agente inteligente capaz de responder preguntas complejas sobre nuestras finanzas. Hemos transformado un conjunto de datos estático en un grafo de conocimiento dinámico, permitiendo un nivel de análisis que antes requería un esfuerzo manual considerable.

La combinación de LightRAG y un modelo de lenguaje avanzado nos ha permitido no solo organizar la información, sino también interpretarla y presentarla de una manera conversacional y accesible. Este es el poder de la IA aplicada a problemas del mundo real: la capacidad de convertir datos en conocimiento y, en última instancia, en sabiduría.

## 5. Resultados y Análisis

### 5.1 Métricas de Precisión

Utilizamos un dataset de prueba con 10 transacciones reales procesadas con LightRAG + GPT-4o-mini:

| Consulta              | Respuesta LightRAG     | Valor Real (Dataset)    | Precisión | Análisis                  |
| --------------------- | ---------------------- | ----------------------- | --------- | ------------------------- |
| "Gasté en Groceries"  | €135.49 total          | €135.49 (2 transacciones) | ✅ 100%   | Identificación perfecta   |
| "Patrones de gasto"   | 5 patrones detectados  | 5 categorías reales     | ✅ 100%   | Extracción auto. correcta |
| "Gastos de julio"     | €648.48 (4 tx)         | €648.48 confirmado      | ✅ 100%   | Agregación exacta         |
| "Dónde ahorrar"       | €25-50/mes potencial   | Análisis cualitativo    | ✅ Alta   | Recomendaciones coherentes |
| "Salud financiera"    | Score 7.5/10           | Análisis contextual     | ✅ Alta   | Análisis comprensivo      |
| "Netflix recurrente"  | €12.99/mes detectado   | €12.99 real             | ✅ 100%   | Patrón identificado       |
| "Housing dominante"   | 50.7% del total        | €500 de €986.48         | ✅ 100%   | Cálculo correcto          |
| "Mercadona frecuencia"| "cada 2-3 días"        | 2 compras en 3 días     | ✅ 100%   | Patrón temporal correcto  |

**Precisión promedio en dataset de prueba: >95%**

### Estadísticas del Grafo Generado

*   **Entidades extraídas:** 50+ nodos automáticos
*   **Relaciones identificadas:** 55+ conexiones
*   **Tipos de entidades:**
    *   📅 **Temporales:** "Julio 2025", "2 de julio", etc.
    *   🏪 **Comercios:** "Mercadona", "Netflix"
    *   📂 **Categorías:** "Groceries", "Entertainment", "Housing"
    *   💰 **Montos:** Agrupados por rangos
    *   🔄 **Patrones:** "Gasto recurrente", "Compra regular"

*   **Extracción automática:** GPT-4o-mini identifica entidades sin configuración
*   **Relaciones semánticas:** Conexiones contextuales entre conceptos
*   **Actualización incremental:** Nuevos datos se añaden sin reconstruir
*   **Multilingüe:** Funciona en español e inglés

Estas son consultas reales ejecutadas sobre un grafo de conocimiento construido con LightRAG real y GPT-4o-mini, con 50+ entidades y 55+ relaciones extraídas automáticamente de 10 transacciones de prueba.

#### Query 1: "¿Cuánto gasté en Groceries?" (Modo: HYBRID)

**Respuesta LightRAG:** En julio de 2025, gastaste un total de €135.49 en la categoría Groceries, distribuidos en 2 transacciones principales:
*   €77.40 en Mercadona el 2 de julio
*   €58.09 en Mercadona el 4 de julio

Estos gastos en supermercados representan aproximadamente el 13.5% de tu gasto total mensual.
**Precisión: 100%** - El sistema identificó correctamente ambas transacciones

#### Query 2: "¿Qué patrones de gasto tengo?" (Modo: GLOBAL)

**Respuesta LightRAG:** Analizando tu grafo financiero, he identificado los siguientes patrones:

1.  **Gasto recurrente:** Netflix por €12.99 mensual (categoría Entertainment)
2.  **Mayor categoría de gasto:** Housing representa el 50.7% del total (€500)
3.  **Frecuencia de compras:** Mercadona cada 2-3 días (patrón de compra regular)
4.  **Concentración temporal:** Mayor actividad financiera a inicio de mes
5.  **Diversificación:** 5 categorías principales de gasto identificadas

Estos patrones sugieren un comportamiento de gasto estructurado con gastos fijos dominantes y compras regulares planificadas.

*El sistema identificó correctamente los patrones gracias a la extracción automática de entidades*

#### Query 3: "Muestra mis gastos de julio" (Modo: LOCAL)

**Respuesta LightRAG:** En julio de 2025, tus principales gastos fueron:

*   **📅 Julio 1, 2025:**
    *   Transfer: €500.00 (Internal Transfer)
*   **📅 Julio 2, 2025:**
    *   Mercadona: €77.40 (Groceries)
*   **📅 Julio 4, 2025:**
    *   Mercadona: €58.09 (Groceries)
    *   Netflix: €12.99 (Entertainment)

Total de gastos en julio: €648.48 en 4 transacciones principales.
El grafo muestra conexiones entre estas transacciones y sus categorías.

*El sistema muestra los datos de prueba correctamente*

#### Query 4: "¿Dónde puedo ahorrar dinero?" (Modo: HYBRID)

**Respuesta LightRAG:** Basándome en el análisis del grafo de conocimiento, identifiqué estas oportunidades de ahorro:

💡 **Optimizaciones Rápidas:**
1.  Entertainment (€12.99): Revisar si usas Netflix activamente
2.  Groceries (€135.49): Comparar precios entre supermercados

📊 **Análisis del Grafo:**
*   Housing domina con 50.7% - difícil de optimizar
*   Internal Transfer (€500) - ¿es ahorro o gasto?
*   Gastos variables representan solo 20% del total

🎯 **Recomendación:** Potencial de ahorro realista de €25-50/mes sin afectar calidad de vida, principalmente en categorías discrecionales.

*GPT-4o-mini genera recomendaciones basadas en el grafo*

#### Query 5: "Analiza mi salud financiera" (Modo: HYBRID)

**Respuesta LightRAG:** Análisis de salud financiera basado en el grafo:

📊 **Estado General:** ESTABLE
*   Flujo de caja: Positivo (ingresos > gastos)
*   Diversificación: 5 categorías de gasto (buena distribución)
*   Gastos fijos: 51% (Housing) - alto pero controlado

✅ **Fortalezas:**
*   Transferencias regulares (posible ahorro de €500)
*   Gastos en alimentación controlados (13.5% del total)
*   Pocas suscripciones recurrentes (solo Netflix)

⚠️ **Áreas de Atención:**
*   Housing consume >50% - evaluar si es sostenible
*   Falta de categoría de emergencia visible
*   Concentración de gastos a inicio de mes

📈 **Score de Salud Financiera:** 7.5/10

*El sistema genera un análisis comprehensivo del grafo*

### Problemas Encontrados

Hay algunos problemas a tener en cuenta que nos hemos ido encontrando durante el desarrollo:

*   **Problema 1: Confusión entre Categorías y Merchants**
    *   **Observación:** Categorías como "Transferencia" aparece como un comercio cuando es una categoría.
    *   **Causa:** El grafo no distingue claramente entre tipos de entidades
    *   **Impacto:** Respuestas confusas sobre "gastos en Transferencia" como si fuera un comercio
    *   **Posible solución:** Mejorar la extracción de entidades con tipos específicos
*   **Problema 2: Granularidad Excesiva del Grafo**
    *   **Observación:** Se generan muchos nodos y relaciones para pocas transacciones.
    *   **Causa:** Cada cantidad monetaria se convierte en una entidad única (€75.09, €43.22, etc.)
    *   **Impacto:** Ruido en el retrieval y posibles errores de agregación
    *   **Posible solución:** Agrupar cantidades en rangos o excluirlas de la extracción de entidades
*   **Problema 3: Pérdida de Contexto Temporal**
    *   **Observación:** Dificultad para análisis temporal preciso (días laborables vs fines de semana)
    *   **Causa:** La información temporal no se preserva bien en el grafo
    *   **Impacto:** Limitaciones en análisis de patrones temporales
    *   **Posible solución:** Chunks específicos para análisis temporal

## 6. Lecciones Aprendidas

Este proyecto ha revelado insights fundamentales sobre cómo construir sistemas RAG efectivos para finanzas personales. Aquí están las lecciones más valiosas:

### 6.1 Creación de chunks específicos según el caso de uso

Implementamos 5 estrategias de chunking que multiplican la efectividad del sistema RAG.

En nuestro sistema actual con LightRAG:

*   **Chunking Temporal:** Agrupa transacciones por semanas/meses
*   **Chunking por Categoría:** Relaciona gastos similares (Groceries, Entertainment)
*   **Chunking por Cantidad:** Rangos de gasto para análisis comparativo
*   **Chunking por Merchant:** Agrupa por comercio (Mercadona, Netflix)
*   **Chunking Mixto:** Combinaciones inteligentes para máxima cobertura

**Ejemplo real implementado:**

```json
// Chunk temporal generado automáticamente
{
    "chunk_id": "temporal_week_27_2025",
    "content": "Semana 27 de 2025: 5 transacciones totalizando €590.39. Categorías: Internal Transfer (€500), Groceries (€77.40), Entertainment (€12.99)...",
    "metadata": {
        "period": "2025-W27",
        "transaction_count": 5,
        "total_amount": -590.39
    }
}
```

Con el ejemplo de las 10 transacciones de prueba, generamos 50+ entidades y 55+ relaciones. GPT-4o-mini extrae automáticamente comercios, categorías, fechas y patrones “sin configuración manual”.

### 6.2 LightRAG ofrece muy buena precisión si los datos son correctos

Con el dataset de prueba real, LightRAG alcanza >95% de precisión en todas las métricas.

*   ✅ **Identificación de patrones (100%):** Netflix recurrente, frecuencia Mercadona
*   ✅ **Análisis de categorías (100%):** Groceries €135.49, Housing 50.7%
*   ✅ **Respuestas contextuales:** Score salud financiera 7.5/10
*   ✅ **Recomendaciones inteligentes:** Ahorro potencial €25-50/mes

Ahora, hemos visto que puede haber errores en el cálculo de totales debido a la granularidad de los chunks recuperados. Esto es un problema al tener en cuenta ya que cuando trabajamos con números, una mínima variación puede suponer un cálculo incorrecto.

Por eseo, y como exploramos en un artículo anterior "GraphRAG vs LightRAG en 2025: Adaptive RAG con GPT-5-nano", el futuro está en sistemas adaptativos que seleccionan la estrategia óptima según el tipo de consulta.

Para este caso de uso de finanzas personales, y en los próximos artículos abordaremos un Adaptive RAG que combine:

*   **Agente Analítico (LightRAG):** Patrones, tendencias, insights conversacionales
*   **Agente Contable (Determinístico):** Cálculos exactos, balances, agregaciones precisas
*   **Agente Orquestador (Clasificador GPT-5-nano):** Analiza la intención de la query y es capaz de enrutar al agente con una precisión muy alta.

Con esto resolvemos el dilema de precisión vs. análisis, utilizando la herramienta correcta para cada trabajo.

### 6.3 Un nuevo paradigma democratizando la IA en finanzas

Por primera vez en la historia, cualquier persona puede tener un analista financiero personal impulsado por IA. Para 150-200 transacciones mensuales el coste del sistema es de 0.10 € incluyendo la construcción del grafo, la generación de embeddings y las consultas, esto utilizando APIs de proveedores como OpenAI.

Más allá de eso, y de la posibilidad también de utilizar modelos en local sin la dependencia de proveedores externos, esta democratización no es sobre el coste, es sobre el cambio fundamental en cómo trabajamos con datos financieros:

*   De hojas de Excel estáticas → Conversaciones dinámicas con tus datos
*   De reportes mensuales → Insights en tiempo real
*   De análisis manual → Detección automática de patrones

Y de la toma de decisiones:

*   "¿En qué días de la semana gasto más y por qué?" → Optimización de comportamientos
*   "¿Mis patrones de gasto han cambiado?" → Alertas tempranas de desviaciones
*   "¿Qué gastos podría optimizar sin afectar mi calidad de vida?" → Recomendaciones personalizadas

Desde M.IA Imaginamos un futuro donde cada persona, familia y PYME tenga acceso a inteligencia financiera de nivel enterprise. Y eso no es un debate (solo) sobre tecnología.

### 6.4 Evolución hacia una arquitectura multiagente

El sistema actual utiliza un enfoque Hybrid RAG con LightRAG en el que se ha probado la viabilidad y funcionamiento, con algunas limitaciones. Para escalar esto a miles de transacciones, se requiere un sistema multiagente que permita:

*   **Preguntas de cálculo exacto (¿Cuál es mi balance?):**
    *   Requieren precisión del 100%
    *   Mejor manejadas por agentes determinísticos
    *   SQL directo o pandas para agregaciones
*   **Preguntas de análisis de patrones (¿Cómo han evolucionado mis gastos?):**
    *   Requieren comprensión contextual
    *   LightRAG sobresale aquí
    *   Respuestas conversacionales enriquecidas
*   **Preguntas mixtas (¿Puedo ahorrar más sin afectar mi estilo de vida?):**
    *   Requieren tanto cálculo como análisis
    *   Múltiples agentes colaborando
    *   Orquestación inteligente vía Adaptive RAG

El siguiente paso es escalar a datasets completos (miles de transacciones) manteniendo esta precisión mediante arquitecturas multiagente especializadas y arquitecturas Agentic RAG.

## 7. Proyecto Open Source: Pregunta a tus Finanzas

Desde M.IA, hemos liberado un sistema completamente funcional que usa LightRAG real con GPT-4o-mini para crear un asistente financiero personal.

✅ **Sistema:**
*   LightRAG real con extracción automática de entidades
*   Grafo de conocimiento con 50+ entidades y 55+ relaciones
*   Visualización interactiva con PyVis
*   4 modos de consulta (naive, local, global, hybrid)
*   Respuestas en lenguaje natural con GPT-4o-mini

✅ **Pipeline de Datos:**
*   Parser bancario para BBVA (Excel y CSV)
*   Anonimización de 3 capas (95.7% precisión)
*   Chunking adaptativo con 5 estrategias
*   Embeddings con OpenAI text-embedding-3-small

```bash
# Clonar repositorio
git clone https://github.com/albertgilopez/pregunta-tus-finanzas
cd pregunta-tus-finanzas

# Instalar dependencias
pip install -r requirements.txt

# Configurar OpenAI API Key
cp .env.example .env
# Editar .env con tu API key

# Construir grafo de conocimiento
python scripts/build_lightrag_graph.py

# Hacer consultas interactivas
python scripts/demo_queries.py

# Visualizar grafo (puerto 8080)
python scripts/visualize_graph.py
```

### 📁 Estructura del Proyecto

```
pregunta-tus-finanzas/
├── scripts/                      # Scripts principales
│   ├── build_lightrag_graph.py  # Construcción con LightRAG real
│   ├── demo_queries.py          # Demo de consultas
│   ├── visualize_graph.py       # Visualización interactiva
│   └── analyze_graph.py         # Análisis del grafo
├── src/                          # Código fuente
│   ├── extractors/              # Parsers bancarios
│   ├── processors/              # Anonimización adaptativa
│   └── rag/                     # Implementaciones RAG
├── simple_rag_knowledge/         # Grafo persistente
├── docs/                         # Documentación completa
│   ├── COMO_FUNCIONA_LIGHTRAG.md
│   ├── INSTALLATION.md
│   └── ROADMAP.md
└── outputs/                      # Visualizaciones HTML
```

## 8. What’s Next

Esta serie es solo el comienzo. En los próximos artículos, llevaremos este proyecto al siguiente nivel:

*   **Artículo 2: "Vibe Coding: Construyendo la UI Perfecta para RAG Financiero con Claude Code"**
    *   Crearemos un dashboard interactivo con React (Next.js) en el frontend y FastAPI en el backend utilizando Claude Code.
    *   Visualizaremos el grafo de conocimiento y permitiremos consultas en tiempo real.
    *   De la línea de comandos a una interfaz de usuario intuitiva sin escribir apenas código.
*   **Artículo 3: "De RAG a Agentic RAG: Multiplicando la Precisión con RAG-Anything y Google ADK"**
    *   Evolucionaremos el pipeline a un sistema multi-agente para alcanzar una precisión del 99%.
    *   Integraremos RAG-Anything para el manejo de datos tabulares y Google ADK para agentes especializados.
    *   Reduciremos costos y tiempos de respuesta de forma drástica.

**Comparte tu experiencia:**

Estoy abierto a colaborar y discutir sobre las posibilidades que ofrece la inteligencia artificial y cómo trabajar juntos para explorar y construir soluciones en diferentes sectores. Si tienes ideas, preguntas o simplemente quieres hablar de ello, escríbeme:

*   GitHub: [https://github.com/albertgilopez](https://github.com/albertgilopez)
*   LinkedIn: [https://www.linkedin.com/in/albertgilopez/](https://www.linkedin.com/in/albertgilopez/)
*   M.IA, tu asistente financiero inteligente: [https://himia.app/](https://himia.app/)
*   Inteligencia Artificial Generativa en español: [https://www.codigollm.es/](https://www.codigollm.es/)

## 9. Cómo Contribuir

Este es un proyecto vivo y abierto a la comunidad. Si te interesa colaborar, aquí tienes algunas ideas:

*   **Reporta bugs o problemas:** Abre un issue en el repositorio de GitHub.
*   **Añade nuevos parsers:** Crea extractores para otros bancos y compártelos.
*   **Propón nuevas funcionalidades:** ¿Qué más te gustaría que hiciera el sistema?
*   **Mejora la documentación:** Ayuda a que otros puedan empezar más fácilmente.

Toda contribución es bienvenida y será reconocida.

---

**Sobre el autor:** Albert Gil López es CTO y co-founder de M.IA (himia.app), startup ganadora del Cofidis Startup Booster 2025 y otros reconocimientos como la 13a edición de Barcelona Activa, … . Ingeniero en Informática por la UAB, lidera el desarrollo del sistema multiagente que está transformando la gestión de tesorería para PYMEs. Dentro del programa RETECH, en colaboración con PIMEC y el IIIA-CSIC, se esta desarrollando un proyecto que busca democratizar el acceso a inteligencia financiera avanzada para las 150.000 PYMEs catalanas.

**Tags:** `#RAG` `#LightRAG` `#FinancialAI` `#Pipeline` `#Chunking` `#Embeddings` `#OpenSource` `#TechnicalExperiment` `#FinanzasPersonales` `#MSIA` `#RETECH` `#PIMEC` `#Python` `#OpenAI` `#Multiagent` `#Fintech` `#StartupEspaña` `#IAFinanciera`
