# "Pregunta a tus Finanzas": Cómo Construí un "ChatGPT" para mis Extractos Bancarios (y lo estoy convirtiendo en un Sistema Multi-Agente)

**Autor: Albert Gil López**
**CTO @ M.IA | Tiempo de lectura: 15 min | Proyecto Open Source | Código Disponible | Experimento Técnico Detallado**

![RAG Finance Pipeline](banner-image-placeholder.jpg)

## TL;DR - Lo que hemos construido (y lo que aprenderás)

- **Pipeline de Indexación Completo (Fase 1)**: Un sistema robusto que cubre las 6 etapas iniciales del ciclo RAG: ingesta, limpieza, chunking, embeddings, construcción de grafos y sistema de queries.
- **Anonimización Inteligente y Adaptativa**: Alcanzamos un **95.7% de precisión** en la detección de PII usando un sistema de 3 capas que aprende y mejora con el tiempo, sin dependencias externas.
- **100% Privado y Open Source**: Todo el código y los datos están en GitHub. Para proteger tu privacidad, los datos son anonimizados antes de ser procesados por cualquier proveedor externo via API.
- **Costo Ultra-Bajo**: Procesar las 126 transacciones de una persona durante un mes costó solo **$0.03**, demostrando una viabilidad económica excepcional para el análisis financiero personal.
- **Sistema de Consultas Multi-Modo**: Implementamos 4 modos de consulta (naive, local, global, híbrido) con latencias de 4 a 9 segundos, permitiendo un análisis flexible.
- **Lecciones del Mundo Real**: Aprende de una implementación práctica sobre datos bancarios reales, incluyendo los errores y las soluciones que nos llevaron a un sistema funcional.

## 1. Introducción: De un análisis teórico a una implementación práctica

> "¿Y si pudieras preguntarle directamente a tus finanzas personales cualquier cosa y obtener respuestas instantáneas e inteligentes?"

En un artículo anterior, exploramos en detalle las [8 arquitecturas RAG que definen el panorama en 2025](https://medium.com/@jddam/graphrag-vs-lightrag-en-2025-adaptive-rag-con-gpt-5-nano-caso-real-uab-97e3ce509fba), desde el RAG básico hasta los sistemas multi-agente. Pero la teoría solo te lleva hasta cierto punto. La pregunta inevitable era: **¿cómo se comporta este ecosistema de arquitecturas con la complejidad de los datos financieros del mundo real?**

Este experimento nace de esa curiosidad técnica y de la necesidad de explorar arquitecturas RAG robustas para el sector financiero. Aunque el objetivo inicial era entender profundamente cada decisión arquitectónica usando mis propios extractos bancarios, esta saga de tres artículos busca sentar las bases de una herramienta funcional. La meta final está alineada con la visión de **M.IA**: optimizar la tesorería, empezando por las finanzas personales para luego escalar el modelo a las pymes catalanas y ayudarles a tomar mejores decisiones.

Este artículo es el primero de una serie de tres. En el segundo, construiremos una interfaz de usuario interactiva vibe-codeando con Claude Code, y en el tercero, evolucionaremos nuestro pipeline a un sistema multi-agente utilizando el Agent Development Kit (ADK) de Google. 

## 2. Background: El Problema que Resolvemos en M.IA

En **M.IA** (himia.app), hemos identificado una brecha tecnológica crítica. Según el "Barómetro de adopción de la IA en las pymes españolas" de IndesIA 2025, solo el 2.9% de las PYMEs españolas utilizan herramientas de inteligencia artificial, y en Cataluña, el porcentaje es de apenas el 3.7%. Esta brecha es especialmente grave en la gestión financiera, donde las aplicaciones tradicionales siguen el mismo patrón de hace una década: importas tu CSV, obtienes gráficos predefinidos, categorización automática mediocre (60-70% de precisión), y cero capacidad de hacer preguntas complejas sobre tus datos.

Mi frustración personal, que refleja la de miles de emprendedores y gestores financieros, vino cuando intenté responder una pregunta aparentemente simple: "¿Cuánto gasto en promedio cuando salgo a cenar con amigos vs cuando como solo?". Ninguna app podía responderlo. Necesitaba contexto, relaciones, y comprensión semántica que va más allá de simples categorías.

Esta experiencia personal alimenta directamente también nuestro trabajo en el proyecto con el IIIA-CSIC y PIMEC, donde estamos desarrollando un sistema multiagente de IA para transformar las transacciones económicas entre PYMEs catalanas y optimizar su tesorería. El concepto "Pregunta a tus Finanzas" nace de esta visión: democratizar el acceso a inteligencia financiera avanzada, explicable, que genere confianza y sirva para tomar mejores decisiones.

### ¿Por qué RAG para Finanzas Personales?

La tecnología RAG (Retrieval-Augmented Generation) es una de las aplicaciones más potentes que los LLMs han habilitado, permitiendo la creación de chatbots sofisticados de preguntas y respuestas (Q&A) que pueden razonar sobre información específica de una fuente de datos privada.

Una aplicación RAG típica tiene dos componentes principales:

1.  **Indexación**: un proceso para ingestar datos de una fuente y organizarlos. Esto generalmente ocurre offline e implica:
    *   **Carga (Load)**: Cargar los datos, en nuestro caso, los extractos bancarios.
    *   **División (Split)**: Dividir grandes documentos en trozos más pequeños para facilitar la búsqueda y el procesamiento por parte del modelo.
    *   **Almacenamiento (Store)**: Guardar e indexar estos trozos, a menudo utilizando un `VectorStore` y un modelo de `Embeddings`.

2.  **Recuperación y Generación**: la cadena RAG en tiempo de ejecución que toma la consulta del usuario, recupera los datos relevantes del índice y luego los pasa al modelo para generar una respuesta.
    *   **Recuperación (Retrieve)**: Dado una entrada del usuario, se recuperan los trozos relevantes del almacenamiento.
    *   **Generación (Generate)**: Un `ChatModel` / `LLM` produce una respuesta utilizando un prompt que incluye tanto la pregunta como los datos recuperados.

Esta arquitectura es ideal para nuestro caso de uso porque nos permite construir un sistema que no solo "habla" sobre finanzas, sino que "entiende" y "razona" sobre *tus* finanzas, manteniendo la privacidad y el contexto.

RAG (Retrieval-Augmented Generation) combina lo mejor de dos mundos:
- **Precisión de datos estructurados**: Cada transacción es un hecho inmutable
- **Comprensión contextual de LLMs**: Capacidad de entender preguntas naturales y encontrar patrones

Pero aplicar RAG a datos financieros presenta desafíos únicos:

1. **Privacidad extrema**: Los datos bancarios son altamente sensibles
2. **Estructura tabular**: RAG tradicionalmente funciona mejor con texto narrativo
3. **Precisión requerida**: Un error del 5% en categorización puede significar cientos de euros mal contabilizados (spoiler: el objetivo no es la contabilidad, sino el análisis de patrones y tendencias).
4. **Contexto temporal**: Las transacciones tienen relaciones temporales y causales

### LightRAG: La Elección Técnica

Entre las opciones disponibles en 2025, como GraphRAG de Microsoft, LlamaIndex o LangChain, elegí **LightRAG de la Universidad de Hong Kong (HKU)**. La decisión se basa en su enfoque pragmático y eficiente para resolver los desafíos específicos de este proyecto. Como se detalla en su paper académico (https://arxiv.org/abs/2410.05779), LightRAG fue diseñado para superar las limitaciones de los sistemas RAG tradicionales que dependen de representaciones de datos planas, incorporando estructuras de grafos para mejorar la conciencia contextual.

Las tres razones clave para esta elección son:

1.  **Arquitectura Híbrida y Velocidad**: A diferencia de sistemas puramente basados en grafos como GraphRAG, que pueden tardar entre 30 y 40 segundos por consulta, LightRAG opera como un **Hybrid RAG**. Combina búsqueda vectorial, búsqueda en grafo y búsqueda textual simple (naive) en paralelo, fusionando y reordenando los resultados. Esto le permite ofrecer tiempos de respuesta de entre 20 y 100 milisegundos, ideal para las consultas directas y factuales que caracterizan el análisis financiero personal.

2.  **Actualización Incremental**: El entorno de las finanzas personales es dinámico. LightRAG está diseñado con un algoritmo de actualización incremental que permite añadir nuevas transacciones o fuentes de datos sin necesidad de reconstruir todo el índice desde cero. Esta capacidad, destacada en su repositorio oficial (https://github.com/HKUDS/LightRAG), es crucial para mantener el sistema relevante y eficiente a lo largo del tiempo.

3.  **Sistema de Recuperación de Doble Nivel**: El paper de LightRAG describe un sistema de recuperación de doble nivel que permite descubrir conocimiento tanto a bajo nivel (entidades específicas) como a alto nivel (conceptos y relaciones complejas). Esta dualidad es perfecta para nuestro caso de uso: podemos preguntar por un gasto concreto ("¿Cuánto costó la cena en 'La Pizzería'?") y también por patrones más amplios ("¿Cuál es mi gasto promedio en restaurantes italianos?").

En resumen, mientras que GraphRAG es una herramienta potente para análisis holísticos y descubrir conexiones ocultas en grandes volúmenes de datos narrativos, LightRAG ofrece una solución más ágil y rápida, optimizada para la velocidad y la relevancia contextual en un entorno de datos que cambia constantemente.

¿Cómo lo hemos implementado? A continuación, desglosamos el pipeline paso a paso.

[El resto del contenido del artículo continúa aquí...]

## 10. Cómo Contribuir

Este es un proyecto vivo y abierto a la comunidad. Si te interesa colaborar, aquí tienes algunas ideas:

- **Reporta bugs o problemas**: Abre un issue en el [repositorio de GitHub](https://github.com/yourtechtribe/pregunta-a-tus-finanzas).
- **Añade nuevos parsers**: Crea extractores para otros bancos y compártelos.
- **Propón nuevas funcionalidades**: ¿Qué más te gustaría que hiciera el sistema?
- **Mejora la documentación**: Ayuda a que otros puedan empezar más fácilmente.

Toda contribución es bienvenida y será reconocida.

---

**Sobre el autor**: Albert Gil López es CTO y co-founder de M.IA (himia.app), startup ganadora del Cofidis Startup Booster 2025. Ingeniero en Informática por la UAB, lidera el desarrollo del sistema multiagente que está transformando la gestión de tesorería para PYMEs. El proyecto RETECH, en colaboración con PIMEC y el IIIA-CSIC, busca democratizar el acceso a inteligencia financiera avanzada para las 150.000 PYMEs catalanas.

**Enlaces**:
- **Código del proyecto**: [github.com/yourtechtribe/pregunta-a-tus-finanzas](https://github.com/yourtechtribe/pregunta-a-tus-finanzas)
- **M.IA (B2B)**: [himia.app](https://www.himia.app)
- **LinkedIn**: [linkedin.com/in/albertgilopez](https://www.linkedin.com/in/albertgilopez/)

**Tags**: #RAG #LightRAG #FinancialAI #Pipeline #Chunking #Embeddings #OpenSource #TechnicalExperiment #FinanzasPersonales #MSIA #RETECH #PIMEC #Python #OpenAI #Multiagent #Fintech #StartupEspaña #IAFinanciera