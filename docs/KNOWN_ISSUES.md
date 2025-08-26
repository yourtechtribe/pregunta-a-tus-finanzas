# Known Issues and Solutions

## LightRAG Async Context Manager Error (`__aenter__`)

### Issue Description
When using LightRAG (lightrag-hku), users encounter an `AttributeError: __aenter__` error when trying to insert or query data. This occurs when the async context managers in the storage layer are not properly initialized.

### Error Message
```
AttributeError: __aenter__
```

### Root Cause
The error occurs in the storage lock mechanism:
```python
File "/home/Usuario/.local/lib/python3.10/site-packages/lightrag/kg/json_doc_status_impl.py", line 68, in filter_keys
    async with self._storage_lock:
AttributeError: __aenter__
```

### Official Repository Issue
This is a known issue tracked in the official HKUDS/LightRAG repository:
- GitHub: https://github.com/HKUDS/LightRAG/issues
- Related issues: #255 (Unstable context), #917 (File reading issues)

### Solutions

#### Solution 1: Proper Initialization
Ensure both storage and pipeline status are initialized:

```python
import os
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

# Enable logging to debug issues
setup_logger("lightrag", level="INFO")

# Create working directory
WORKING_DIR = "./rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

# Initialize with proper configuration
rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
)
```

#### Solution 2: Use Synchronous Methods
If async issues persist, use synchronous methods:

```python
# Instead of async methods
# await rag.ainsert(text)
# await rag.aquery(query)

# Use synchronous versions
rag.insert(text)
rag.query(query)
```

#### Solution 3: Check Dependencies
Ensure all required dependencies are installed:

```bash
pip install --upgrade lightrag-hku
pip install aiofiles asyncio
```

### Workaround for This Project

Due to this known issue, we recommend using alternative approaches:

1. **For Simple Queries**: Use the synchronous API when possible
2. **For Complex Graphs**: Consider using alternative RAG implementations like LangChain or LlamaIndex
3. **For Production**: Wait for the official fix or use a stable release

### Testing Status

As of testing on 2025-08-26:
- **Version**: lightrag-hku 1.4.6
- **Python**: 3.10.12
- **Status**: Issue persists in both async and sync methods
- **Impact**: Knowledge graph building fails, preventing RAG functionality

### References

- [HKUDS/LightRAG GitHub Repository](https://github.com/HKUDS/LightRAG)
- [LightRAG PyPI Package](https://pypi.org/project/lightrag-hku/)
- [Issue #255: Unstable Context](https://github.com/HKUDS/LightRAG/issues/255)

## Alternative RAG Implementations

Given the current issues with LightRAG, consider these alternatives:

### 1. LangChain with Chroma
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
```

### 2. LlamaIndex
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
```

### 3. Custom Implementation
Use OpenAI embeddings directly with a vector database like Pinecone or Weaviate.

---

*Last updated: 2025-08-26*
*Documented during testing of pregunta-tus-finanzas system*