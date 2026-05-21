<p align="center">
<img width="1081" height="396.5" alt="ChatGPT Image 20 may 2026, 20_23_22" src="https://github.com/user-attachments/assets/744ad33f-b6b2-49ad-98ac-141f7cad38e4" />
<p/>
    
# Memory Etch

**Memoria persistente local-first para agentes AI.** SQLite + FTS5 + HRR vectors + embeddings opcionales.
Sin servicios externos, sin GPU, sin API keys. ~0.8ms por búsqueda.

```bash
pip install "memory-etch[hrr]"
```

---

## Tabla de Contenidos

- [¿Por qué Memory Etch?](#por-qué-memory-etch)
- [Instalación](#instalación)
- [Primeros pasos](#primeros-pasos)
- [Arquitectura](#arquitectura)
- [Características](#características)
- [Embedding Providers](#embedding-providers)
- [MCP Server](#mcp-server)
- [Web Viewer](#web-viewer)
- [Benchmarks](#benchmarks)
- [API](#api)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

---

## ¿Por qué Memory Etch?

Los agentes AI necesitan memoria persistente para ser útiles. Pero las opciones existentes implicaban elegir entre:

- **Dependencia de APIs externas** (Pinecone, OpenAI embeddings) — tu agente deja de funcionar sin internet.
- **Infraestructura pesada** (Chroma, Qdrant, AgentMemory con iii-engine) — 2GB+ de descarga, runtimes externos, config compleja.
- **Archivos JSON artesanales** — crecen como plaga, sin búsqueda, sin estructura.

Memory Etch es el punto medio: **SQLite embedded, sin servidores, sin dependencias obligatorias, sin llamadas externas.** Tu información nunca sale de tu máquina.

```
pip install memory-etch
python -c "from memory_etch import EtchStore; s = EtchStore('memory.db'); print('anda')"
```

Eso es todo lo que necesitás para arrancar.

---

## Instalación

```bash
# Mínimo: FTS5 + Jaccard (solo stdlib de Python)
pip install memory-etch

# Recomendado: FTS5 + HRR vectors (necesita numpy)
pip install "memory-etch[hrr]"

# Con embeddings semánticos locales (BGE-small via fastembed)
pip install "memory-etch[embeddings]"

# Con MCP server (para integrar con agentes vía MCP)
pip install "memory-etch[mcp]"

# Todo junto
pip install "memory-etch[all]"
```

**Requisitos:** Python 3.10+ | Sin GPU | Sin CUDA | Sin runtime externo.

---

## Primeros pasos

```python
from memory_etch import EtchStore, EtchRetriever

# Crear o abrir la base de datos
store = EtchStore("memory.db")

# Guardar hechos
store.add_fact("Python es un lenguaje interpretado", category="tech")
store.add_fact("SQLite soporta FTS5 para búsqueda de texto completo", category="tech")
store.add_fact("FastAPI está construido sobre Starlette", category="tech")

# Guardar con campos estructurados (v1.0)
store.add_fact(
    content="Usar httpx para llamadas HTTP asincrónicas en Python",
    what="Decisión técnica",
    why="httpx tiene mejor soporte de async/await que requests",
    where="src/http_client.py",
    learned="httpx funciona con anyio y trio, no solo asyncio",
)

# Buscar
retriever = EtchRetriever(store)
results = retriever.search("búsqueda de texto completo")
for r in results:
    print(f"[{r['_score']:.2f}] {r['content']}")

# Búsqueda inteligente con fallback automático (v1.0)
results = retriever.search(
    "¿cómo hago requests HTTP en Python?",
    mode="auto",  # FTS5 → HRR multi-query → embeddings (si están configurados)
    limit=5,
)

# Detección automática de proyecto (v1.0)
# Si estás en un repo git, el proyecto se detecta solo del remote origin
store = EtchStore("project.db", project="auto")
```

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    Tu Agente AI                       │
├─────────────────────────────────────────────────────┤
│         MCP Server (stdio)  │  Python API            │
├─────────────────────────────────────────────────────┤
│  EtchRetriever                                        │
│  ┌─────────┬──────────┬───────────┬──────────────┐   │
│  │  FTS5   │   HRR    │  Jaccard  │  Embeddings  │   │
│  │ (exact) │(vectors) │ (n-gram)  │ (semántico)  │   │
│  └────┴────┴────┴─────┴────┴──────┴──────┴───────┘   │
│              │           │                            │
│         Reciprocal Rank Fusion (RRF)                  │
│              │                                        │
│  EtchStore — SQLite + FTS5 + triggers automáticos     │
└─────────────────────────────────────────────────────┘
```

**Tres capas de búsqueda, sin dependencias externas por defecto:**

| Capa | Qué hace | Costo | Dependencia |
|---|---|---|---|
| **FTS5** | Búsqueda exacta por palabras clave | ~0.05ms | stdlib |
| **HRR** | Similaridad semántica holográfica | ~0.8ms | numpy (opt-in) |
| **Jaccard** | Re-ranking por n-gramas | incluido en HRR | numpy (opt-in) |
| **Embeddings** | Búsqueda semántica densa | ~185ms | fastembed (opt-in) |

Por defecto usa solo FTS5 + Jaccard. Con `pip install memory-etch[hrr]` ganás HRR.
Con `pip install memory-etch[embeddings]` ganás embeddings densos.
Cada nivel es opcional, aditivo, y retrocompatible.

---

## Características

### Core (v0.x)

| Feature | Descripción |
|---|---|
| **FTS5** | Búsqueda de texto completo con triggers auto-sincronizados |
| **HRR vectors** | Representaciones holográficas sin modelos, sin GPU |
| **Jaccard re-rank** | Overlap de n-gramas para ordenar resultados |
| **Soft delete** | Los hechos no se borran, se ocultan |
| **Consolidación activa** | LLM decide ante hechos duplicados o contradictorios |
| **Entity tracking** | N:M entre entidades con tipos y alias |
| **Fact relations** | compatible, conflicts_with, supersedes |
| **Session timeline** | Contexto cronológico por sesión |
| **Web viewer** | SPA en puerto :9120 |
| **Trust scoring** | Puntuación de confianza que se refuerza con retrievals |
| **Topic upsert** | Hechos que evolucionan: mismo topic_key, se actualizan |

### v1.0

| Feature | Descripción |
|---|---|
| **MCP Server** | 6 tools (add, search, get, delete, timeline, similar) vía stdio |
| **Structured facts** | Campos what/why/where/learned para memorias disciplinadas |
| **Project detection** | Detecta el proyecto desde git remote automáticamente |
| **Embedding providers** | Pluggable: NoopProvider, FastembedProvider, OllamaProvider |
| **Search expanded** | FTS5 con expansión progresiva (full query → OR → single terms) |
| **HRR multi-query** | Búsqueda paralela con variaciones semánticas de la query |
| **Dynamic RRF** | k adaptativo según cantidad de resultados |
| **Fallback chain** | Modo "auto" que cascada FTS5 → HRR → embeddings |
| **SHA-256 dedup** | Deduplicación exacta con ventana de 60s |
| **Conflict surfacing** | Detecta hechos similares al insertar y muestra conflictos |
| **Circuit breaker** | Protege contra fallos en cadena de LLM externos (3 fallos, 60s cooldown) |
| **Auto-eviction** | Elimina facts stale (trust < 0.1 o 30 días sin retrieve) |
| **Session summaries** | Genera resúmenes estructurados de sesiones |
| **Progressive disclosure** | Search devuelve resumen (200 chars), get_fact_full() da el contenido completo |

---

## Embedding Providers

Tres modos de búsqueda semántica, plug and play:

```python
# 1. Sin embeddings (FTS5 + HRR, cero overhead)
store = EtchStore("memory.db")  # NoopProvider por defecto

# 2. Con fastembed (local, ONNX, sin API key)
#    pip install memory-etch[embeddings]
from memory_etch.embedding import FastembedProvider
store = EtchStore("memory.db", embedding_provider=FastembedProvider())

# 3. Con Ollama (si ya tenés Ollama corriendo)
from memory_etch.embedding import OllamaProvider
store = EtchStore("memory.db", embedding_provider=OllamaProvider(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
))
```

Cada provider se puede usar en cualquier combinación con el MCP server.

---

## MCP Server

Para integrar memory-etch con cualquier agente que soporte MCP (Claude Code, Codex, Gemini CLI, etc.):

```bash
pip install "memory-etch[mcp]"

# Con variable de entorno
set MEMORY_ETCH_DB_PATH=./memory.db
python -m memory_etch.mcp
```

**Tools disponibles:**

| Tool | Descripción |
|---|---|
| `add_fact` | Guarda un hecho con contenido, proyecto, y metadatos opcionales |
| `search_facts` | Búsqueda híbrida con FTS5 + HRR + mode="auto" |
| `get_fact` | Obtiene un hecho completo por ID |
| `delete_fact` | Soft-delete de un hecho |
| `get_timeline` | Timeline cronológico de una sesión o proyecto |
| `similar_facts` | Encuentra hechos similares por contenido |

Configuración vía `MEMORY_ETCH_DB_PATH` (default: `memory.db` en el CWD).

---

## Benchmark

Benchmark integrado para medir recall@k con dataset sintético y juez Gemini:

```bash
# Requiere GEMINI_API_KEY
export GEMINI_API_KEY="..."

# Benchmark memory-etch (FTS5 + HRR)
python -m memory_etch.benchmark --verbose

# Benchmark contra baseline JSON (para comparar)
python -m memory_etch.benchmark --provider json-baseline --verbose

# Personalizar dataset
python -m memory_etch.benchmark --n-docs 500 --seed 42 --output results.json
```

Para benchmarkear OTRO sistema de memoria contra el mismo benchmark,
implementá ``MemoryProvider``:

```python
from memory_etch.benchmark import MemoryProvider, BenchmarkRunner

class MyMemory(MemoryProvider):
    name = "mi-sistema"
    def ingest(self, documents): ...
    def retrieve(self, query, k=10, user_id=None): ...

runner = BenchmarkRunner(MyMemory())
results = runner.run(verbose=True)
print(f"Accuracy: {results['accuracy']:.1%}")
```

Resultado de referencia (100 docs, 18 queries):

| Provider | Accuracy | Avg retrieve |
|---|---|---|
| memory-etch (FTS5 + HRR) | **94.4%** (17/18) | **5.2ms** |
| JSON baseline (word overlap) | ~40% | ~0.1ms |

---

## Web Viewer
<p align="center">
<img width="1080" height="1080" alt="Diseño sin título (3)" src="https://github.com/user-attachments/assets/297c461c-b7dc-4fe3-9ace-aed647b774ca" />
<p/>
Visualizá toda la memoria de tu agente en un SPA local, sin servidores, sin config.

```bash
python -m memory_etch.viewer --db ./memory.db
# http://127.0.0.1:9120
```

**Qué ves:**

| Feature | Para qué sirve |
|---|---|
| **Buscador** | Buscá facts por contenido, proyecto, o categoría |
| **Timeline** | Cronología por sesión — qué pasó y cuándo |
| **Relaciones** | Facts conectados: compatible, conflicts_with, supersedes |
| **Metadata** | trust_score, retrieval_count, categoría, proyecto |
| **Soft delete** | Facts archivados no se pierden, se ocultan |

**Combinado con la DB versionable:**

```bash
# Compartí la misma memoria con tu equipo
git add memory.db
git commit -m "seed data: 500 facts de referencia"
git push

# Otro dev hace pull y abre el viewer
git pull
python -m memory_etch.viewer --db memory.db
# → ve exactamente los mismos facts, relaciones, timeline
```

Útil para debuggear el estado de un agente, revisar qué facts acumuló, o compartir datasets de prueba con el equipo.

---

## Benchmarks

### Benchmark sintético (100 documentos, 18 queries)

| Modo | Recall | Latencia | Dependencias |
|---|---|---|---|
| FTS5 + HRR (search_expanded + re-score) | **94.4%** (17/18) | **5.2ms** | numpy |
| Solo FTS5 raw | ~5% | ~0.05ms | stdlib |
| Con embeddings (BGE-small) | ~72% | ~185ms | fastembed + 65MB |

Benchmark reproducible:

```bash
set GEMINI_API_KEY=...
pip install "memory-etch[hrr]"
python scripts/run_amb_benchmark.py --n-docs 100 --verbose
```

### Benchmarks en producción (VPS con facts reales de agente)

| Métrica | FTS5 solo | FTS5 + HRR | Embeddings densos |
|---|---|---|---|
| Coverage @100 facts | 39.2% | **69.7%** | 72% |
| Latencia por query | ~0.05ms | **~0.8ms** | ~185ms |
| Dependencias extra | ninguna | numpy | fastembed + ONNX |

HRR es 200-400x más rápido que embeddings densos con ~97% de su cobertura.

---

## API

Documentación detallada en [`docs/api/`](docs/api/):

- **[EtchStore](docs/api/store.md)** — Core SQLite: CRUD, FTS5, HRR, sesiones, relaciones, consolidación.
- **[EtchRetriever](docs/api/retrieval.md)** — Búsqueda híbrida: FTS5 + HRR + Jaccard + embeddings con RRF.
- **[QueryClassifier](docs/api/classifier.md)** — Clasificador rule-based para rutear estrategias de búsqueda.

---

## Proyectos relacionados

| Proyecto | Diferenciador |
|---|---|
| **memory-etch** | Local-first, KISS, SQLite, sin runtime externo, HRR vectors |
| **CodeGraph** | Code intelligence (tree-sitter + grafo de símbolos), NO es memoria de agente |
| **AgentMemory** | Memoria full-featured con iii-engine dedicado, más features, más complejidad |
| **Engram** | Memoria para agentes Go/MCP, sin embeddings, curada por el agente |

---

## Contribuir

```bash
git clone https://github.com/Basiliskode/memory-etch
cd memory-etch
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Todos los PRs son bienvenidos. Usamos conventional commits y TDD estricto.

---

## Licencia

MIT. Construí algo útil.

---

> Memory Etch nació dentro de un agente AI real que necesitaba acordarse de las cosas sin depender de servicios externos. Hoy corre en producción y está probado con miles de facts.
>
> Si estás construyendo un agente que necesite memoria, probalo. Son 30 segundos.
>
> ```bash
> pip install "memory-etch[hrr]"
> python -c "from memory_etch import EtchStore; s = EtchStore('test.db'); print('anda')"
> ```
