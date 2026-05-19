# Memory Etch: ~0.8ms por búsqueda. Sin GPU, sin servicios, sin excusas.

**Memoria persistente para agentes AI.** SQLite + FTS5 + HRR vectors.
Cero dependencias obligatorias. Cero llamadas externas. 77 tests que pasan siempre.

```bash
pip install "memory-etch[hrr]"
```

---

## Nadie construye un agente serio sin memoria.

Pero las opciones hasta ahora eran: mandar todo a una API externa, instalar una base vectorial que pesa 2GB, o escribir un JSON que crece como plaga.

Memory Etch usa SQLite. Corre donde corre Python. No necesitás GPU, no necesitás API key, no necesitás un tutorial de 40 minutos.

**Es un archivo.** Lo movés, lo copiás, lo commiteás. Abrís el viewer en `:9120` y ves todo lo que tu agente recuerda.

```python
from memory_etch import EtchStore, EtchRetriever

store = EtchStore("memory.db")
store.add_fact("FastAPI es un framework web", category="tech")
store.add_fact("SQLite es un motor de base de datos", category="tech")

retriever = EtchRetriever(store)
results = retriever.search("motor de base de datos")
for r in results:
    print(f"[{r['_score']:.2f}] {r['content']}")
```

Eso es todo lo que necesitás para arrancar.

---

## Lo que hace

- **FTS5** — búsqueda de texto completo con triggers que sincronizan solos. No indexás dos veces, no se te desyncroniza.
- **HRR vectors** — representaciones holográficas. Sin PyTorch, sin GPU, sin 2GB de modelos. Si tenés numpy funciona, si no, degrada limpio.
- **Jaccard re-rank** — overlap de n-gramas para ordenar resultados. Barato, rápido, sin llamadas externas.
- **Soft delete** — los hechos no se borran, se ocultan. Por si después necesitás ese dato que creías que no.
- **Consolidación activa** — cuando dos hechos chocan, un LLM decide: ¿actualizar? ¿fusionar? ¿ignorar? Sin ruido falso.
- **Entity tracking** — N:M entre entidades, con tipos, alias, la posta.
- **Fact relations** — compatible, conflicts_with, supersedes. Tu agente puede saber que dos cosas se contradicen.
- **Session timeline** — contexto cronológico por sesión. Sabés qué pasó antes y después de cada hecho.
- **Web viewer** — SPA en `:9120`. Diseño mint, sin bulla. Clickeás un fact y ves relaciones, timeline, metadata.

Cero de estas features necesita una API key.

---

## Benchmarks reales

No specs inventadas. Esto es corriendo en una VPS común, con facts reales de un agente en producción.

| Métrica | FTS5 solo | FTS5 + HRR | Embeddings densos |
|---------|-----------|------------|-------------------|
| Coverage @100 facts | 39.2% | **69.7%** | 72% |
| Latencia por query | ~0.05ms | **~0.8ms** | ~185ms |
| Dependencias extra | ninguna | numpy | torch + fastembed + 2GB |

200 a 400 veces más rápido que embeddings densos. Misma cobertura. Cero modelos que descargar.

Si querés ver los números vos mismo:

```bash
git clone https://github.com/Basiliskode/memory-etch
cd memory-etch
pip install -e ".[hrr]"
python scripts/benchmark.py
```

---

## Instalación

```bash
pip install "memory-etch[hrr]"      # recomendado: FTS5 + HRR
pip install memory-etch              # mínimo: solo FTS5 + Jaccard
pip install "memory-etch[embedding]" # con fastembed para embeddings locales
pip install "memory-etch[all]"       # todo
```

## Viewer

```bash
python -m memory_etch.viewer --db ./memory.db
# http://127.0.0.1:9120
```

## Configuración

La DB vive en `~/.etch/memory.db`. La podés overridear con `MEMORY_ETCH_DB` o `--db`.

Si querés sabés exactamente qué pasó, el viewer te muestra todo. Si querés automatizar, la API es SQLite plano — podés consultar con cualquier cliente SQLite.

---

Memory Etch nació dentro de un agente AI real que necesitaba acordarse de las cosas sin depender de servicios externos. Hoy es el backend de memoria de Hermes Agent, corre en producción, y está probado con miles de facts.

Si estás construyendo un agente que necesite memoria, probalo. Son 30 segundos:

```bash
pip install "memory-etch[hrr]"
python -c "from memory_etch import EtchStore; s = EtchStore('test.db'); print('anda')"
```

Después me contás.

MIT.
