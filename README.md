# AlloyDB agentic tiered memory

Python and SQL source extracted from Google's [How to build long-term AI agent memory with AlloyDB AI](https://codelabs.developers.google.com/alloydb-agentic-tiered-memory) codelab, organized as an installable repository. The original filenames and memory algorithm are retained so you can follow the tutorial directly.

The agent combines recent turns and rolling summaries in Memorystore for Valkey with durable entities and episodic memories in AlloyDB. It extracts facts in a background thread, retrieves them with hybrid search, and optionally integrates with Google ADK.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Cloud execution also requires the Google Cloud CLI, AlloyDB, Memorystore for Valkey, Vertex AI access, and connectivity to both private database endpoints.

```bash
git clone https://github.com/HenryHamster/alloydb-agentic-tiered-memory.git
cd alloydb-agentic-tiered-memory
uv sync --extra adk --extra dev
```

Alternatively, create a virtual environment and run `python -m pip install -r requirements.txt`. Development tools are available through `python -m pip install -e '.[adk,dev]'`.

## Configure and run

1. Provision the cloud resources using [the setup guide](docs/SETUP.md). Existing shared infrastructure can be reused; authentication and environment variables are specific to each developer.
2. Copy `.env.example` to `.env` and fill in the project, endpoint, and IAM user values. Python reads exported environment variables; it does not load `.env` automatically.
3. Apply the [SQL files](sql/README.md) in order in AlloyDB Studio.
4. Authenticate and run from the VM or another host with access to the private endpoints:

```bash
gcloud auth application-default login
set -a
source .env
set +a
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="$GENAI_LOCATION"

# Codelab's five-turn, two-session memory demonstration
uv run python test_memory_system.py

# Optional Google ADK demonstration
uv run --extra adk python test_adk_agent.py
```

The two `test_*.py` files are **live integration demonstrations** that call cloud services and write sample memories. The ADK terminal tool is a mock that returns a string. The local test suite and CI run only `tests/` and do not invoke these demos.

## Repository map

| Files | Purpose |
| --- | --- |
| `db_clients.py` | IAM-authenticated AlloyDB and Valkey connections |
| `valkey_buffer.py` | Sliding window of recent turns with summarize-before-trim |
| `turn_context.py` | Per-turn context: the sliding window plus Recall under one token budget |
| `async_worker.py` | Background fact extraction and upserts |
| `retain/` | Retain pipeline: facts, entities, links, observations; see [Retain notes](docs/RETAIN_NOTES.md) |
| `reflect/` | Reflect: tiers, redundancy/conflicts, macro memories; see [Reflect notes](docs/REFLECT_NOTES.md) |
| `shared/chunking.py` | Lossless length-bounded text chunking |
| `hybrid_retriever.py` | Query decomposition and hybrid retrieval |
| `recall/` | Recall read path and its agent tools; see [Recall notes](docs/RECALL_NOTES.md) |
| `agent_orchestrator.py` | Prompt construction and agent turn loop |
| `enterprise_engine.py` | Tool permission evaluation and memory compaction |
| `adk_*.py` | ADK ambient context, archive search tool, and guardrail callback |
| `test_memory_system.py`, `test_adk_agent.py` | Original cloud demonstrations |
| `cleanup_memory_system.py` | Clears the demo memory tables and selected Valkey database |
| `sql/` | Schema, model registration, auto-embedding setup, Retain and Reflect tables |
| `docs/` | Cloud setup, source provenance, and validation limits |
| `tests/` | Offline checks of repository integration and memory behavior |

## Local checks

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra adk --extra dev pytest
```

The code is an educational sample. Read [validation and inherited limitations](docs/VALIDATION.md) for what local checks cover and what still requires a real cloud run. No published benchmark result is claimed by this repository.

For optional memory reset and cloud teardown, see [setup and cleanup](docs/SETUP.md#cleanup).

## Source and license

Source captured on October 10, 2026. Google licenses the codelab's code samples under Apache 2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), and [source provenance](docs/SOURCE.md). This is an independent extraction and is not an official Google repository.
