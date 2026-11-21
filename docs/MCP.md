# Use the memory from any agent (MCP server and Claude Code plugin)

`mcp_server.py` exposes the tiered memory as MCP tools. The repository is also a Claude
Code plugin: installing it starts the server and adds a `tiered-memory` skill that tells
the agent when to recall and when to save.

| Tool | What it does |
| --- | --- |
| `recall_context(query)` | Fast per-turn context: long-term facts plus this session's recent turns |
| `search_memory(query, time_range?, include_episodes?)` | Deeper search, optionally date-bounded |
| `get_episode(episode_id)` | The original conversation round behind a memory |
| `get_supporting(observation_id)` | The facts behind an observation |
| `remember_turn(user_text, assistant_text, ...)` | Retain one exchange (synchronous; a few seconds) |
| `finish_session()` | Build observations at the end of a conversation |

User, project and session are never tool arguments. The server reads them from the
environment, so the model can't read or write another user's memory.

## Prerequisites

The server uses the same backend as the rest of the repository: AlloyDB with schemas
004–006 applied, Memorystore for Valkey, and Vertex AI. Follow [SETUP.md](SETUP.md) and
[sql/README.md](../sql/README.md) first. The server must run somewhere that can reach the
private endpoints (the codelab's VM, or a host on that VPC). You also need
[uv](https://docs.astral.sh/uv/) and Application Default Credentials
(`gcloud auth application-default login`).

## Configure

Export these in the shell that starts your agent; the server inherits them.

```bash
# Whose memory this is (required), and optional project visibility
export MEMORY_USER_ID="alice"
export MEMORY_PROJECT_ID="my-project"    # optional
# export MEMORY_SESSION_ID="..."         # optional; default is a new id per agent session

# Backend, as in .env.example
export PROJECT_ID="<gcp-project>" REGION="us-east1" GENAI_LOCATION="global"
export DB_HOST="<alloydb-private-ip>" DB_USER="<iam-db-user>" DB_NAME="postgres"
export VALKEY_HOST="<valkey-ip>"
# export LLM_MODEL="gemini-4-hydrogen"  MEMORY_LOG_LEVEL="INFO"
```

The server starts without these and lists its tools; a tool call then returns an error
naming the missing setting.

Gemini is the default for the server's own generation. To use Claude or OpenAI, set
`LLM_PROVIDER`, `LLM_MODEL`, and the corresponding API key as described in
[LLM generation](CANDIDATE.md#llm-generation). Install the matching extra in the
checkout used by the server, for example `uv sync --extra mcp --extra anthropic`.
AlloyDB, Valkey, and Google embeddings are still required.

## Install in Claude Code

```
/plugin marketplace add HenryHamster/alloydb-agentic-tiered-memory
/plugin install tiered-memory@alloydb-agentic-tiered-memory
```

On Claude Code 2.1.275 or later, one line does both:
`/plugin install tiered-memory --marketplace HenryHamster/alloydb-agentic-tiered-memory`.

Restart Claude Code. `/mcp` should list `tiered-memory`; the skill loads when a request
involves past conversations or "remember…".

To try a local checkout instead: `/plugin marketplace add /path/to/this/repo`.

## Other MCP clients

Any client that runs stdio servers can use the same command:

```json
{
  "mcpServers": {
    "tiered-memory": {
      "command": "uv",
      "args": ["run", "--quiet", "--directory", "/path/to/repo", "--extra", "mcp",
               "python", "-m", "mcp_server"],
      "env": { "MEMORY_USER_ID": "alice" }
    }
  }
}
```

For Claude or OpenAI, add `"--extra", "anthropic"` or `"--extra", "openai"` before
`"python"` in the args above. Without uv, use `pip install -e '.[mcp]'` (or
`'.[mcp,anthropic]'` or `'.[mcp,openai]'`) and run `tiered-memory-mcp`.

Copy the guidance in [skills/tiered-memory/SKILL.md](../skills/tiered-memory/SKILL.md) into
your agent's system prompt if it doesn't support skills.

## Notes

- One server process is one memory session. Calls are serialized, as `TieredMemory` requires.
- `remember_turn` commits before returning; Reflect stays off, as in `build_memory_system`.
- Strict mode is on: if Valkey or AlloyDB is unreachable, tools fail rather than answering
  from partial memory.
- `search_memory` results leave out the per-stage `trace` to save the agent's context.
