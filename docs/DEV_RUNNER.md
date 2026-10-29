# Developer question runner

Use a local CLI to run the **current checkout** on the team's existing VM. This includes tracked edits and unignored Python/SQL files. History, questions, code hashes, raw provider receipts, prepared memory, retrieved context, answers, latency and token usage are saved for each run. This runs the integrated Retain/Reflect/Recall implementation; the original codelab benchmark remains in `bench/`.

## Developer quick start

Install Git, Python 3.11+, uv and the Google Cloud CLI; ensure `git` and `gcloud` are on PATH. An administrator must first complete the VM setup below.

```bash
uv sync --locked --extra dev
gcloud auth login
cp dev.config.example.json dev.config.json
uv run python dev.py status
uv run python dev.py start
uv run python dev.py run --history fixtures/dev/history.json --questions fixtures/dev/questions.json --question-id chosen_movie --name movie-v1 --budget 2
uv run python dev.py stop
```

`start` waits for IAP, the installed shutdown controller, and AlloyDB readiness (allow about seven minutes). `run` requires a ready environment; it does not start it implicitly. Run one question with `--question-id ID`, all questions by omitting that flag, or ad hoc text with `--question 'Which movie did we pick?'` instead of `--questions`. PowerShell uses `Copy-Item` instead of `cp` if preferred.

`--budget` is a **new API allowance for the entire invocation**, including Retain, Reflect, Recall, answers and retries. Default generation limits match the production maximum (65,536 output/thinking tokens); a small budget can refuse a single call's conservative reservation. Explicitly choose `--max-output-tokens 4096` for cheaper, shorter development responses. This changes model configuration and therefore its cache key.

## Input and experiment versions

Copy [the history](../fixtures/dev/history.json) and [questions](../fixtures/dev/questions.json) into your private experiment directory. JSON must be UTF-8. Keep sensitive histories outside Git. The runner does not truncate history.

```json
{"id":"film-club","sessions":[{"id":"planning","messages":[
  {"role":"user","content":"We picked Inception. Its IMDb ID is tt1375666."},
  {"role":"assistant","content":"Got it."}
]}]}
```

History IDs, session IDs and question IDs must be unique within their respective input and contain 1–80 letters, digits, dots, underscores or hyphens, starting with a letter or digit. Messages support `user`, `assistant`, `tool`, `system`, and JSON content/tool_calls. Optional `timestamp` must include a timezone, e.g. `2026-10-28T12:00:00Z`. Without dates the runner uses a reproducible synthetic execution clock and marks source dates unknown. It widens extracted session-visible facts to the fixture's project namespace so cross-session questions can use them, as the Mem2Act harness does.

```json
{"questions":[
  {"id":"movie","question":"Which movie did we pick?","expected_answer":"Inception"},
  {"id":"reviews","question":"Get reviews for our movie.",
   "tools":[{"name":"Get Movie Reviews","parameters":{"type":"object","properties":{"movie_id":{"type":"integer"}},"required":["movie_id"]}}],
   "expected_tool_call":{"name":"Get Movie Reviews","arguments":{"movie_id":1375666}}}
]}
```

Expected labels are optional and stay local. Scoring is strict text/JSON equality, not the official benchmark judge. Tool questions request JSON tool proposals; external tools are not executed.

For changed history, save `history-v2.json` and pass that path. Keep multiple versions as ordinary files or Git commits. Each invocation has a generated run ID even if `--name` repeats; previous results remain intact. No database migrations or manual cache resets are needed for a new fixture.

## Cache and isolation

- `--mode cached` (default): reuse an unchanged prepared snapshot, then exact provider requests. Changed preparation code or history rebuilds memory; unchanged provider requests within that rebuild remain reusable.
- `--mode replay`: require recorded provider responses; a missing request stops with `cache_miss`, without a paid fallback. It still needs the VM, PostgreSQL and Valkey.
- `--mode fresh`: execute preparation and provider requests again. Existing cache baselines are preserved. Use a new `--cache-epoch experiment-v2` if you want a separate reusable baseline.

Prepared keys include Retain/Reflect/shared/orchestration/SQL code, runner/harness code, history, provider project/locations, output limit and cache epoch. Recall-only edits can reuse prepared memory. Provider keys include the complete request and configuration. Different developers have separate OS-user caches; there is no shared response pool across users.

Each preparation and **each question** uses a new random database schema and a copy of prepared memory, including lifecycle/access state. Query results are not retained into history. Valkey keys use a run prefix. Schemas and touched Valkey keys are removed on normal completion/failure. Hard termination can leave a schema; an administrator can inspect/drop the specifically recorded `dev_<UUID>` schema later. These namespaces prevent accidental experiment contamination; they are not security boundaries for arbitrary developer code.

## Results and costs

Local results are in `dev_runs/<name>-<UUID>/report.json`; the nested UUID directory contains raw worker output. `usage.json` separates cache hits, generation token counts (prompt, visible output, thinking), embeddings/ranking reserves and per-stage call latency. Answers include their retrieved context and exact answer prompt. Source manifests hash all uploaded code, including uncommitted changes. Keep artifacts private; they contain history and prompts.

API usage is estimated from a pinned rate card for `gemini-3.5-flash`, `text-embedding-005` and the current ranker. The runner reserves before each request and retains uncertain-completion reserves. It refuses unsupported generation/embedding models. Update the rate card and tests when changing models or pricing. SDK automatic retries are disabled; adapter retries are metered separately. A budget stop/cache miss returns exit code 2 and exports partial results. This is an application guard for this runner, not a GCP billing cap or a sandbox against modified code.

Infrastructure is separate from `--budget`. Planning rates for the existing us-east1 resources, before credits (28 October 2026):

| Component | Estimate |
|---|---:|
| e2-standard-2 VM | $0.06701/hour while running |
| AlloyDB 2-vCPU primary | $0.31136/hour while running |
| Existing Valkey nano | $0.7632/day, continues while VM/DB are stopped |
| 20 GiB VM standard disk | $0.80/month |
| AlloyDB storage, backups, public IPv4 and traffic | Additional usage charges |

At two active hours/day for 30 days, VM + DB + existing Valkey + VM disk are about **$46.40/month**, plus models, storage/backups/network. Allow roughly $50/month plus models at that usage. Startup and idle time count toward active hours. The report includes run-duration infrastructure share estimates; overlapping runs are not additive billing, and startup/idle time is excluded from that share. See [Compute pricing](https://cloud.google.com/compute/all-pricing), [AlloyDB pricing](https://cloud.google.com/alloydb/pricing), [Memorystore pricing](https://cloud.google.com/memorystore/valkey/pricing), and [Vertex pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing).

The root-owned controller checks every 30 seconds, stops AlloyDB then the VM after 30 idle minutes, and refuses manual shutdown while jobs are active. Its four-hour boot deadline can terminate a long-running job; the VM has a separate provider stop deadline ten minutes later. Idle time starts after the last process finishes, including dependency setup. A failed DB stop request keeps the VM watchdog alive to retry. A complete VM/controller failure can still leave AlloyDB running: use `dev stop` to reconcile it. A strict shutdown guarantee across VM failures would require an external watchdog, outside this small design.

Caches/source/result files stay on each developer's VM home disk. Keep valuable result exports locally; delete old per-run work/results or unused cache epochs explicitly when no jobs are using them. There is no automatic deletion of experiment evidence.

## One-time administrator setup (review before applying)

This PR does not deploy or grant access. Reuse the existing private VM/AlloyDB/Valkey; no new managed database instance is needed. Set a billing alert separately if desired.

1. On the VM, install Python **3.11.8+** (tar extraction filters), Google Cloud CLI and uv at `/usr/local/bin/uv`. Ensure the VM's attached service account has the `cloud-platform` OAuth scope and metadata-service access. The VM must reach both private endpoints and Vertex/package registries. Follow [OS Login](https://docs.cloud.google.com/compute/docs/oslogin/set-up-oslogin) and [IAP forwarding](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding): enable OS Login, allow SSH from `35.235.240.0/20`, and grant named developers `roles/compute.osLogin` (not osAdminLogin), `roles/iap.tunnelResourceAccessor`, and `roles/iam.serviceAccountUser` on this attached service account. External-organization users may additionally need OS Login External User. Scope grants to this VM/identity where supported. Do not give developers sudo just to run questions.
2. Grant developers a custom lifecycle role containing `compute.instances.get`, `compute.instances.start`, `compute.instances.setScheduling`, `compute.projects.get`, `compute.zoneOperations.get`, `alloydb.instances.get`, `alloydb.instances.update`, `alloydb.operations.get`, `resourcemanager.projects.get`, and service usage consumer access. Scope to this development project's resources using supported IAM bindings/conditions; do not reuse this identity for unrelated production databases. `update` permissions can change more than activation policy, so this is a trusted developer role.
3. Attached VM service account needs Vertex AI User, AlloyDB Database User, Service Usage Consumer, plus a narrowly scoped custom shutdown role with `alloydb.instances.update`, `alloydb.instances.get`, `alloydb.operations.get`, `compute.instances.stop`, `compute.instances.get`, `compute.zoneOperations.get`. The controller uses that service account, never an administrator's cached gcloud login. The AlloyDB service agent also needs the existing Vertex/ranking permissions from the codelab setup.
4. Configure [AlloyDB IAM database authentication](https://docs.cloud.google.com/alloydb/docs/database-users/manage-iam-auth) for the attached service account. As DB administrator create a separate **`developer_runner` database**, not `candidate_benchmark` or the original codelab database. Connect to it as administrator, install `vector`, `pg_trgm`, and `google_ml_integration CASCADE`; configure AlloyDB AI/preview/ranking prerequisites from the existing setup. Grant that IAM DB user CONNECT/CREATE on this database, USAGE on public and the AI extension schemas, and the required ranking function execution permission. Do not grant table ownership or write access to benchmark databases. Workers create tables with migrations 004/005 inside their own schemas; do **not** apply the original auto-embedding trigger to these tables. Retain supplies embeddings explicitly.
5. Copy this checkout to the VM. Fill `deploy/dev-vm.example.json` into a private `dev-vm.json` with the IAM database username and private addresses. There are **no passwords or keys** in this file. Run:

   ```bash
   sudo bash deploy/install-dev-vm.sh /absolute/path/to/dev-vm.json
   sudo systemctl status tiered-dev-idle.timer
   python3 /opt/tiered-dev/control.py status
   ```

   Install/upgrade only with no active jobs. Install on a freshly started VM: a boot older than the configured four-hour deadline is stopped immediately. The installer preserves the lifecycle lock inode and installs root-owned controller/config files. Its timer begins immediately; do not use this VM for unrelated always-on work. The idle/runtime values in developers' local config must match the installed values.
6. Verify an ordinary developer can `dev status`, start, run the tiny fixture with an approved API allowance, and stop. This live IAP/IAM/Vertex smoke is the deployment acceptance step; offline tests do not establish those project-specific permissions.

Developers authenticate personally with `gcloud auth login` locally. Do not upload ADC files, service account keys, `.env`, or access tokens. Uploaded source is Python/SQL plus pyproject/lockfile, along with explicitly supplied fixture data. Workers explicitly use Compute Engine credentials for DB/Vertex; local user ADC is not consulted. All trusted users who can execute code on this VM can exercise its attached service identity. Use a separate project/VM if that trust is inappropriate.

## Maintainer checks

```bash
uv sync --locked --extra adk --extra dev
uv run --extra adk --extra dev ruff check .
uv run --extra adk --extra dev ruff format --check .
uv run --extra adk --extra dev pytest
```

Optional SQL integration: point `DEV_TEST_DSN` at a **disposable PostgreSQL 16 + pgvector** database and run `pytest tests/test_dev_postgres.py`. These tests install a deterministic SQL ranker fixture and use scripted model responses: real migrations, SQL stores, vectors, ranges, cyclic foreign keys, snapshots and pipeline wiring; no paid API calls. Do not point them at a real benchmark database. The cloud acceptance smoke remains separate.
