# AlloyDB initialization

Open AlloyDB Studio for your `postgres` database and authenticate as the IAM
database user described in the setup guide.

1. Run `001_schema.sql` to create the extensions, memory tables, and indexes.
2. Edit `002_register_model.sql`, replacing `PROJECT_ID` with your Google Cloud
   project ID, then run it as a separate execution block.
3. Run `003_auto_embeddings.sql` by itself, outside a transaction shared with
   any other statement. `ai.initialize_embeddings` manages its own transaction.

These files preserve the codelab SQL. They require AlloyDB AI, the
`google_ml_integration`, `vector`, and `rum` extensions, the AI query engine
flag, preview AI functions, and Vertex AI permissions on the AlloyDB service
agent. They do not target a generic local PostgreSQL installation.

Embeddings use `text-embedding-005` with 768 dimensions. Keep the table dimension
and all embedding call sites consistent if you adapt the model. The registered
LLM is `gemini-3.5-flash`, matching the tutorial's Python examples.
