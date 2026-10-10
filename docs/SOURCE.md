# Source provenance

Upstream: [How to build long-term AI agent memory with AlloyDB AI](https://codelabs.developers.google.com/alloydb-agentic-tiered-memory),
Google Codelabs. Retrieved October 10, 2026.

The original code blocks were extracted from the page's HTML with a standard
HTML parser, decoding character references and removing syntax-highlighting tags.
All 12 Python modules and the three SQL blocks parsed/extracted successfully.
Original decoded-block SHA-256 hashes, source steps, and the HTML hash are in
[`source_manifest.json`](source_manifest.json).

The first Git commit preserves the extracted Python and SQL before repository
cleanup. Subsequent changes format Python, sort/remove unused imports, remove
the ADK demo's upstream project fallback, and add packaging, documentation,
configuration examples, tests, and CI. The codelab's model names and memory
algorithm are retained. Compare Git commits for the exact adaptation diff.

Python filenames match the codelab. The SQL blocks are named
`001_schema.sql`, `002_register_model.sql`, and `003_auto_embeddings.sql`.
Cloud setup and teardown command samples are organized in `SETUP.md`.

The page states that its code samples are Apache 2.0 licensed. Its prose is
licensed separately under CC BY 4.0; the repository's documentation is newly
written and links to the tutorial for explanatory material. See `LICENSE` and
`NOTICE` at the repository root.
