# memsteward - an autonomous agent that keeps a living map of a codebase

A small demo **agent** that consumes Jennah's public memory APIs the way any
external agent would: plain HTTP/JSON through the `jennah-proxy` gateway,
authenticated with a `jennah_sk_` API key. No Jennah server internals are
imported - this is a standalone Go module, so it doubles as a reference for
outside integrators. It's a sibling to
[`jennah-memchat`](../jennah-memchat).

Where **memchat** is *reactive* (a human speaks, it recalls and replies),
**memsteward** runs *unattended* over a source tree and stresses the parts of
Jennah a long-running agent leans on: the **execution log as durable state**,
the **graph as structural knowledge**, and **incremental work driven by what it
remembers**.

## What it does each run

```
memsteward -repo /path/to/repo

  walk the tree ──► memory:query (log) ── recall each file's last content hash
                          │
      per file: ──────────┤ NEW / MODIFIED ─► brain analyzes the file, then
                          │                    memory:commit (atomic):
                          │                      • log   ── hash + summary  (per-file state)
                          │                      • vector ── file summary   (semantic recall)
                          │                      • graph ── (repo)-[CONTAINS]->(file)
                          │                                 (file)-[DEFINES]->(symbol)
                          │                                 (file)-[IMPORTS]->(dep)
                          │                                 (file)-[NOTES]->(decision)
                          │ UNCHANGED ─────────► skip (no LLM call)
                          └ REMOVED (in log, gone on disk) ─► memory:commit a drift log step
```

The first run analyzes everything. **Re-run it after editing a few files** and
it re-analyzes only those - because it read every file's last content hash back
out of the execution log first. Delete a file and re-run, and it reports the
**drift**. That incremental behaviour and drift detection come entirely from
remembered state; the agent keeps no local cache beyond the agent id.

### Why the log *and* the graph

The split is deliberate and shows off both memory types:

- **Execution log = versioned per-file state.** Each analyzed file writes a log
  step carrying its content hash. A log query returns every field of every
  step (newest first), so the steward reconstructs "what did each file look
  like last time?" - the basis for the NEW/MODIFIED/UNCHANGED/REMOVED diff.
- **Graph = structure.** Files, the symbols they define, the dependencies they
  import, and notable design decisions become nodes and edges you can traverse.
  (Graph traversal rows don't reliably project node *properties*, which is
  exactly why the hash lives in the log, not on the file node.)
- **Vector chunks** hold each file's summary for semantic recall.

Cross-run memory is just **reusing the same `agent_instance_id`**, persisted to
`memsteward-state.json`. Node/edge/log ids are deterministic content hashes, so
re-running converges instead of fragmenting. Delete the state file to steward a
fresh workspace.

## Prerequisites

1. A Jennah API key for an **approved, entitled** enterprise. Mint one after
   logging in (console or `jnh`):
   `POST /v1/apikeys {"label":"memsteward"}` → copy the `secret` (shown once).
2. An analysis model - Anthropic, or Gemini (via **Google AI Studio** with an
   API key, or via **Vertex AI** with a GCP project + ADC).

The analysis brain is pluggable: only the LLM differs, every Jennah memory call
is identical. `-provider auto` (the default) picks **Anthropic** when an
Anthropic key is configured, otherwise **Gemini**; force it with
`-provider gemini|anthropic`.

The agent's home region is chosen at creation with `-region` (or
`$JENNAH_REGION`); it's applied only on first launch, since an agent is pinned
to one region for its lifetime, and empty uses the platform default. List the
available regions with `jnh agents regions`. The target region must have managed
embeddings configured (prod `db0001` / `us-central1` does) - the demo sends
plain text and lets the server embed each file summary.

## Run

```sh
export JENNAH_API_KEY=jennah_sk_...

# Anthropic:
export ANTHROPIC_API_KEY=sk-ant-...
go run . -repo /path/to/repo                    # auto-selects Anthropic

# …or Gemini via Google AI Studio (API key):
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
go run . -repo /path/to/repo

# …or Gemini via Vertex AI (GCP project + ADC, no API key):
gcloud auth application-default login           # once
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1        # optional; defaults to "global"
go run . -repo /path/to/repo

go run . -provider gemini -repo .   # force a provider regardless of which keys are set
go run . -show                      # print the remembered graph and exit (no analysis)
go run . -verbose -repo .           # show per-file NEW/MODIFIED/unchanged + commit receipts
go run . -ext .go,.py,.ts -repo .   # analyze more than Go
go run . -endpoint http://127.0.0.1:8090   # against a local proxy instead
go run . -region us-central1               # pin the agent's home region (or $JENNAH_REGION)

# …or pass the keys as flags instead of env vars:
go run . -jennah-api-key jennah_sk_... -anthropic-api-key sk-ant-... -repo .
```

On start it prints the chosen brain, e.g.
`analysis model: anthropic/claude-sonnet-5`.

Try: point it at a small repo, let it finish, then edit one file and run it
again - it says `1 modified, N unchanged (skipped)`. Then `-show` to read back
the graph it built.

## Flags

| flag | default | meaning |
|------|---------|---------|
| `-repo` | `.` | source tree to steward |
| `-ext` | `.go` | comma-separated extensions to analyze |
| `-max-files` | `40` | cap on files considered per run (0 = no limit) |
| `-show` | `false` | print the remembered graph and exit |
| `-provider` | `auto` | `auto\|gemini\|anthropic` |
| `-region` | `$JENNAH_REGION` | home region, applied only at creation |
| `-endpoint` | `https://jennah.alphaus.cloud` | proxy origin |
| `-state` | `memsteward-state.json` | local state file (agent id) |
| `-verbose` | `false` | per-file classification + commit receipts |

## Notes

- Each provider defaults to a snappy/cheap model (`claude-sonnet-5`,
  `gemini-2.5-flash`); edit `anthropicModel` in `brain_anthropic.go`
  (→ `anthropic.ModelClaudeOpus4_8`) or `geminiModel` in `brain_gemini.go`
  (→ `gemini-2.5-pro`) for max capability. Backends live behind the `brain`
  interface in `brain.go`.
- The model is **forced** to answer through the `analyze_file` tool (Anthropic
  `tool_choice`, Gemini function-calling mode `ANY`), so every file comes back
  as one structured payload - no free-text parsing.
- File contents are truncated to ~8000 chars before analysis to keep calls
  cheap; large files are summarized from their head.
- Removed files are recorded as drift in the log, never deleted from the graph
  (graph writes are insert-only). A file that reappears is picked up as new.
- Fusion (`link:true`) is intentionally not used - it returns Unimplemented.
