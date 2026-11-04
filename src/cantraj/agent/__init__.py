"""Tool-using LLM agent study (supplementary reference baseline, ICDE paper Section VI).

Two layers with different import footprints:
- `runner` (graph + reconstructors; runs in the tool-cache process where `io.load_gps` is denied) computes every
  candidate, its observable diagnostics and the junction tables once, and writes them to a per-window cache;
- `toolbox` (read-only cache, no graph, no reference) is what the LLM sees; it enforces which tools were called and
  which candidates are visible. `selector` is the supervised selector (B), `llm` the OpenAI-compatible client,
  `loop` the budgeted control loop (C).
"""
