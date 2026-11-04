# Agent study protocol v1 (supplementary reference baseline)

Status: FROZEN at 2026-11-04 11:48:29 (local time), after the validation pilot and before any formal-cohort agent run. Nothing below changes
after this point; the formal tool cache (deterministic, no result consulted) was already being built when the protocol was frozen.
FROZEN: 2026-11-04 11:48:29

## Question
Under the same inputs, map access and underlying tool budget, does a single fixed LLM controller that chooses which of three
existing reconstructors to run and which of their candidates to return recover more routes than (A) the strongest fixed
reconstructor alone and (B) a supervised selector over the same tools?

## Protocol and inputs (identical for A, B, C)
- Benchmark v1.1, protocol F-M-Hp: offline, OpenStreetMap graph of the episode's visible disc (tiles_v2, per-episode
  visibility), start position and start heading given, signal group G3 (speed, steering-wheel angle; static wheelbase and
  steering ratio with the training-fold calibration fields). Time compensation "train", map_start rule as published.
- Neither the agent nor any tool sees reference positions, the true route, scores, other episodes, or anything outside the
  visible disc. The tool cache is built in a process where `io.load_gps` is denied (run_baseline.py's `_deny`); the cache
  holds no coordinates of the start, no reference time stamps and no score; the agent process (`run_agent.py`) imports neither
  the graph nor the data loaders. Fields that depend on a reconstructor the agent has not run are reported as unknown.

## Cohort (frozen before any agent run)
- Rule: for every drive of the test and few_shot_heldout folds, the first eligible five-minute window (smallest k) that is
  also map_ok; one window per drive. No method output consulted.
- `manifests/icde_agent_cohort.csv`, sha256 d026d236e21ab6e1f32f3c62d0fe2abad32249541f0a5e116127685608889761:
  659 windows = 659 drives, 92 drivers (test 268 windows / 42 drivers; few_shot_heldout 391 / 50). The 92 drivers are fewer than
  the 94 held-out drivers of the main board because the cohort requires an eligible, map-ok five-minute window.
- Validation cohort (selector fitting, pilot): `manifests/icde_agent_val.csv`, sha256 53cad84be699ba2e758dd675b03f8a46ef9c20f30e63e8ad665c135bd33493c1,
  610 windows / 43 drivers; pilot subset `manifests/icde_agent_pilot.csv` (first 40 windows in (driver, drive) order, at most 4 per
  driver; 13 drivers), sha256 639face588a9b4229cd632030583ae3e42d1f298074a4caec985adffe5caa977.
- The test and few-shot folds were inspected in earlier versions of this benchmark; this is not a never-seen test.

## Tools (frozen configurations = published defaults; `src/cantraj/agent/runner.py` TOOL_SPECS)
- beam: `MapBeamSearch("steer")` (B = 200, K = 10); published stem map_beam_steer__G3.
- pf: `MapParticleFilter("steer", readout="lineage")` (N = 2000, seed 0; one candidate); stem map_pf_steer_lineage__G3.
- ep: `ElasticPathing(heading=True, friction=0.15, v_zero=1.0)` (K = 10; budget clip(0.2 n, 30, 300) s, clip(70 n, 2e4, 2e5)
  branch calls); stem elastic_pathing_hp__G1.
- Identity: on the formal cohort the cached Top-1 and candidates of beam and pf are byte-identical to icde_v11; ep is identical on
  windows that did not hit its budget (verified on 3 windows before the formal run; full check in agent_identity_check.csv).
- Diagnostics (observable only; `diagnostics.py`): path length vs CAN distance, absolute heading consistency against the given
  start heading plus integrated steering, drift after removing a constant offset, turn counts and unmatched turns, U-turns,
  overlap / end distance / divergence against other candidates, junction tables from the visible graph.

## Configurations
- A: map beam Top-1 (icde_v11 output re-aggregated on the cohort).
- B: supervised selector `configs/agent/selector_v1.json` (sha256 9f451624ca6a7d0b4e5f30e5d18075509c0198778efd0ea6b828b8063a98bc95; 12 features, coefficients in the file; validation cross-fit
  route rates: selector 81.8 % [76.8, 86.5], beam-1 78.7 % [71.5, 83.9], pf-1 74.6 %, ep-1 28.0 %, pool oracle 91.0 %; 610 windows, 43 drivers): logistic regression over the diagnostics, labels = validation route_ok of every pool candidate; standardisation and
  regression fitted together inside each training fold (GroupKFold by driver, 5 folds) for the report, then fitted on all 610
  validation windows and frozen. B always runs all three tools.
- C: one Qwen3 controller through vLLM (OpenAI-compatible tool calls). Model Qwen/Qwen3-14B, revision
  40c069824f4251a91eefaf281ebe4c544efd3e18, dtype bfloat16; vllm 0.33.0, torch 2.14.0+cu130, transformers 5.20.2, CUDA 13.0, one H100 80GB;
  `vllm serve --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes
  --reasoning-parser qwen3 --seed 0` with `VLLM_USE_FLASHINFER_SAMPLER=0` (the host has no nvcc for the FlashInfer JIT; vLLM's PyTorch
  top-k / top-p / min-p sampler is used, same sampling semantics). Client: openai 1.55.3 through an ssh tunnel.
  Sampling (`configs/agent/sampling.json`, sha256 b5e3a0ff30d03c2f064b31246ed7a77a41f7e2ee594b082cc8090883045a9d3a): thinking mode with the model card's values (T 0.6, top_p 0.95, top_k 20,
  min_p 0, presence penalty 0); the forced final turn runs with enable_thinking=false. Nominal seeds 0, 1, 2 (request-level `seed`;
  identical repeats in the smoke test, not guaranteed under batching).
- Prompt and schema: `configs/agent/system_prompt.txt` (sha256 242e491413b7780ed2eee240b1737bdd23f08be3bb1b8271c53ebc0429e19185), `configs/agent/tools_schema.json` (sha256 9347a827e516234d5ce1d2310df4e12fe6f36b845402b53e5b1b049b3a99cd36);
  budget `configs/agent/budget.json` (sha256 fb5d76eaaa7238d38d2ddd07e00c1779ff042d00443c551bcf75626ebb773583); tool runner code sha 193657dd0d465a37 (cache run manifests).
- Budget (`configs/agent/budget.json`): at most 4 decision turns; at most 7 non-submit tool calls plus one reserved submit_choice;
  each reconstructor at most once; at most 4,096 completion tokens in total (vLLM's count, thinking included), non-final turns
  get max_tokens = min(2,048, remaining - 256), the 256 are reserved for the final turn; the final turn forces
  submit_choice through tool_choice and runs without thinking; a reply cut by max_tokens with budget left is a per_turn_truncation
  (one decision used), with no budget left the failure total_token_budget; the virtual clock (LLM wall + logical tool time +
  tool execution) must stay below 600 s or the window fails with timeout. Candidate ids are re-validated by the program; only
  candidates returned by a tool the agent ran are accepted.
- Failures (no output, counted in the denominator with ADE +inf and route 0): invalid_choice, no_submit, total_token_budget,
  timeout, parse_error, llm_error, no_candidates. No silent fallback. A separate row "C with selector fallback" substitutes the
  selector's choice when C has no valid output; its extra cost is the logical time of the tools the agent had not run.
- Oracle (diagnostic only, evaluator side): per window the best pool candidate (any route_ok; min ADE); never shown to the agent.

## Costs (three named quantities)
- reconstructed serial cost = LLM wall time + logical time of the tools actually run (measured once, single worker, when the
  cache was built) + tool execution time; the table's cost column and the 600 s clock use this quantity. It is not an
  uncached end-to-end latency.
- cached-mode elapsed = real wall time of the cached run (concurrency 12 for the formal run; 1 for the single-stream pilot run).
- live check = 10 pilot windows (seed 0, concurrency 1) with every reconstructor call executed for real (load, graph, search);
  reported next to the reconstructed cost of the same windows (agent_live_check.csv).

## Statistics
- Driver-clustered bootstrap, 1,000 resamples, 95 % percentile intervals. Rates (route, no output, route differences) pool the
  three seeds (S x N values, driver labels tiled) so the seed mean is taken inside every resample; failure-aware ADE medians
  through `stats.seed_median` (per-seed median, averaged inside the resample); paired ADE differences per seed
  (`stats.paired`, windows with output on both sides) and aggregated inside the resample (`stats.seed_paired`).
- Primary comparison: route recovery of C minus B (unconditional, paired by window). Secondary: C minus A. Both folds reported
  separately; no fold is merged.

## Pilot and adjustments
- Pilot (validation, 40 windows x seeds 0, 1, 2 = 120 conversations, concurrency 12): 100 % submitted; 0 invalid choices, 0 token-budget
  failures, 0 timeouts, 0 parse errors; 1 per-turn truncation (0.8 %); 3.8 decisions and 2.9 tool calls per window (max 4), forced submit
  in 83 %; completion tokens mean 1,717 (p95 2,659, max 3,899), reasoning tokens mean 1,537; LLM wall median 35 s per window.
  Tool use: beam 100 %, pf 99 %, ep 89 %; choices beam 60, pf 53, ep 7.
- Allowed one-time adjustment after the pilot: NOT used (thinking mode and the 2,048 per-turn cap kept). Before the pilot, with nothing
  frozen, one sentence was added to the system prompt stating that several tool calls may be requested in one turn.
- Formal-run plan: 659 windows x 3 seeds at concurrency 12 (about 35 min per seed from the pilot's throughput).
