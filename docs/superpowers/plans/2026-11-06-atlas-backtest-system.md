# Atlas Backtest System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, backtestable, self-improving 12-agent equity trading system (engine, scorecard, backtest harness, autoresearch) with a deterministic risk layer.

**Architecture:** A new top-level Python package `atlas/` (the existing `src/` and `prompts/examples/` stay untouched as reference). A daily pipeline runs 4 layers of schema-validated LLM agents; a deterministic risk layer clips orders; `SimBroker` fills at next open; a SQLite scorecard computes excess-return rank IC; Darwinian weights update through a shrinkage + dead-zone rule; an autoresearch loop edits one prompt on a git branch and keeps it only if a paired replay beats a null-test margin. All LLM calls are cached by input hash, so reruns are free.

**Tech Stack:** Python 3.11+, pydantic v2, pandas, numpy, scipy, pyyaml, anthropic SDK, yfinance, requests (FRED/ALFRED), ib_async (IBKR historical data only), sqlite3, matplotlib, pytest, hypothesis.

**Spec:** `docs/superpowers/specs/2026-11-06-atlas-backtest-system-design.md`

## Global Constraints

Every task's requirements include these, copied from the spec.

- Python; Claude API for agents; daily end-of-day cycle only; US equities/ETFs; **long-only** (shorts configurable, default off).
- Roster is 12 agents: macro (central bank/rates, volatility/risk regime, dollar/commodities); sector (semiconductors, financials, energy, healthcare/biotech); superinvestor (Druckenmiller-style, Ackman-style); decision (CRO, CIO, autonomous execution). The risk layer is code, not an agent.
- Benchmarks: SPY for macro and superinvestor picks; SMH, XLF, XLE, XLV for the four sector desks.
- Agent output contract: JSON with `agent, date, view (bullish|bearish|neutral), confidence 0-1, picks[{ticker, direction long|short|none, conviction 0-1, horizon_days, rationale}]`.
- Fills at the **next trading day's open**; commission and slippage are config values, pessimistic by default.
- Temperature 0; LLM calls cached by hash of model, prompt and inputs; editing a prompt invalidates only that agent's cache entries.
- Scoring: excess return over benchmark net of costs; primary metric **rank IC** (Spearman), plus hit rate; horizons 1, 5, 10 days (1-day drives weight updates; weight updates use non-overlapping samples); agents with **< 30 scored picks are "warming up"** (no weight change, ineligible for autoresearch).
- Weights start **1.0**, bounded **[0.3, 2.5]**, step **x1.05 up / x0.95 down**, only outside a null-test-derived dead zone; IC is shrunk toward the group mean in proportion to few observations.
- Risk defaults (`limits.yaml`): max single position **15%** of equity; max gross exposure **100%**; max positions **8**; drop orders under **$50** or with commission above **1%** of the order; no new buys if equity is down **3%** from prior close; kill switch at **15%** drawdown from peak (flatten to cash, halt until manual restart). Risk layer is fail-closed.
- LLM failure reason codes: `malformed_json, schema_invalid, empty_response, truncated, refusal, api_error, rate_limited, timeout`. Partial responses are stored verbatim in the failure row but **never scored, passed on, or traded**. One retry for malformed_json/schema_invalid/truncated/empty_response; backoff for api_error/rate_limited/timeout; no retry for refusal. A day with fewer than a configured minimum of valid agents is skipped, not traded.
- Autoresearch edits only prompt files on a git branch; paired replay on **40+** trading days; keep only if paired IC beats the null-test margin; final holdout period never seen by autoresearch; caps on edits and total cost.
- Backtests never read data after the as-of date; macro series use real-time vintages (ALFRED) with publication lags; no silent forward-fill of gaps in backtests.
- Reports must label pre-cutoff results "not evidence of skill" and include the date-only control agent's IC, null-test distribution, CIO ablation, failure table, API cost.
- Nothing in this plan places real orders or connects to a live account.

## Review Focus

Failure modes the spec implies but does not spell out, most likely first. Each has a pinning test in the owning task.

1. An agent picks a ticker **outside the fixed universe** (hallucinated or misspelled): rejected and logged, never traded or scored (Task 5).
2. **Constant convictions or fewer than 3 picks** make rank IC undefined (NaN): treated as "no signal", never a crash and never a weight change (Tasks 9, 10).
3. The **next-open price is missing** (holiday, halt, data gap): the order is dropped with a reason, never filled at a stale price (Task 8).
4. **Tiny equity makes integer share counts round to zero**: the order is dropped as below minimum, not submitted as a 0-share order (Task 8).
5. **Resuming after a crash mid-day** must not double-fill or double-score that day (Task 12).

---

## File Structure

```
pyproject.toml
atlas/
  __init__.py
  schemas.py            # Pick, AgentOutput, FailureReason, AgentResult
  config.py             # AgentSpec, Limits, Settings loaders
  config/agents.yaml, limits.yaml, settings.yaml
  prompts/*.md          # 12 agent prompts + control agent
  data/{provider.py, yf_provider.py, fred_provider.py, ibkr_provider.py, store.py}
  engine/{llm.py, failures.py, agent.py, pipeline.py}
  portfolio/{models.py, simbroker.py}
  risk/layer.py
  scoring/{store.py, metrics.py, weights.py}
  backtest/{runner.py, checkpoint.py, predlog.py, nulltest.py, control.py, ablation.py, report.py}
  autoresearch/{gitops.py, loop.py}
  cli.py
tests/                   # mirrors the package
```

---

### Task 1: Project scaffold and config loading

**Files:**
- Create: `pyproject.toml`, `atlas/__init__.py`, `atlas/config.py`, `atlas/config/agents.yaml`, `atlas/config/limits.yaml`, `atlas/config/settings.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AgentSpec(name:str, layer:int, kind:Literal["macro","sector","superinvestor","cro","cio","execution","control"], prompt_file:str, benchmark:str|None)`; `Limits(max_position_pct:float, max_gross_pct:float, max_positions:int, min_order_usd:float, max_commission_frac:float, daily_loss_pct:float, drawdown_kill_pct:float, long_only:bool)`; `Settings(universe:list[str], commission_per_share:float, commission_min:float, slippage_bps:float, min_valid_agents:int, rolling_ic_window:int, warmup_picks:int, budget_usd:float, model:str, model_cutoff:date)`; `load_agents(path)->list[AgentSpec]`, `load_limits(path)->Limits`, `load_settings(path)->Settings`.

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`: `test_agents_yaml_has_12_agents_in_expected_layers` (3 macro layer 1, 4 sector layer 2, 2 superinvestor layer 3, cro/cio/execution layer 4, control excluded from the 12 by kind); `test_limits_defaults_match_spec` (0.15, 1.0, 8, 50, 0.01, 0.03, 0.15, long_only True); `test_sector_benchmarks` (semiconductors→SMH, financials→XLF, energy→XLE, healthcare→XLV, macro/superinvestor→SPY); `test_unknown_kind_rejected`.
- [ ] **Step 2: Run** `pytest tests/test_config.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the models as pydantic classes in `atlas/config.py` and write the three YAML files with the spec's values. `pyproject.toml` declares the stack above and `pytest` config (`testpaths=["tests"]`, marker `live` skipped by default).
- [ ] **Step 4: Run** `pytest tests/test_config.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git add pyproject.toml atlas tests && git commit -m "feat: scaffold atlas package and config loading"`

### Task 2: Core schemas and failure types

**Files:**
- Create: `atlas/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `Pick`, `AgentOutput` (pydantic, fields per the contract; conviction/confidence constrained to [0,1], horizon_days > 0); `FailureReason(StrEnum)` with the eight codes; `FailureRecord(agent:str, date:date, reason:FailureReason, detail:str, attempt:int, raw_response:str)`; `AgentResult(agent:str, date:date, output:AgentOutput|None, failure:FailureRecord|None)` with property `status -> Literal["ok","no_opinion"]`.

- [ ] **Step 1: Write failing tests**: `test_valid_output_parses`; `test_conviction_above_one_rejected`; `test_direction_must_be_long_short_none`; `test_agent_result_status_no_opinion_when_failure_set`.
- [ ] **Step 2: Run** `pytest tests/test_schemas.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the models in `atlas/schemas.py`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add agent output schemas and failure types`

### Task 3: PriceStore with as-of snapshots (no-lookahead core)

**Files:**
- Create: `atlas/data/provider.py`, `atlas/data/store.py`
- Test: `tests/data/test_store_lookahead.py`

**Interfaces:**
- Produces: `PriceProvider(Protocol).fetch(tickers:list[str], start:date, end:date)->pd.DataFrame` (long format, columns `date,ticker,open,high,low,close,adj_close,volume`); `MacroProvider(Protocol).fetch(series_id:str, asof:date)->pd.Series` (values as known on `asof`, indexed by observation date); `Snapshot(asof:date, prices:pd.DataFrame, macro:dict[str,pd.Series])`; `PriceStore(prices:PriceProvider, macro:MacroProvider, cache_dir:Path)` with `snapshot(asof, tickers, lookback_days)->Snapshot`, `trading_days(start,end)->list[date]`, `next_open(ticker, after:date)->tuple[date,float]|None`, `outcome_prices(ticker, start:date, end:date)->pd.Series` (explicit forward access, used only by scoring and brokers).

- [ ] **Step 1: Write failing tests** with an in-memory fake provider spanning 60 days: `test_snapshot_never_contains_rows_after_asof` (max date <= asof for every ticker); `test_snapshot_macro_uses_vintage_as_of_date` (fake macro returns different values by `asof`; snapshot value equals the as-of vintage); `test_next_open_is_strictly_after_given_day`; `test_next_open_returns_none_when_no_later_bar`; `test_provider_cached_to_disk_second_call_makes_no_fetch`.
- [ ] **Step 2: Run** `pytest tests/data/test_store_lookahead.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `PriceStore`; `snapshot` slices the cached frame with `date <= asof` and never exposes the full frame; cache is parquet/CSV per ticker under `cache_dir`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add PriceStore with as-of snapshots`

### Task 4: Concrete data providers

**Files:**
- Create: `atlas/data/yf_provider.py`, `atlas/data/fred_provider.py`, `atlas/data/ibkr_provider.py`
- Test: `tests/data/test_providers.py`

**Interfaces:**
- Consumes: `PriceProvider`, `MacroProvider` from Task 3.
- Produces: `YFinanceProvider()`, `FredAlfredProvider(api_key:str, publication_lag_days:dict[str,int])`, `IBKRProvider(host:str, port:int, client_id:int)` (historical daily bars via `ib_async`; requires a running IB Gateway; not used by tests or the default config).

- [ ] **Step 1: Write failing tests** with mocked HTTP/clients: `test_yfinance_provider_returns_required_columns`; `test_fred_provider_requests_vintage_for_asof` (asserts the ALFRED call carries `realtime_end=asof`); `test_fred_provider_applies_publication_lag` (observation newer than `asof - lag` dropped); `test_ibkr_provider_maps_bars_to_schema` (fake client object); a `@pytest.mark.live` smoke test per provider.
- [ ] **Step 2: Run** `pytest tests/data/test_providers.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three providers; the IBKR one imports `ib_async` lazily so the package works without it.
- [ ] **Step 4: Run** the tests (live tests skipped). Expected: PASS.
- [ ] **Step 5: Commit** `feat: add yfinance, FRED/ALFRED and IBKR data providers`

### Task 5: LLM client, cache, budget, failure log, agent runner

**Files:**
- Create: `atlas/engine/llm.py`, `atlas/engine/failures.py`, `atlas/engine/agent.py`
- Test: `tests/engine/test_llm.py`, `tests/engine/test_agent.py`

**Interfaces:**
- Consumes: schemas from Task 2; `AgentSpec` from Task 1.
- Produces: `LLMResponse(text:str, input_tokens:int, output_tokens:int, stop_reason:str)`; `LLMClient(Protocol).complete(system:str, user:str, max_tokens:int)->LLMResponse` (may raise `RateLimited`, `Timeout`, `ApiError`); `AnthropicClient(model:str)`; `StubClient(fn:Callable[[str,str],str])`; `CachedLLM(client, cache_path:Path, model:str, usd_per_mtok:tuple[float,float], budget_usd:float)` with `call(system,user)->LLMResponse`, property `spent_usd`, raising `BudgetExceeded`; `FailureLog(db_path:Path)` with `record(FailureRecord)` and `summary()->pd.DataFrame`; `run_agent(spec:AgentSpec, prompt:str, inputs:dict, asof:date, llm:CachedLLM, failures:FailureLog, universe:set[str])->AgentResult`.

- [ ] **Step 1: Write failing tests**: `test_cache_hit_makes_no_client_call_and_costs_nothing`; `test_changed_prompt_misses_cache`; `test_budget_exceeded_raises`; `test_malformed_json_retried_once_then_no_opinion_with_raw_stored`; `test_schema_invalid_reason_lists_failing_fields`; `test_truncated_stop_reason_maps_to_truncated`; `test_refusal_not_retried`; `test_rate_limit_uses_backoff_then_gives_up` (injected sleep, no real waiting); `test_partial_json_stored_verbatim_but_result_has_no_output`; `test_pick_outside_universe_is_dropped_and_logged` (Review Focus 1; the rest of the output stays valid, the bad pick is excluded, a `schema_invalid`-style note is recorded).
- [ ] **Step 2: Run** `pytest tests/engine -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Cache key = sha256 of model, system, user. Retry policy exactly as in Global Constraints. Temperature fixed to 0 in `AnthropicClient`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add cached LLM client, failure log and agent runner`

### Task 6: Agent roster prompts and layer pipeline

**Files:**
- Create: `atlas/prompts/*.md` (12 agents + `control_date_only.md`), `atlas/engine/pipeline.py`
- Test: `tests/engine/test_pipeline.py`

**Interfaces:**
- Consumes: `run_agent`, `AgentSpec`, `Snapshot`, `Portfolio` (Task 8, passed as plain `dict` state here to avoid a forward dependency: `{"cash":float,"positions":dict[str,int],"equity":float}`).
- Produces: `Regime = Literal["RISK_ON","RISK_OFF","NEUTRAL"]`; `DayResult(asof:date, regime:Regime|None, results:dict[str,AgentResult], proposed_orders:list[dict]|None, skipped_reason:str|None)`; `Pipeline(specs, prompts:dict[str,str], llm, failures, universe, min_valid_agents:int)` with `run_day(asof, snapshot, weights:dict[str,float], state:dict)->DayResult`; `aggregate_regime(macro_outputs:list[AgentResult], weights:dict[str,float])->Regime`.

- [ ] **Step 1: Write failing tests** using `StubClient` with scripted JSON: `test_layers_run_in_order_1_to_4`; `test_macro_agents_in_one_layer_run_concurrently` (stub records overlapping calls with a barrier); `test_regime_weighted_vote` (higher-weight agent breaks a tie); `test_day_skipped_when_valid_agents_below_minimum` (reason lists failed agents and codes); `test_failed_agent_output_not_passed_to_next_layer`; `test_execution_agent_receives_cio_targets_and_equity`.
- [ ] **Step 2: Run** `pytest tests/engine/test_pipeline.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `Pipeline`. Use `ThreadPoolExecutor` per layer for parallelism. Write each prompt `.md` adapting `prompts/examples/*.md` to the output contract, instructing JSON-only output; superinvestor prompts keep the Druckenmiller and Ackman philosophies from the root README.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add agent prompts and 4-layer pipeline`

### Task 7: Risk layer

**Files:**
- Create: `atlas/risk/layer.py`
- Test: `tests/risk/test_layer.py`

**Interfaces:**
- Consumes: `Limits` (Task 1).
- Produces: `Order(ticker:str, side:Literal["buy","sell"], shares:int)` in `atlas/portfolio/models.py` (create this one dataclass here; Task 8 extends the file); `RiskDecision(orders:list[Order], rejected:list[tuple[Order,str]], halted:bool, halt_reason:str|None)`; `RiskLayer(limits:Limits, commission_per_share:float, commission_min:float)` with `apply(orders, positions:dict[str,int], cash:float, prices:dict[str,float], prior_close_equity:float, peak_equity:float)->RiskDecision`.

- [ ] **Step 1: Write failing tests**, one per limit: `test_single_position_clipped_to_15pct`; `test_gross_exposure_never_above_100pct`; `test_ninth_position_rejected`; `test_order_under_50_dollars_rejected`; `test_order_with_commission_over_1pct_rejected`; `test_buys_blocked_after_3pct_daily_loss_sells_allowed`; `test_drawdown_15pct_flattens_and_halts`; `test_short_sell_rejected_when_long_only`; `test_missing_price_fails_closed_no_orders`; hypothesis `test_property_exposure_and_cash_invariants` (random order sets: gross <= 1.0 and cash >= 0 after applying the decision).
- [ ] **Step 2: Run** `pytest tests/risk -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `RiskLayer.apply`; clip by reducing share counts (integer floor), always sells before buys.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add deterministic fail-closed risk layer`

### Task 8: Portfolio and SimBroker

**Files:**
- Modify: `atlas/portfolio/models.py`
- Create: `atlas/portfolio/simbroker.py`
- Test: `tests/portfolio/test_simbroker.py`

**Interfaces:**
- Consumes: `Order`, `PriceStore.next_open`.
- Produces: `Fill(order:Order, date:date, price:float, commission:float)`; `Portfolio(cash:float, positions:dict[str,int])` with `equity(prices:dict[str,float])->float`, `apply(fill)`; `Broker(Protocol).submit(orders:list[Order], decision_date:date)->tuple[list[Fill], list[tuple[Order,str]]]`; `SimBroker(store:PriceStore, commission_per_share:float, commission_min:float, slippage_bps:float)`; `size_order(target_value:float, price:float)->int` (integer shares, floor).

- [ ] **Step 1: Write failing tests**: `test_buy_fills_at_next_open_plus_slippage`; `test_sell_fills_at_next_open_minus_slippage`; `test_commission_charged_per_share_with_minimum`; `test_missing_next_open_drops_order_with_reason` (Review Focus 3); `test_tiny_equity_rounds_to_zero_shares_and_order_dropped` (Review Focus 4: $100 equity, 15% cap, $400 stock → no order); `test_cash_never_negative_after_fills`.
- [ ] **Step 2: Run** `pytest tests/portfolio -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add portfolio and SimBroker with costs`

### Task 9: Scoring store and metrics

**Files:**
- Create: `atlas/scoring/store.py`, `atlas/scoring/metrics.py`
- Test: `tests/scoring/test_metrics.py`, `tests/scoring/test_store.py`

**Interfaces:**
- Consumes: `AgentResult`, `PriceStore.outcome_prices`, `AgentSpec.benchmark`.
- Produces: `rank_ic(convictions:Sequence[float], excess:Sequence[float])->float` (NaN when fewer than 3 points or either series constant); `hit_rate(excess)->float`; `excess_return(ticker_prices:pd.Series, bench_prices:pd.Series, horizon:int, cost_frac:float)->float`; `ScoreStore(db_path:Path)` with `record_picks(result:AgentResult)`, `score_matured(asof:date, store:PriceStore, specs:list[AgentSpec], cost_frac:float)->int` (returns rows scored), `picks_frame(agent:str|None, horizon:int)->pd.DataFrame` (columns `agent,date,ticker,conviction,horizon,excess,scored`), `non_overlapping(df, horizon)->pd.DataFrame`.

- [ ] **Step 1: Write failing tests**: `test_rank_ic_perfect_monotone_is_one`; `test_rank_ic_constant_convictions_is_nan` and `test_rank_ic_fewer_than_three_points_is_nan` (Review Focus 2); `test_excess_return_subtracts_benchmark_and_costs` (hand-computed 5-day case); `test_pick_not_scored_before_horizon_matures`; `test_none_direction_not_scored`; `test_non_overlapping_samples_spaced_by_horizon`; `test_no_opinion_result_records_nothing`.
- [ ] **Step 2: Run** `pytest tests/scoring -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with scipy `spearmanr` guarded for NaN; SQLite schema one row per pick with horizon columns for 1, 5, 10.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add excess-return scoring, rank IC and score store`

### Task 10: Darwinian weights

**Files:**
- Create: `atlas/scoring/weights.py`
- Test: `tests/scoring/test_weights.py`

**Interfaces:**
- Consumes: `rank_ic`, `ScoreStore.picks_frame`.
- Produces: `AgentStats(agent:str, n_picks:int, ic:float)`; `shrunk_ic(ic:float, n:int, group_mean:float, k:float)->float` (`(n*ic + k*group_mean)/(n+k)`); `update_weights(weights:dict[str,float], stats:list[AgentStats], dead_zone:float, k:float, warmup_picks:int)->dict[str,float]`; `WeightBook(path:Path)` persisting weights as JSON with `load()` / `save()`.

- [ ] **Step 1: Write failing tests**: `test_shrinkage_pulls_small_samples_toward_mean`; `test_inside_dead_zone_weights_unchanged`; `test_above_dead_zone_multiplies_by_1_05`; `test_below_dead_zone_multiplies_by_0_95`; `test_warming_up_agents_never_change`; `test_nan_ic_treated_as_no_signal_weight_unchanged` (Review Focus 2); hypothesis `test_weights_always_within_0_3_and_2_5` over many random update rounds.
- [ ] **Step 2: Run** `pytest tests/scoring/test_weights.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `k` and `dead_zone` come from `Settings`/null test; group mean excludes warming-up and NaN agents.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add Darwinian weights with shrinkage and dead zone`

### Task 11: Prediction log and checkpoints

**Files:**
- Create: `atlas/backtest/predlog.py`, `atlas/backtest/checkpoint.py`
- Test: `tests/backtest/test_predlog.py`, `tests/backtest/test_checkpoint.py`

**Interfaces:**
- Produces: `PredictionLog(path:Path)` append-only JSONL with `append(day:date, results:dict[str,AgentResult])->str` (returns hash chained to the previous line's hash) and `verify()->bool`; `Checkpoint(path:Path)` with `save(state:dict)` (atomic write) and `load()->dict|None`; the state dict includes `last_completed_day`, portfolio, weights path and spend so far.

- [ ] **Step 1: Write failing tests**: `test_log_hash_chain_verifies`; `test_tampering_with_a_past_line_fails_verify`; `test_log_append_only_never_rewrites`; `test_checkpoint_atomic_write_survives_partial_write` (simulate crash by writing a temp file only); `test_checkpoint_roundtrip`.
- [ ] **Step 2: Run** `pytest tests/backtest/test_predlog.py tests/backtest/test_checkpoint.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add hash-chained prediction log and checkpoints`

### Task 12: Backtest runner

**Files:**
- Create: `atlas/backtest/runner.py`
- Test: `tests/backtest/test_runner.py`

**Interfaces:**
- Consumes: Tasks 3, 5-11.
- Produces: `Periods(train:tuple[date,date], validation:tuple[date,date], holdout:tuple[date,date])`; `BacktestConfig(periods:Periods, settings:Settings, limits:Limits, anonymize:bool, dry_run:bool)`; `BacktestResult(equity:pd.Series, spy:pd.Series, basket:pd.Series, day_results:dict[date,DayResult], skipped:dict[date,str])`; `estimate_cost(cfg, n_days:int, n_agents:int)->float`; `run_backtest(cfg, store, pipeline, broker, risk, scores, weightbook, predlog, checkpoint)->BacktestResult`; `anonymize_snapshot(snapshot:Snapshot, universe_sectors:dict[str,str])->tuple[Snapshot, dict[str,str]]` (opaque ids, prices rebased to 100, relative day index).

- [ ] **Step 1: Write failing tests** on a 30-day synthetic market with `StubClient`: `test_periods_must_be_chronological_and_non_overlapping`; `test_decisions_on_day_t_fill_at_t_plus_1_open`; `test_skipped_day_makes_no_trades_and_is_logged`; `test_cost_estimate_over_ceiling_requires_confirmation`; `test_dry_run_makes_zero_llm_calls`; `test_resume_after_interruption_does_not_double_fill_or_double_score` (Review Focus 5: run 10 days, kill after the fills of day 6, resume, final portfolio equals an uninterrupted run); `test_anonymize_hides_tickers_and_dates_but_keeps_sector`; `test_prediction_log_written_before_outcomes_scored`.
- [ ] **Step 2: Run** `pytest tests/backtest/test_runner.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the day loop in the order from spec section 4; write the prediction log entry before `score_matured` is called; checkpoint after each day completes (fills, scores, weights together).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add backtest runner with checkpointing and anonymisation`

### Task 13: Null test, control agent, CIO ablation

**Files:**
- Create: `atlas/backtest/nulltest.py`, `atlas/backtest/control.py`, `atlas/backtest/ablation.py`
- Test: `tests/backtest/test_nulltest.py`, `tests/backtest/test_control.py`, `tests/backtest/test_ablation.py`

**Interfaces:**
- Consumes: `ScoreStore.picks_frame`, `rank_ic`, `run_agent`.
- Produces: `NullDistribution(ics:np.ndarray)` with `quantile(q)->float`, `dead_zone()->float` (95th percentile of |IC|), `keep_margin()->float`; `null_test(picks:pd.DataFrame, n_shuffles:int, seed:int)->NullDistribution`; `run_control_agent(asof, universe, prompt, llm, failures)->AgentResult` (gets date and ticker list only, no market data); `cio_ablation(cio_returns:pd.Series, baseline_returns:pd.Series)->dict` with keys `cio_total, baseline_total, cio_beats_baseline`.

- [ ] **Step 1: Write failing tests**: `test_null_ic_centred_near_zero` (random picks, mean |mean IC| < 0.05 with fixed seed); `test_null_test_is_seeded_and_reproducible`; `test_dead_zone_is_95th_pct_of_abs_null`; `test_control_prompt_contains_no_prices` (assert user message has no digits besides the date); `test_ablation_reports_cio_not_beating_baseline`; `test_null_test_flags_leakage_when_shuffled_ic_is_high` (feed a leaky frame, expect a raised `LeakageSuspected`).
- [ ] **Step 2: Run** `pytest tests/backtest -k "null or control or ablation" -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules; shuffling permutes convictions within each agent across rows.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add null test, date-only control agent and CIO ablation`

### Task 14: Autoresearch

**Files:**
- Create: `atlas/autoresearch/gitops.py`, `atlas/autoresearch/loop.py`
- Test: `tests/autoresearch/test_gitops.py`, `tests/autoresearch/test_loop.py`

**Interfaces:**
- Consumes: `AgentStats`, `shrunk_ic`, `NullDistribution.keep_margin`, `run_backtest` replay pieces, `FailureLog.summary`.
- Produces: `GitOps(repo:Path)` with `create_branch(name)`, `commit_file(path, message)`, `merge_branch(name)`, `discard_branch(name)` (all restricted to paths under `atlas/prompts/`, else raise `ForbiddenPath`); `select_target(stats:list[AgentStats], failure_rates:dict[str,float], max_failure_rate:float, k:float, warmup_picks:int)->str|None`; `propose_edit(llm:CachedLLM, spec:AgentSpec, current_prompt:str, diagnostics:str)->str`; `PairedResult(old_ic:float, new_ic:float, n_days:int, diff:float)`; `paired_replay(agent:str, old_prompt:str, new_prompt:str, days:list[date], run_one_day:Callable)->PairedResult`; `run_autoresearch(cfg, ...)->list[EditRecord]` honouring `max_edits` and the cost budget, never touching holdout days.

- [ ] **Step 1: Write failing tests** against a temp git repo and stubs: `test_gitops_rejects_paths_outside_prompts_dir`; `test_select_target_skips_warming_up_and_high_failure_agents`; `test_select_target_none_when_all_ineligible`; `test_edit_kept_only_when_diff_exceeds_keep_margin`; `test_edit_reverted_when_diff_below_margin_main_unchanged`; `test_paired_replay_requires_at_least_40_days`; `test_holdout_days_never_passed_to_replay`; `test_loop_stops_at_max_edits_and_at_budget`; `test_failed_evaluation_leaves_main_unchanged`.
- [ ] **Step 2: Run** `pytest tests/autoresearch -v`. Expected: FAIL.
- [ ] **Step 3: Implement** using `subprocess` for git; paired replay reruns only the target agent with the new prompt (other agents' calls hit the cache) and compares paired IC on identical days.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add autoresearch loop with paired replay and prompt-only git ops`

### Task 15: Report generation

**Files:**
- Create: `atlas/backtest/report.py`
- Test: `tests/backtest/test_report.py`

**Interfaces:**
- Consumes: `BacktestResult`, `ScoreStore`, `NullDistribution`, `FailureLog.summary`, autoresearch `EditRecord`s, ablation dict, control IC.
- Produces: `build_report(result, scores, null, edits, ablation, control_ic:float|None, failures:pd.DataFrame, cost_usd:float, out_dir:Path)->Path` writing `report.md`, `equity.png`, `agents.csv`, `autoresearch.csv`.

- [ ] **Step 1: Write failing tests**: `test_report_contains_all_required_sections` (benchmarks, per-period metrics, per-agent table, control IC, autoresearch log, null thresholds, CIO ablation, failure table, API cost, verdict line); `test_pre_cutoff_periods_labelled_not_evidence_of_skill`; `test_verdict_says_no_evidence_when_ic_inside_null`; `test_metrics_net_of_costs_match_hand_computed_sharpe_and_drawdown`.
- [ ] **Step 2: Run** `pytest tests/backtest/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with pandas and matplotlib (Agg backend); metrics computed per period separately.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add backtest report generation`

### Task 16: CLI and golden end-to-end backtest

**Files:**
- Create: `atlas/cli.py`, `tests/golden/test_golden_backtest.py`, `tests/golden/fixtures/` (small synthetic price CSVs and scripted stub responses)
- Modify: `README.md` (add a short "Running the backtest system" section)

**Interfaces:**
- Consumes: everything above.
- Produces: console script `atlas` with subcommands `backtest --config PATH [--dry-run] [--yes]`, `autoresearch --config PATH`, `report --run-dir PATH`.

- [ ] **Step 1: Write failing tests**: `test_golden_backtest_is_identical_across_two_runs` (stub model, fixed fixtures, compares equity series and score DB hashes); `test_golden_backtest_resume_matches_uninterrupted_run`; `test_cli_backtest_dry_run_exits_zero_and_makes_no_llm_calls`; `test_cli_requires_confirmation_above_cost_ceiling_unless_yes`; `test_null_test_on_golden_data_gives_ic_near_zero`.
- [ ] **Step 2: Run** `pytest tests/golden -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `atlas/cli.py` using argparse and wire the components; document prerequisites (API key, FRED key, optional IB Gateway) in the README section.
- [ ] **Step 4: Run** the full suite: `pytest -q`. Expected: PASS, with `live` tests skipped.
- [ ] **Step 5: Commit** `feat: add CLI and golden end-to-end backtest`

---

## Self-review notes

- **Spec coverage:** roster and benchmarks (T1, T6); data flow, contract, fills, costs (T2, T3, T6, T8); scoring, weights, null test (T9, T10, T13); autoresearch (T14); risk layer (T7); harness, periods, determinism, cost control, resumability (T5, T12); validity controls 1-7 (labelling T15; control agent T13; anonymisation T12; contamination probe and older-model check are marked optional/opportunistic in the spec and are **deliberately left out of this plan**; point-in-time data T3-T4; prediction log T11); report (T15); error handling (T5, T6, T7); testing strategy including the golden and null tests (T3, T13, T16).
- **Deferred by the spec:** open items in spec section 12 (final risk values, holdout model cutoff, universe, window lengths) are `Settings`/`Limits` config values, finalised after the first null-test run.
- **Type consistency:** `Order` is first defined in Task 7 and extended in Task 8; the pipeline in Task 6 uses a plain dict state to avoid depending on Task 8's `Portfolio`.
