# Atlas Error Log, Hub-and-Spoke Models and Validation Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a run-level error log, hub-and-spoke model routing under one shared budget, and an out-of-sample (validation) confirmation step with abstention accounting for autoresearch.

**Architecture:** Three independent changes to the merged `atlas/` package, built in this order: (A) `atlas/errorlog.py` plus wiring in the runner, CLI and report; (C) autoresearch hardening inside `atlas/autoresearch/loop.py` and its CLI caller; (B) a shared `Budget` and per-agent model routing through config, `CachedLLM`, `Pipeline`, runner and CLI. Each part leaves the full suite green and works alone.

**Tech Stack:** Python 3.11+, pydantic v2, pandas, numpy, sqlite3, pytest, hypothesis (already in the project).

**Spec:** `docs/superpowers/specs/2026-11-07-atlas-errorlog-hubspoke-confirm-design.md`

## Global Constraints

Every task's requirements include these (copied from the spec and the earlier system).

- No live or paper execution is added; the risk layer stays the only path from orders to fills; no new network calls besides the existing Anthropic, yfinance, FRED paths.
- Keys come from the environment only and never appear in any file, log line, exception message or report. Every string written to `errors.log` goes through `atlas.engine.llm.sanitise` and is truncated; raw model replies are never written there.
- Writing to `errors.log` can never raise into the run (degrade to one stderr warning per process) and is thread-safe (agents run in a `ThreadPoolExecutor`).
- `errors.log`: JSON Lines, append-only, flushed and fsynced per event, mode 0600, rotate at 5 MB to `errors.log.1` (keep one); dry runs write under `<run_dir>/dry_run/`.
- Effective model per agent = `AgentSpec.model` > `Settings.spoke_model` (kinds `macro`, `sector`, `superinvestor`) > `Settings.model`; the `control` agent uses the spoke model (or `Settings.model` if none). With `spoke_model` unset and no per-agent models, behaviour is byte-identical to today.
- One hard budget across all models in a run directory; per-model spend is tracked inside it; the cache key already includes the model.
- A run is refused (exit 2) if any model in use has a documented training cutoff in `MODEL_TRAINING_CUTOFFS` later than `Settings.model_cutoff`; the shipped `model_cutoff` becomes `2026-06-30`.
- Validation confirmation uses only validation-period days whose scoring exit is also inside validation (never holdout); confirmation results never feed target selection or proposal diagnostics (both stay train-only).
- Defaults: `confirm_on_validation=True`, `confirm_min_days=20`, `min_pair_ratio=0.95`, `impute_abstentions=True`; an edit with fewer than `confirm_min_days` validation days is decision `unconfirmed` (not kept).
- Keep rule on validation: `val.diff > keep_margin * max(1, sqrt(rolling_ic_window / val.new_pairs))` and the edit gate passes on the validation result.
- Existing tests, exit codes (0 ok, 1 operational, 2 usage) and the stored-run record format keep working; new Settings/AgentSpec fields have defaults so old `run_config.json` files still load (a run started before this change may be refused by the config check, with the usual message).

## Review Focus

Failure modes the spec implies but no task's tests would exercise by themselves, most likely first. Each has a pinning test in the owning task.

1. **Logging breaks the run**: a read-only or full run directory, or two agent threads logging at once, must not raise or interleave lines (Task A1).
2. **Mixed-model budget race**: two `CachedLLM` instances sharing one `Budget` called from many threads must never exceed the cap, and a resume must restore spend once, not once per model (Task B2).
3. **Silent contamination by a per-agent model**: an agent override or `spoke_model` with a later documented cutoff than `model_cutoff` must refuse the run, not warn (Task B1).
4. **Confirmation peeks at holdout**: the last validation day's exit date lies in the holdout; it must be excluded, and a holdout day in the confirmation set must raise `HoldoutViolation` (Task C3).
5. **Confirmation shopping**: an edit confirmed on validation is the only way to keep one; repeated attempts must each be logged with their confirmation numbers and capped by `max_edits` (Task C3).

---

## File Structure

```
atlas/errorlog.py                      # NEW  ErrorLog, NullErrorLog, read_events
atlas/engine/failures.py               # mirror failures into the error log
atlas/engine/llm.py                    # Budget; CachedLLM(budget=...)
atlas/engine/pipeline.py, agent.py     # route each agent to its model's CachedLLM
atlas/config.py                        # AgentSpec.model, Settings new fields, MODEL_TRAINING_CUTOFFS
atlas/config/settings.yaml, agents.yaml(unchanged defaults)
atlas/backtest/runner.py               # events, estimate_cost per model, spent_by_model
atlas/backtest/report.py               # Errors section, model column, per-model cost
atlas/autoresearch/loop.py             # DayPairs, imputation, validation confirmation
atlas/cli.py                           # ErrorLog wiring, `errors` subcommand, per-model clients
tests/...                              # one test module per task
```

---

## Part A: Error log

### Task A1: ErrorLog module

**Files:**
- Create: `atlas/errorlog.py`
- Test: `tests/test_errorlog.py`

**Interfaces:**
- Produces:
  - `class ErrorLog` with `__init__(self, path: Path | None, *, run_id: str | None = None, max_bytes: int = 5_000_000)`; `event(self, kind: str, message: str = "", *, level: str = "error", day: date | str | None = None, agent: str | None = None, reason: str | None = None, **fields) -> None` (never raises); `exception(self, kind: str, exc: BaseException, *, level: str = "error", day=None, agent=None, **fields) -> None` (type name, sanitised message, at most 8 frames as `file:line func`, no locals, no `repr` of arguments).
  - `class NullErrorLog(ErrorLog)` (path `None`: every call is a no-op).
  - `read_events(path: Path, *, kind: str | None = None) -> list[dict]` (missing file → `[]`; a torn final line is skipped; a malformed earlier line raises `ValueError`).
  - Line format: `{"ts_utc": ISO8601, "level": ..., "kind": ..., "day": ..., "agent": ..., "reason": ..., "message": ..., "fields": {...}, "run_id": ...}`.

- [ ] **Step 1: Write the failing tests** in `tests/test_errorlog.py`: `test_event_appends_one_json_line_with_all_keys`; `test_strings_are_sanitised_and_truncated` (a canary `sk-ant-api03-CANARY`, a URL with `?api_key=CANARY`, a 5000-char message: none of the canary survives, message ≤ 500 chars); `test_exception_has_sanitised_traceback_without_locals` (a local variable holding the canary must not appear); `test_never_raises_when_directory_is_read_only_or_path_is_a_directory` (warns once on stderr, second failure silent); `test_concurrent_threads_write_whole_lines` (16 threads x 50 events, every line parses, 800 lines); `test_rotates_at_max_bytes_keeping_one_backup`; `test_file_mode_is_0600`; `test_read_events_skips_torn_final_line_but_raises_on_earlier_corruption`; `test_null_error_log_is_a_noop`; `test_unknown_level_is_coerced_to_error`.
- [ ] **Step 2: Run** `pytest tests/test_errorlog.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `ErrorLog` with a `threading.Lock` around write/rotate; open with `O_APPEND|O_CREAT`, mode `0o600`; fsync each event; wrap everything in `try/except Exception` and emit one stderr line `warning: could not write error log (<ExceptionType>)` the first time only. Reuse `atlas.engine.llm.sanitise` (import inside the module, no new regex).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: add run-level ErrorLog`.

### Task A2: Mirror failures and emit runner events

**Files:**
- Modify: `atlas/engine/failures.py`, `atlas/backtest/runner.py`
- Test: `tests/engine/test_failures_mirror.py`, `tests/backtest/test_runner_errorlog.py`

**Interfaces:**
- Consumes: `ErrorLog` from A1.
- Produces: `FailureLog.__init__(self, db_path, run_id=None, error_log: ErrorLog | None = None)`; `record()` additionally writes `error_log.event("call_failure", detail, level="warning", day=rec.date, agent=rec.agent, reason=rec.reason.value, attempt=rec.attempt)` for kind `call_failure` and `event("dropped_pick", ..., level="info")` for `dropped_pick` (never `raw_response`). `run_backtest(..., error_log: ErrorLog | None = None)` writes: `day_skipped` (level warning; fields `failed` = agent→code), `run_paused` (fields `pause_kind`, `failed`), `budget_exceeded`, `fatal_api_error`, `risk_error` (the swallowed exception, via `exception()`), `halted` (level error, with `halt_reason`).

- [ ] **Step 1: Write the failing tests**: `test_failure_record_is_mirrored_without_raw_response`; `test_dropped_pick_mirrored_as_info`; `test_skipped_day_event_lists_failed_agents_and_codes`; `test_transient_pause_writes_run_paused_before_raising`; `test_budget_pause_writes_budget_exceeded`; `test_fatal_api_error_event_is_sanitised` (canary in the exception text); `test_risk_layer_exception_is_logged_and_sends_no_orders` (patch `RiskLayer.apply` to raise); `test_kill_switch_writes_halted_event_once`; `test_resumed_run_appends_to_the_same_log`. Reuse the fixtures in `tests/backtest/test_runner.py` (fake provider, `StubClient`).
- [ ] **Step 2: Run** the new test files. Expected: FAIL.
- [ ] **Step 3: Implement** the optional parameter and the event calls at the existing sites (`pause()` closure, the `FatalApiError` branch, the skip branch where `st.skipped[T]` is set, the risk `except`, the halt branch). `None` means no logging, so existing callers and tests are unchanged.
- [ ] **Step 4: Run** `pytest tests/engine tests/backtest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: log failures and run events to the error log`.

### Task A3: CLI wiring and `atlas errors`

**Files:**
- Modify: `atlas/cli.py`
- Test: `tests/golden/test_errorlog_cli.py`

**Interfaces:**
- Consumes: A1, A2.
- Produces: constant `ERRORS_NAME = "errors.log"`; `Run.errors: ErrorLog` created in `build_run` at `rc.path(ERRORS_NAME)` (so dry runs land in `dry_run/`) and passed to `FailureLog(error_log=...)` and `run_backtest(error_log=...)`; module-level `ACTIVE_ERRORLOG: ErrorLog | None` set by `build_run`, reset at the start of `main()`; `main()` records `usage_error`/`operational_error`/`resume_refused`/`data_error`/`fatal_api_error` events to `ACTIVE_ERRORLOG` before printing, and a final `except Exception` writes `unexpected_exception` (sanitised traceback), prints `error: unexpected <Type>: <sanitised message> (details: <path to errors.log>)` and returns 1; new subcommand `errors --run-dir PATH [--tail N=20] [--kind K]` printing one human line per event (`ts level kind day agent reason message`), exit 1 with `error: no error log in <dir>` if missing.

- [ ] **Step 1: Write the failing tests**: `test_rejected_request_writes_fatal_api_error_event_and_exits_1`; `test_unexpected_exception_is_caught_logged_and_exits_1_without_traceback` (inject a client that raises `RuntimeError("boom sk-ant-CANARY")`); `test_paused_run_leaves_run_paused_event_and_resume_appends`; `test_dry_run_writes_log_under_dry_run_dir_only`; `test_errors_subcommand_prints_tail_and_filters_by_kind`; `test_errors_subcommand_missing_log_exits_1`; `test_canary_key_absent_from_errors_log_and_stdout_stderr`; `test_config_error_before_run_known_prints_error_but_creates_no_log`.
- [ ] **Step 2: Run** `pytest tests/golden/test_errorlog_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** as described; `errors` is read-only and needs no keys.
- [ ] **Step 4: Run** `pytest tests/golden -q`. Expected: PASS (existing golden tests unchanged).
- [ ] **Step 5: Commit** `feat: wire the error log into the CLI and add atlas errors`.

### Task A4: Report section and docs

**Files:**
- Modify: `atlas/backtest/report.py`, `atlas/cli.py` (pass events to `build_report`), `README.md`
- Test: `tests/backtest/test_report_errors.py`

**Interfaces:**
- Produces: `build_report(..., errors: list[dict] | None = None)` adds `## Errors` after `## Failures`: `None` → "not provided"; `[]` → "no errors logged"; else counts by `kind` and `level`, the last 10 events (sanitised again, table cells escaped like the other tables) and the path `errors.log`. The verdict logic is untouched.

- [ ] **Step 1: Write the failing tests**: `test_errors_section_absent_value_says_not_provided`; `test_errors_section_counts_and_last_ten`; `test_errors_section_escapes_pipes_and_newlines`; `test_verdict_section_unchanged_with_and_without_errors` (byte-identical Verdict section).
- [ ] **Step 2: Run** the file. Expected: FAIL.
- [ ] **Step 3: Implement**; `cmd_report` reads `read_events(rc.path(ERRORS_NAME))` (tolerant); README "Error log" paragraph (where it is, how to read it, that raw replies stay in `failures.sqlite`).
- [ ] **Step 4: Run** `pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: report the error log`.

---

## Part C: Autoresearch validation confirmation and abstention accounting

### Task C1: Settings and the configurable pair floor

**Files:**
- Modify: `atlas/config.py`, `atlas/autoresearch/loop.py` (`edit_gate`)
- Test: `tests/test_config.py`, `tests/autoresearch/test_gate.py`

**Interfaces:**
- Produces: `Settings.confirm_on_validation: bool = True`, `confirm_min_days: int = Field(20, ge=5)`, `min_pair_ratio: float = Field(0.95, gt=0, le=1)`, `impute_abstentions: bool = True`; `edit_gate` uses `cfg.min_pair_ratio` in place of the literal `0.8` (message names the setting).

- [ ] **Step 1: Write the failing tests**: `test_new_settings_defaults`; `test_gate_rejects_ninety_percent_of_pairs_by_default` (90 of 100 pairs → rejected; previously passed); `test_gate_accepts_at_exactly_the_ratio`; `test_pair_ratio_setting_is_honoured` (set 0.8 → the same result passes); `test_old_run_config_still_loads_without_new_fields`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/autoresearch tests/test_config.py -q`. Expected: PASS (update any existing test that relied on 0.8).
- [ ] **Step 5: Commit** `feat: configurable pair-ratio floor and confirmation settings`.

### Task C2: DayPairs and abstention imputation

**Files:**
- Modify: `atlas/autoresearch/loop.py`, `atlas/cli.py` (replay `scorer`)
- Test: `tests/autoresearch/test_imputation.py`

**Interfaces:**
- Produces: `class DayPairs(NamedTuple): tickers: list[str]; convictions: list[float]; excess: list[float]`; `RunOneDay = Callable[[date, str], DayPairs | tuple[list[float], list[float]] | None]`; `paired_replay(agent, old_prompt, new_prompt, days, run_one_day, *, impute: bool = False, min_days: int = MIN_REPLAY_DAYS) -> PairedResult` where, with `impute=True` and `DayPairs` results, a ticker present on one side only enters the other side with conviction `0.0` and the first side's excess (counts in new `PairedResult.old_imputed`, `new_imputed: int = 0`); plain 2-tuples never impute; `old_pairs`/`new_pairs` stay the RAW counts (the gate uses them). The CLI scorer returns `DayPairs`.

- [ ] **Step 1: Write the failing tests**: `test_dropping_a_loser_no_longer_removes_it_from_the_sample` (stub where the new prompt omits the day's worst pick: with `impute=True` its IC gain is smaller than with `impute=False`; assert both numbers); `test_symmetric_imputation_for_picks_only_the_new_prompt_makes`; `test_two_tuple_results_are_unchanged_and_never_imputed`; `test_raw_pair_counts_exclude_imputed`; `test_imputed_pairs_have_zero_conviction_and_other_sides_excess`; `test_min_days_parameter_lowers_the_required_days`; `test_cli_scorer_returns_tickers`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** in `paired_replay`; `run_autoresearch` passes `impute=cfg.impute_abstentions`.
- [ ] **Step 4: Run** `pytest tests/autoresearch tests/golden -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: impute abstentions in paired replay`.

### Task C3: Validation confirmation in the loop

**Files:**
- Modify: `atlas/autoresearch/loop.py`
- Test: `tests/autoresearch/test_confirmation.py`

**Interfaces:**
- Consumes: C1, C2.
- Produces: `validation_days(trading_days: Sequence[date], periods: Periods, horizon: int = 1, n: int = 60) -> list[date]` (the `n` most recent validation days whose `horizon`-day exit is also inside validation); `check_validation_only(days, trading_days, periods, horizon) -> None` (raises `HoldoutViolation` for any day or exit outside validation); `confirm_margin(keep_margin: float, window: int, new_pairs: int) -> float` = `keep_margin * max(1.0, sqrt(window / max(1, new_pairs)))`; `Decision` gains `"unconfirmed"`; `EditRecord.confirm: dict | None = None` with keys `n_days, old_ic, new_ic, diff, margin, new_pairs, passed, reason`; inside `run_autoresearch`, after a train win and BEFORE `merge_branch`, run the validation replay when `cfg.confirm_on_validation`; merge only if confirmed. Fewer than `cfg.confirm_min_days` validation days → `unconfirmed`. The train/validation guards are separate (`allowed_train`, `allowed_validation`).

- [ ] **Step 1: Write the failing tests** (real temp git repo as in `tests/autoresearch/test_loop.py`): `test_train_win_and_validation_win_is_kept_and_merged`; `test_train_win_validation_loss_is_unconfirmed_and_branch_discarded` (original branch byte-identical); `test_validation_too_short_is_unconfirmed_not_kept`; `test_confirmation_disabled_keeps_old_behaviour`; `test_selective_abstention_learned_on_train_fails_validation` (stub prompt that drops exactly the train-period worst picks: passes train, fails validation); `test_last_validation_day_with_holdout_exit_is_excluded`; `test_holdout_day_in_confirmation_raises_holdout_violation`; `test_confirm_margin_grows_with_fewer_pairs_and_never_below_keep_margin`; `test_confirmation_numbers_are_logged_in_the_jsonl_record`; `test_confirmation_result_never_reaches_proposal_diagnostics_or_target_selection` (spy on `build_diagnostics`/`select_target` inputs); `test_every_attempt_is_logged_and_max_edits_caps_confirmations`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** with a validation `guarded` wrapper mirroring the train one; the gate (`edit_gate`) is applied to the validation `PairedResult` with `min_days=cfg.confirm_min_days`; record before cleanup as today.
- [ ] **Step 4: Run** `pytest tests/autoresearch -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: confirm autoresearch edits on validation days`.

### Task C4: CLI, report, docs

**Files:**
- Modify: `atlas/cli.py` (`cmd_autoresearch`), `atlas/backtest/report.py` (Autoresearch section), `README.md`
- Test: `tests/golden/test_autoresearch_confirm.py`

**Interfaces:**
- Produces: `cmd_autoresearch` passes the replay builder for validation days (the existing `ReplayContext.context(day)` already serves any day in `weights_history`); refuses with a clear message when `confirm_on_validation` is true and the stored run's validation period has fewer than `confirm_min_days` eligible days (exit 1, "give the validation period at least N trading days or set `settings.confirm_on_validation: false`"); report Autoresearch table shows the confirmation columns and the count of confirmations run; README documents the validation requirement and that validation is partly spent by each confirmation.

- [ ] **Step 1: Write the failing tests**: `test_cli_refuses_short_validation_with_actionable_message`; `test_cli_runs_confirmation_end_to_end_with_stub_model_on_a_long_golden_fixture` (extend the fixture generator to 80 days: train 45, validation 25, holdout 10 pre-cutoff-safe); `test_report_autoresearch_section_lists_confirmation_numbers`; `test_confirmation_count_stated`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: wire validation confirmation into the CLI and report`.

---

## Part B: Hub-and-spoke model routing

### Task B1: Config, price lookup, cutoff table

**Files:**
- Modify: `atlas/config.py`, `atlas/config/settings.yaml`, `atlas/cli.py` (`resolve_settings` validation, shipped-cutoff warning text)
- Test: `tests/test_config.py`, `tests/golden/test_model_cutoff.py`

**Interfaces:**
- Produces: `AgentSpec.model: str | None = None`; `Settings.spoke_model: str | None = None`, `spoke_usd_per_mtok_input/output: float | None` (both-or-neither validator); `SPOKE_KINDS = frozenset({"macro","sector","superinvestor"})`; `Settings.agent_model(spec: AgentSpec) -> str` (precedence in Global Constraints; `control` → `spoke_model or model`); `Settings.prices_for(model: str) -> tuple[float, float]` (explicit pair for `model`/`spoke_model` if set, else the table, else `UnknownModelPrice` naming the field to set; agent-level override models need a table entry); `MODEL_TRAINING_CUTOFFS: dict[str, date]` (`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-1` → `date(2026, 6, 30)`; comment cites the Models overview page and the read date); `check_model_cutoffs(settings, specs) -> list[str]` returning warnings for unknown models and raising `ModelCutoffError(ValueError)` when a used model's table cutoff is later than `settings.model_cutoff`; `SHIPPED_MODEL_CUTOFF = date(2026, 6, 30)` and `settings.yaml` updated; the CLI warning now says "model_cutoff is the shipped default; re-check it on the Models overview page" only for models absent from the table.

- [ ] **Step 1: Write the failing tests**: `test_agent_model_precedence_override_then_spoke_then_main`; `test_control_uses_spoke_model`; `test_unset_spoke_model_changes_nothing`; `test_prices_for_uses_table_and_explicit_pair_and_refuses_unknown`; `test_spoke_price_pair_both_or_neither`; `test_cutoff_later_than_model_cutoff_refuses_run_exit_2` (spoke model with a later table date than `model_cutoff`); `test_per_agent_override_model_checked_too`; `test_unknown_model_warns_and_requires_explicit_cutoff_at_least_max_known`; `test_shipped_default_cutoff_is_2026_06_30`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**; `resolve_settings`/`build_run` call `check_model_cutoffs` and map `ModelCutoffError` to `UsageError`.
- [ ] **Step 4: Run** `pytest tests/test_config.py tests/golden -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: spoke model, per-agent model and training-cutoff table`.

### Task B2: Shared Budget

**Files:**
- Modify: `atlas/engine/llm.py`
- Test: `tests/engine/test_budget.py` (plus existing `tests/engine/test_llm.py` stays green)

**Interfaces:**
- Produces: `class Budget` with `__init__(self, cap_usd: float, *, initial_spent_usd: float = 0.0)`, `spent_usd` (property), `reserved_usd`, `spent_by_model() -> dict[str, float]`, `restore_spent(usd)` (raise-only, as today), `reserve(estimate: float) -> None` (raises `BudgetExceeded` with today's message), `settle(estimate: float, model: str, actual: float | None) -> None`, a single `threading.Lock`; `CachedLLM.__init__(..., budget_usd: float = 0.0, *, initial_spent_usd: float = 0.0, budget: Budget | None = None)` (when `budget` is given it is shared and `budget_usd` is ignored; otherwise a private `Budget` is created, preserving today's behaviour); `CachedLLM.spent_usd` and `restore_spent` delegate to the budget.

- [ ] **Step 1: Write the failing tests**: `test_two_models_share_one_cap_under_32_threads` (stub clients, cap for N calls, exactly N calls succeed in total, spend ≤ cap); `test_spent_by_model_sums_to_total`; `test_restore_spent_once_on_shared_budget`; `test_pre_call_reservation_uses_each_models_own_prices`; `test_failed_call_releases_reservation_on_shared_budget`; `test_single_model_behaviour_identical_to_before` (the existing suite is the oracle).
- [ ] **Step 2: Run** `pytest tests/engine/test_budget.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** by moving `_spent`, `_reserved`, `_lock` and the reserve/settle logic into `Budget`; `call_with_meta` keeps its structure.
- [ ] **Step 4: Run** `pytest tests/engine -q`. Expected: PASS.
- [ ] **Step 5: Commit** `refactor: shared Budget for multi-model spend`.

### Task B3: Routing and per-model estimate

**Files:**
- Modify: `atlas/engine/pipeline.py`, `atlas/backtest/runner.py` (`estimate_cost`)
- Test: `tests/engine/test_pipeline_routing.py`, `tests/backtest/test_estimate_models.py`

**Interfaces:**
- Consumes: B1, B2.
- Produces: `Pipeline.__init__(..., llms: Mapping[str, CachedLLM] | None = None, models: Mapping[str, str] | None = None)` where `models` maps agent name → effective model and `llms` maps model → `CachedLLM`; `Pipeline.llm_for(spec) -> CachedLLM` (falls back to `self.llm`); `run_day`, `run_single` and the layer workers call `run_agent(..., llm=self.llm_for(spec))`; `estimate_cost(cfg, n_days, n_agents, agent_models: Sequence[str] | None = None) -> float` summing `settings.prices_for(model)` per agent (unchanged result when `agent_models` is `None`).

- [ ] **Step 1: Write the failing tests**: `test_each_agent_call_goes_to_its_models_client` (record which stub client served each agent; spokes → spoke stub, hub → hub stub); `test_run_single_routes_like_run_day`; `test_missing_model_mapping_falls_back_to_default_llm`; `test_estimate_matches_hand_computed_mixed_cost` (9 spokes + 3 hub at known prices); `test_estimate_unchanged_without_agent_models`; `test_cache_keys_differ_between_models_for_identical_prompts`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/engine tests/backtest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: route agents to their models and estimate per model`.

### Task B4: CLI construction, control, autoresearch, per-model spend

**Files:**
- Modify: `atlas/cli.py`, `atlas/backtest/runner.py` (checkpoint `spent_by_model`, optional), `atlas/backtest/report.py`
- Test: `tests/golden/test_hub_spoke_cli.py`

**Interfaces:**
- Consumes: B1-B3.
- Produces: `anthropic_client(settings, model: str | None = None)`; `build_run` builds one client and one `CachedLLM` per distinct effective model, all sharing one `Budget` and the same cache file, sets `run.llm` to the hub (`Settings.model`) `CachedLLM`, `run.llms` (model → `CachedLLM`) and passes `llms`/`models` to `Pipeline`; dry runs use `_NoCalls` for every model; `estimate_cost` gets the effective models (including control when `--with-control`); `run_control_pass` uses `run.llms[settings.agent_model(control_spec)]`; autoresearch proposals use `run.llm` (hub), replays use each agent's model via the pipeline; checkpoint stores optional `spent_by_model` and `restore_spent` is called once on the shared budget; the report Cost section prints one line per model and the Agents table a `model` column (`build_report(..., agent_models: dict[str, str] | None = None, spent_by_model: dict[str, float] | None = None)`); Limitations gains "mixed models: results cannot be attributed to the hub or the spokes" only when more than one model is in use.

- [ ] **Step 1: Write the failing tests**: `test_mixed_run_dry_run_makes_zero_calls_and_prints_models`; `test_mixed_run_calls_reach_the_right_stub_clients_and_spend_is_split`; `test_hard_cap_spans_both_models`; `test_resume_restores_spend_once_for_all_models`; `test_changing_spoke_model_on_an_existing_run_is_refused_by_the_stored_record`; `test_control_agent_uses_spoke_model`; `test_report_shows_model_column_and_per_model_cost_only_when_mixed`; `test_single_model_golden_run_is_byte_identical_to_before` (equity, score rows, prediction-log bytes against a run with `spoke_model` unset).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/golden -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: build one client per model under a shared budget`.

### Task B5 (optional): `atlas probe --model`

**Files:**
- Modify: `atlas/cli.py`
- Test: `tests/golden/test_probe_cli.py`

**Interfaces:**
- Produces: `probe --run-dir PATH [--samples N] [--model MODEL]` (optional; help says so): probes the named model (default `Settings.model`), priced via `prices_for`, refused if unknown; `contamination.json` records the model; the report line names it.

- [ ] **Step 1: Write the failing tests**: `test_probe_model_flag_uses_that_client_and_prices`; `test_probe_unknown_model_without_prices_exits_2`; `test_contamination_json_records_model`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: optional probe --model`.

### Task B6: Docs and Limitations

**Files:**
- Modify: `README.md`, `atlas/backtest/runner.py` (`LIMITATIONS`), `atlas/config/run.example.yaml`
- Test: `tests/test_docs_consistency.py`

**Interfaces:**
- Produces: README sections for `spoke_model`, per-agent `model`, the cutoff table (what it is, that it is month-granular and copied from the Models overview page, how to re-check it), the validation-days requirement for autoresearch, `atlas errors`; example YAML shows `spoke_model: claude-haiku-5-5` commented; `LIMITATIONS` gains the cutoff-table caveat and the probe-covers-one-model note.

- [ ] **Step 1: Write the failing tests**: `test_every_documented_cli_flag_exists` (parse the README code blocks, run each `--help`); `test_limitations_mention_cutoff_table_and_probe_scope`; `test_run_example_yaml_loads`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q` (whole suite, about 4 minutes). Expected: PASS.
- [ ] **Step 5: Commit** `docs: document the new features and their limits`.

---

## Self-review notes

- **Spec coverage:** error log 2.1 (A1-A4), hub-and-spoke 2.2 (B1-B6 incl. cutoff table, budget, estimate, report, checkpoint), confirmation and abstention accounting 2.3 (C1-C4). The spec's optional probe flag is B5.
- **Type consistency:** `ErrorLog` (A1) is the only logger type used by A2-A4; `Budget` (B2) is what `CachedLLM(budget=...)` takes in B3-B4; `DayPairs` (C2) is what the CLI scorer returns in C2/C4; `Settings` fields added in C1 and B1 do not overlap.
- **Order and independence:** A, C and B do not depend on each other except that B4 touches `build_run` and `cmd_report` after A3/A4 did, so build in order A, C, B; each part's final task leaves the suite green.
- **Known follow-ups not planned:** dynamic hub dispatch, portfolio-level edit scoring, error-log shipping/alerting.
- **Operational note:** to use validation confirmation, the validation period must have at least `confirm_min_days` (20) trading days whose exit is inside validation; the private runbook's 11-day validation window must grow to about 30 days (for example train 2026-03-02..2026-05-15, validation 2026-05-18..2026-06-30).
