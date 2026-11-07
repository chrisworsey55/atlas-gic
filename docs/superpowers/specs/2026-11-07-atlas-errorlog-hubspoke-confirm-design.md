# Atlas follow-ups: run error log, hub-and-spoke models, validation confirmation

Date: 2026-11-07
Status: Draft for review
Builds on: `docs/superpowers/specs/2026-11-06-atlas-backtest-system-design.md` (merged in PR #1).
Scope: three independent changes to the existing `atlas/` package. No live or paper execution is added.

## 1. Why

1. **Error log.** Failed model calls and dropped picks are stored in `failures.sqlite`, and skipped days live in the checkpoint. Run-level events (pauses, fatal API errors, swallowed risk-layer errors, halts, budget stops, corrupt-file errors, unexpected exceptions) only print to the terminal and are lost. A person returning to a paused run has no single place to read what happened.
2. **Hub-and-spoke models.** At the user's $20 test budget, Sonnet 5.5 for all twelve agents costs about $0.16 a day, so a 148-day run (about $24) does not fit. The nine spoke agents (macro, sector, superinvestor) are the bulk of the calls; the three hub agents (CRO, CIO, execution) make the decisions. Routing spokes to a cheaper model and keeping a stronger model at the hub cuts the estimate to about $0.057 a day.
3. **Validation confirmation.** The autoresearch keep rule judges an edit on the same train days whose worst picks were shown to the proposing model. An edit can therefore learn to skip picks like those (selective abstention) and look better in-sample. A simulation with an oracle abstainer inflated paired IC by up to +0.32. The principled fix is to confirm a winning edit on days the proposer never saw, and to make abstention cost something.

## 2. Decisions

### 2.1 Error log

* New `<run_dir>/errors.log` (in `<run_dir>/dry_run/` for dry runs), **append-only JSON Lines**, one event per line, flushed and fsynced per event.
* Event: `ts_utc`, `level` (`info|warning|error`), `kind`, `day`, `agent`, `reason`, `message`, `fields`. Every string goes through the existing `sanitise()` (keys, URLs, bearer tokens) and is truncated; raw model replies are **never** written (they stay in `failures.sqlite`).
* Kinds: `call_failure`, `dropped_pick`, `day_skipped`, `run_paused`, `fatal_api_error`, `risk_error`, `halted`, `budget_exceeded`, `resume_refused`, `data_error`, `operational_error`, `autoresearch_error`, `unexpected_exception`.
* **Logging can never break a run.** Any failure to write (disk full, permissions) degrades to one stderr warning per process and the run continues. The writer is thread-safe because agents run in a `ThreadPoolExecutor`.
* Size cap: rotate at 5 MB to `errors.log.1` (keep one). File mode 0600.
* A tolerant reader (`read_events`) ignores a torn final line, like `read_ar_log`.
* CLI: `main()` records known errors before printing them, and a **catch-all** turns any unexpected exception into one `unexpected_exception` event (sanitised, limited traceback, no locals) plus a one-line `error: ...` message and exit 1. New read-only subcommand `atlas errors --run-dir PATH [--tail N] [--kind K]`.
* Report: a new `## Errors` section (counts by kind, last 10 events, pointer to the file).
* Not changed: exit codes, checkpoint contents, `failures.sqlite` schema.

### 2.2 Hub-and-spoke model routing

* **Roles by kind:** spokes are `macro`, `sector`, `superinvestor`; hub is `cro`, `cio`, `execution`. The `control` agent uses the spoke model (it is compared against the spokes).
* **Config:** `Settings.spoke_model: str | None = None` (default: everything uses `Settings.model`, i.e. today's behaviour), optional `spoke_usd_per_mtok_input/output` pair, and `AgentSpec.model: str | None` as an explicit per-agent override. Effective model = `AgentSpec.model` > `spoke_model` (spoke kinds) > `Settings.model`.
* **One budget across models.** The spend counter, reserved-cost counter, lock and cap move out of `CachedLLM` into a `Budget` object shared by one `CachedLLM` per model (same cache file; the cache key already includes the model). It also keeps spend per model. Existing single-model behaviour and tests are unchanged.
* **Estimate:** `estimate_cost` sums per-agent cost at each agent's model prices.
* **Training cutoff:** a new `MODEL_TRAINING_CUTOFFS` table (from Anthropic's Models overview page, read 2026-11-07: Jun 2026 for Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.1, taken as the month end, 2026-06-30). If any model in use has a table entry later than `Settings.model_cutoff`, the run is refused (exit 2) naming the model and date. The shipped default becomes `2026-06-30`. A model not in the table produces a warning, and the user must set `model_cutoff` for it.
* **Checkpoint/resume:** the per-agent model is part of the stored agents record, and `spoke_model` is part of the stored settings, so changing either on an existing run is refused by the existing check. The checkpoint gains an optional `spent_by_model`.
* **Report:** the Agents table gets a `model` column, the Cost section a per-model line, and Limitations says mixed models mean results cannot be attributed to the hub or the spokes.
* **Not changed:** the contamination probe still probes `Settings.model` only (documented; a `--model` flag is an optional task). Autoresearch proposals use the hub model; replays use each agent's own model.

### 2.3 Validation confirmation and abstention accounting

* A new Settings group: `confirm_on_validation: bool = True`, `confirm_min_days: int = 20`, `min_pair_ratio: float = 0.95` (the hard-coded 0.8 becomes this), `impute_abstentions: bool = True`.
* **Flow:** an edit that passes the failure/pair gate and beats `keep_margin` on train is *not yet merged*. It is replayed on **validation** days (those whose scoring exit is also inside validation, never touching holdout). It is kept only if (a) there are at least `confirm_min_days` such days, (b) the same gate passes, and (c) `val.diff > keep_margin * max(1, sqrt(rolling_ic_window / val.new_pairs))`. The scaling is a heuristic: an IC's null spread shrinks roughly with the square root of the sample, so fewer pairs need a bigger margin. Fewer than `confirm_min_days` days gives decision `unconfirmed` and the edit is not kept (fail safe).
* **No leakage back:** confirmation results never feed target selection or the proposal diagnostics, which stay train-only. A new `check_validation_only` mirrors `check_train_only`. Each confirmation uses up part of the validation set, so every attempt and its confirmation numbers are logged (`EditRecord.confirm`) and the report states how many confirmations were run.
* **Abstention accounting:** `RunOneDay` may return a `DayPairs(tickers, convictions, excess)`; the old 2-tuple still works (no imputation). With imputation on, a ticker one prompt picked and the other dropped enters the other side with conviction 0 and the same excess, so dropping a loser no longer removes it from the sample. The gate's pair ratio is applied to the raw pairs (before imputation), so dropping many picks is still rejected. Approximation: the imputed excess uses the other side's cost-adjusted value (a few basis points).
* **Not changed:** the train keep rule, target selection, git handling, the holdout rule.

## 3. Alternatives considered

* Error log as plain text lines: harder to filter and to render in the report; JSON Lines costs nothing extra.
* Putting the error log into `failures.sqlite`: mixes run-level events with per-call failures and cannot hold events with no agent; kept separate, with call failures mirrored.
* Hub as a dispatcher that calls spokes dynamically: loses repeatable, cached runs; the layered pipeline already makes the CIO the hub. Rejected.
* Per-model separate budgets: would make the $20 cap meaningless; one shared budget.
* Confirming on holdout: burns the only uncontaminated stretch; rejected.
* Portfolio-level P&L scoring for edits: a larger redesign; left as future work.

## 4. Risks and costs

* Mixed models make attribution ambiguous (a good result may come from the hub, the spokes, or their interaction).
* Validation confirmation adds about 25% replay cost per kept-candidate and consumes validation days; with a short validation period every edit becomes `unconfirmed` (so the user must give validation at least about 30 days).
* The cutoff table is month-granular and copied from a web page that can change; the validator is a guard, not a guarantee.
* The error log duplicates information in `failures.sqlite` by design; the sqlite file remains the source for raw replies.

## 5. Out of scope

Dynamic hub dispatch, portfolio-level edit scoring, per-model contamination probes (beyond an optional flag), error-log shipping/alerting, any live or paper trading.
