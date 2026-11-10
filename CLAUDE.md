# CLAUDE.md: project memory for atlas-gic

Compiled 2026-11-10 from the working sessions so far. Read this first in a new session. Facts are as of that date; check `git log` and the open PR before trusting the status section.

## 1. What this repository is

* The repo root started as a **README-only showcase** of ATLAS (General Intelligence Capital's self-improving multi-agent trading idea). `src/`, `prompts/examples/`, `architecture/` and `results/` are that showcase's reference material. **Do not modify them.** Notably `results/summary.json` shows the original 18-month backtest ending at -5.91%, while the README headlines a +22% deployment phase.
* The new work is a fresh, test-covered Python package **`atlas/`**: a 12-agent LLM **backtest/research system** with honest, leakage-aware evaluation. It has **no live or paper execution** (simulated broker only). It is merged to `main` as PR #1.
* Work branch: `claude/loving-einstein-3w7snu`. Open draft PR #2 (spec + plan + implementation of the follow-ups below): https://github.com/littlepotateo/atlas-gic/pull/2 .

## 2. Layout and how to run

```
atlas/config/*.yaml        settings.yaml, limits.yaml, agents.yaml, run.example.yaml
atlas/prompts/*.md         one static system prompt per agent (+ control.md)
atlas/data/                PriceStore (as-of snapshots, no lookahead), yfinance/FRED(ALFRED)/IBKR(read-only, unverified)/csv providers
atlas/engine/              llm.py (CachedLLM, budget), agent.py (run_agent), pipeline.py (4 layers), failures.py (sqlite FailureLog)
atlas/portfolio/, risk/    Portfolio, SimBroker (next-open fills), deterministic fail-closed RiskLayer
atlas/scoring/             excess-return scoring, rank IC, Darwinian weights
atlas/backtest/            runner (checkpoint/resume), predlog (hash chain), nulltest, control, ablation, report, contamination, oldmodel
atlas/autoresearch/        gitops (prompt-only edits) and loop (paired replay, keep/revert)
atlas/errorlog.py          run-level error log (new, follow-up work)
atlas/cli.py               the `atlas` command (python -m atlas)
tests/                     ~630 tests; golden end-to-end suite in tests/golden (committed synthetic market + scripted stub model)
docs/superpowers/specs|plans/   design docs (two generations, see section 7)
```

* Setup: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`. Python 3.11+.
* Tests: `.venv/bin/python -m pytest -q` (full suite about 4 minutes; `live` tests are skipped unless `--live` and a key are present). Run focused tests while iterating.
* CLI: `atlas backtest|calibrate|autoresearch|report|probe|errors --config/--run-dir ...`. Workflow: `backtest --dry-run`, `backtest --with-control`, `calibrate`, pass 2 in a NEW run_dir with `calibration:` set, `report`, optional `autoresearch` (on a dedicated git branch of a clean checkout) and `probe`. See README "Running the backtest system".
* Keys come **only from the environment** (`ANTHROPIC_API_KEY`; `FRED_API_KEY` if macro is on). Never put keys in files or print them.
* A private local runbook exists at `private/RUNBOOK.md` (git-excluded via `.git/info/exclude`, NOT committed; the container is ephemeral).

## 3. Invariants that must not be broken

* **No lookahead**: snapshots contain only data on or before the as-of day; fills happen at the next trading day's open; scoring uses close(T) to close(T+h); the contamination probe sends only anonymised data; autoresearch uses train-period days only (replay days and their scoring exits inside train).
* **Risk layer is the only path from orders to fills**; it is pure, fail-closed (any exception means no orders), and the kill switch halt is sticky across resumes.
* **Checkpoint is the commit point**: the prediction log is appended and fsynced BEFORE scoring; the log head is stored in the checkpoint; resume drops at most one uncommitted tail line; a run directory is bound to its stored `run_config.json` (changed config is refused).
* **One hard budget per run_dir** covers backtest, control, autoresearch, probe and older-model spend (estimate from configured per-model prices, not billing). The cap is per run directory, not per account: also set an account-level spend limit at Anthropic.
* **Failures**: partial model output is recorded but never scored or traded; transient infrastructure failures PAUSE the run (resumable), content failures skip the day; 400/401/403/404 API errors are fatal and never retried.
* **Secrets/logs**: every string written to logs goes through `sanitise` and control-character escaping; raw model replies live only in `failures.sqlite`.
* **Report honesty**: pre-cutoff periods are labelled NOT EVIDENCE OF SKILL; "Possible evidence of skill" needs seven conditions at once (>=3 eligible agents with holdout picks; median raw holdout IC above the null 95th percentile; mean shrunk holdout IC above the null dead zone; holdout excess vs SPY > 0 over >=60 trading days; control IC present and lower than the real agents' median; null calibrated; model cutoff known). "No evidence of skill beyond the null" is an expected, acceptable result.

## 4. Models, cutoff and cost facts (checked 2026-11-07 against the Anthropic Models overview page and the claude-api skill)

* `Settings.model` is the Claude model that powers the agents (currently `claude-sonnet-5-5`). Do not send `temperature`, `top_p` or `top_k` (current models reject them with HTTP 400); determinism on replay comes from the response cache. `effort: low` and `max_output_tokens: 8192` are the packaged defaults.
* **Training-data cutoff**: the Models overview page (https://platform.claude.com/docs/en/about-claude/models/overview) shows **Jun 2026** for Haiku 5.5, Sonnet 5.5, Opus 5.5 and Fable 5.1. It is month-granular, so use the month end, `model_cutoff: 2026-06-30`. The shipped value `2026-06-01` in `settings.yaml` is a placeholder (the follow-up work replaces it). Re-check the page whenever the model changes.
* Prices per million tokens (input/output): Haiku 5.5 $0.10/$0.50 (the built-in table uses the safer >100K-token rate $0.50/$2.50, set `usd_per_mtok_input/output` both to override), Sonnet 5.5 $2/$10, Opus 5.5 $4/$20, Fable 5.1 $10/$50.
* Measured estimate (12 agents/day, effort low): about 20k input tokens and 7k-22k output tokens per day. Sonnet 5.5 for all agents is about $0.16/day (a 148-day run about $24, does not fit $20). Haiku 5.5 for all agents is about $0.008/day (about $1.2 for 148 days). With Sonnet hub + Haiku spokes about $0.057/day.
* **User's test budget is $20.** Plan: do test runs on `claude-haiku-5-5`, pilot 3 days first and measure, split caps per run_dir (pass 1 about $4, pass 2 about $4), keep the rest for one shorter Sonnet 5.5 run.
* A positive verdict needs a holdout of at least 60 trading days that starts AFTER `model_cutoff`; autoresearch needs at least 40 train days; validation confirmation (follow-up work) needs about 30 validation days.

## 5. Hub and spoke agents (follow-up work, roles by kind)

| Role | Agents | Layer | Model after the follow-up |
|---|---|---|---|
| Spoke | macro_rates, macro_volatility, macro_dollar_commodities | 1 | `spoke_model` (e.g. Haiku 5.5) |
| Spoke | sector_semiconductors, sector_financials, sector_energy, sector_healthcare | 2 | `spoke_model` |
| Spoke | superinvestor_druckenmiller, superinvestor_ackman | 3 | `spoke_model` |
| Hub | cro, cio, execution | 4 | `Settings.model` (e.g. Sonnet 5.5) |
| Control (not one of the 12) | control | 4 | `spoke_model` |

Effective model per agent = `AgentSpec.model` override > `spoke_model` (spoke kinds) > `Settings.model`.

## 6. Decisions and rulings (kept so they are not re-litigated)

Original build (PR #1): sonnet implementers/reviewers with opus for risk-critical tasks and the final reviews; execution agent outputs target weights and code converts to whole shares; transient failures pause vs content failures skip; scoring uses signed conviction and long-frame excess, hit rate on `signed_excess`; weights stay frozen until the null test calibrates a dead zone (so pass 1 is frozen, pass 2 adaptive); the null test must be calibrated at the production window (`rolling_ic_window`); a single `control.md` prompt; anonymised backtest/autoresearch runs are REFUSED (the prompt files name tickers and ask for dates); adjusted-price leakage is accepted and disclosed; an autoresearch weakness (a prompt that selectively abstains inside a day can inflate paired IC) is accepted and disclosed, with the follow-up below as the fix; no `temperature` setting exists.

Follow-up decisions (2026-11-07 to 2026-11-10, user-approved unless noted):
* Build three follow-ups, in order: **A error log**, **C autoresearch validation confirmation + abstention accounting**, **B hub-and-spoke models**; use the **subagent-driven** method (fresh implementer per task, independent review after each, fix loop, scoped re-review, final whole-branch review on the strongest model).
* **The contamination probe covers every model in use** (hub, spoke and overrides) on the same anonymised snapshots; `--model` restricts it.
* Defaults the user did not override: control agent uses the spoke model; validation margin = `keep_margin * max(1, sqrt(rolling_ic_window / val.new_pairs))`; `confirm_min_days` 20; `min_pair_ratio` 0.95; `impute_abstentions` true (a ticker only one prompt picked is imputed on the other side as a COPY of that pick, conviction and excess; conviction 0 was rejected because a simulation showed it rewards abstention).
* Existing golden fixtures use an early cutoff and the packaged model name, which the new cutoff-table refusal would reject: switch the fixtures to a stub model name not in the table (with explicit prices) and update `run.example.yaml` to cutoff 2026-06-30 with a holdout after it.
* Error log design: `<run_dir>/errors.log` (dry runs under `dry_run/`), JSON Lines, sanitised, control characters and bidi/zero-width characters escaped, mode 0600, rotation at 5 MB, never raises into the run, one shared instance per run; `atlas errors` is read-only. A logging failure must never change run behaviour; BrokenPipe is quiet only for genuine stdout failures.

## 7. Documents

* Original system: `docs/superpowers/specs/2026-11-06-atlas-backtest-system-design.md` and `docs/superpowers/plans/2026-11-06-atlas-backtest-system.md` (17 tasks, built and merged).
* Follow-ups: `docs/superpowers/specs/2026-11-07-atlas-errorlog-hubspoke-confirm-design.md` and `docs/superpowers/plans/2026-11-07-atlas-errorlog-hubspoke-confirm.md` (14 tasks numbered 1-14: Part A tasks 1-4, Part C tasks 5-8, Part B tasks 9-14).

## 8. Current status of the follow-up build (as of 2026-11-10)

* Tasks 1 (ErrorLog module) and 2 (failure mirroring + runner events) are complete and reviewed, pushed to PR #2.
* Task 3 (CLI wiring + `atlas errors`) is implemented and through one review round; **fix round 2 was in progress** (narrow the BrokenPipeError handling to genuine stdout failures; escape bidi/zero-width characters). Its uncommitted changes may be in the working tree (`atlas/cli.py`, `atlas/errorlog.py`, `tests/golden/test_errorlog_cli.py`). Check `git status`/`git log` before continuing and do not commit another agent's in-progress work.
* Remaining: task 4 (report Errors section + README), tasks 5-8 (Part C), tasks 9-14 (Part B), then a final whole-branch review on the strongest model and a scoped re-review of its fixes.
* Method details: the subagent-driven skill keeps a git-ignored ledger at `.superpowers/sdd/2026-11-07-atlas-errorlog-hubspoke-confirm/progress.md` (task briefs, reports and review diffs live beside it); it is lost if the container is reclaimed, so this section is the durable summary. Model tiers: sonnet by default; opus for task 7 (keep rule), task 10 (shared Budget/locking), task 12 (multi-model CLI construction) and the final review. Implementers commit but never push; the controller pushes after each reviewed task. Review rounds found real defects every time (secrets, resume races, tests that did not pin behaviour), so keep the independent review + mutation-check discipline.
* Deferred minor findings are recorded in the ledger; the notable ones: oldmodel.py's FailureLog is not wired to the error log; `cmd_report`'s FailureLog and early probe errors are not logged; the null test ignores within-day clustering (disclosed, anti-conservative); kept autoresearch edits show "holdout IC not computed" (known spec gap).

## 9. What the user must do before real keys

* Set `model_cutoff` from the model's documented training cutoff (see section 4) and keep the holdout strictly after it.
* Run the 3-day pilot on the chosen model first (the packaged live smoke test only covers `claude-sonnet-5-5`): `pytest --live tests/engine/test_llm_live.py`.
* Set an account-level spend limit at Anthropic; the in-code cap is an estimate.
* Review the ticker universe (current constituents, survivorship-biased), `limits.yaml` (conservative placeholders), `min_valid_agents`, `budget_usd`, `confirm_cost_usd`.
* Run `backtest --with-control` (otherwise the report can never say "Possible evidence of skill"), then `calibrate`, then pass 2 in a new run directory. Run autoresearch on a dedicated branch. One process per run directory.
* Never assume: a pre-cutoff number shows skill; a positive verdict on one short holdout is evidence of future profitability; the simulated fills/costs match a real account; the IBKR provider works (never tested live); the system can trade (it cannot, by design).

## 10. Working agreements with this user

* Answer plainly; give a recommendation rather than a survey; ask only when a decision is genuinely theirs.
* Do not merge without being asked (the user merged PR #1 themselves). New work after a merged PR goes on the same branch name reset to `origin/main`, with a new draft PR.
* Commits end with the attribution trailers the session provides; pull requests are opened as drafts; no model identifier goes into commits, PR text or code.
* The environment's stop hook asks to commit and push uncommitted changes: when an implementer subagent is mid-task, leave its changes alone and push only committed work.
* The user wants private material (the runbook) kept out of git.
