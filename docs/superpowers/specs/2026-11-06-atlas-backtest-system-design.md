# Atlas Backtest System: Design Spec

Date: 2026-11-06
Status: Draft for review
Scope: Stages 1-4 of a staged build (engine, scorecard, backtest harness, autoresearch). Paper trading and live execution are separate, later specs.

## 1. Purpose and context

Build a runnable, self-improving multi-agent equity research and trading system modelled on the architecture documented in this repository (4 layers, Darwinian agent weights, an autoresearch loop that rewrites the weakest agent's prompt and keeps or reverts the change). The original implementation is proprietary and absent from this repo, so this is a fresh implementation from the architecture docs and example prompts.

Ultimate goal: trade real capital (US equities, Interactive Brokers, under $10k). This spec covers only the backtestable system. Live trading is gated on forward (paper) evidence, because backtests on pre-cutoff periods cannot prove skill (see section 8).

Reference point: `results/summary.json` in this repo reports the original 18-month backtest ending at -5.91% total return. Nothing here assumes the approach is profitable; the system is built so that "no evidence of skill" is a visible, acceptable outcome.

### Staged roadmap

1. Core engine: data, 4-layer agent pipeline, decision log (this spec)
2. Scorecard and Darwinian weights (this spec)
3. Backtest harness (this spec)
4. Autoresearch loop (this spec)
5. Paper trading (future spec)
6. Live execution via IBKR, gated on stage 5 evidence (future spec)

### Fixed decisions

- Language: Python. Agents: Claude API.
- Daily end-of-day cycle (no intraday).
- US equities and ETFs; long-only in v1 (shorts configurable but off).
- Data: Interactive Brokers plus free sources (FRED/ALFRED for macro, yfinance or similar for prices).
- Broker access sits behind a `Broker` interface; v1 ships only `SimBroker`.

## 2. Agent roster (12 agents)

| Layer | Agents |
|-------|--------|
| 1 Macro (3) | central bank/rates, volatility/risk regime, dollar/commodities |
| 2 Sector (4) | semiconductors, financials, energy, healthcare/biotech |
| 3 Superinvestor (2) | Druckenmiller-style, Ackman-style |
| 4 Decision (3) | CRO (adversarial risk), CIO (final call), autonomous execution |

The deterministic risk layer is code, not an agent. Agents are config-driven (`config/agents.yaml`); adding one means adding a prompt file and a roster entry. Sector benchmarks: SMH, XLF, XLE, XLV; SPY for macro and superinvestor picks.

Healthcare/biotech was chosen as the fourth sector because its drivers (FDA events, trial results, demographics) are largely independent of semiconductors, financials and energy.

## 3. Module layout

```
atlas/
  config/        agents.yaml, limits.yaml, settings
  prompts/       one .md per agent (the artefact autoresearch edits)
  data/          providers (ibkr, fred, yfinance) -> cached PriceStore
  engine/        agent.py, pipeline.py, llm.py (Claude wrapper, retries, cost tracking)
  risk/          deterministic limits; no LLM; not overridable by agents
  portfolio/     positions, cash, P&L; Broker interface + SimBroker
  scoring/       excess-return scoring, rank IC, shrinkage, Darwinian weights
  autoresearch/  worst-agent selection, edit proposal, git branch, paired replay, keep/revert
  backtest/      replay loop, benchmarks, controls, report
  tests/
```

Each unit has one purpose and a narrow interface so it can be understood and tested alone.

## 4. Daily data flow

For simulated day T, in order:

1. **Snapshot.** `PriceStore` builds an as-of-T view: prices/returns through T's close, macro series lagged by real publication delays. No later data (enforced by test).
2. **Layer 1 (parallel).** Macro agents return a view and confidence; a small aggregation step yields a regime: `RISK_ON`, `RISK_OFF` or `NEUTRAL`.
3. **Layer 2 (parallel).** Sector desks receive the regime plus sector data and return long/short picks.
4. **Layer 3 (parallel).** Superinvestors filter sector picks through their philosophy.
5. **Layer 4 (sequential).** CRO attacks proposals. CIO decides target holdings using agent weights. Execution agent proposes sized orders. The **risk layer** clips or rejects them. The portfolio applies fills at T+1 open.
6. **Scoring.** Matured picks are scored (section 5).

### Agent output contract

Every agent returns schema-validated JSON:

```json
{ "agent": "...", "date": "...", "view": "bullish|bearish|neutral",
  "confidence": 0.0,
  "picks": [{"ticker": "...", "direction": "long|short|none",
             "conviction": 0.0, "horizon_days": 5, "rationale": "..."}] }
```

### Execution assumptions

- Fills at next day's open. Commission and slippage are config values, pessimistic by default.
- Long-only: bearish views can be expressed, but the risk layer permits only long positions unless shorts are enabled.
- Position cap suited to a small account (see section 7).

## 5. Scoring and Darwinian weights

### Scoring

- Each pick is scored on **excess return** over a benchmark (SPY, or the sector ETF for sector desks), net of costs.
- **Primary metric: rank IC**, the Spearman correlation between conviction and subsequent excess return over a rolling window of picks. **Hit rate** (excess return > 0) is stored alongside.
- **Horizons:** 1 day is the fast signal used for weight updates. 5 and 10 days are recorded for reporting. Weight updates use non-overlapping samples because overlapping horizons are not independent.
- Agents with fewer than 30 scored picks are "warming up": no penalty, no weight change, not eligible for autoresearch edits.
- Storage: SQLite, one row per pick, with enough detail to re-score later.

### Darwinian weights

- Initial weight 1.0, bounded to [0.3, 2.5].
- Each agent's IC is **shrunk toward the group mean** in proportion to how few picks it has.
- **Dead zone:** a weight changes only when an agent's shrunk IC differs from the group mean by more than a threshold; the step is x1.05 up or x0.95 down. The threshold comes from the null test (section 8), not a guess.
- The CIO receives each agent's view and weight. CIO and execution agent are scored by **ablation**: the CIO's portfolio versus a simple weight-averaged portfolio from the same inputs.

### Null test

Run the full pipeline with each agent's picks randomly shuffled across tickers and days, many times. The spread of IC values is the pure-luck distribution. It sets the dead-zone threshold and the autoresearch keep margin, and is printed in every report. As a test, shuffled picks must yield IC near zero; otherwise there is leakage.

## 6. Autoresearch

- **Selection:** the agent with the lowest shrunk IC among those past warm-up and not flagged for high failure rate.
- **Edit:** one targeted prompt change on a git branch. The loop can modify only prompt files, never code, config or risk limits.
- **Evaluation:** paired replay. Rerun old and new prompt on the same 40+ trading days. Cached inputs mean unchanged agents are free; only the edited agent's calls cost money.
- **Keep** the edit only if the new prompt beats the old on paired IC by more than the null-test margin; otherwise `git revert`. Cancelled or failed evaluations leave `main` unchanged.
- **Holdout:** the final period is never seen by autoresearch. Kept edits are reported on the holdout to show whether they generalise.
- **Budget:** configured caps on edits per period and total cost, so a runaway loop cannot spend unbounded API credit.

## 7. Risk layer

Plain code, limits in `limits.yaml`, applied after the execution agent and before any fill. Fail-closed: if limits cannot be evaluated, no orders are sent.

| Limit | Default |
|-------|---------|
| Max single position | 15% of equity |
| Max gross exposure | 100% (no leverage) |
| Max positions | 8 |
| Min order | drop orders under $50 or with commission above 1% of the order |
| Daily loss limit | no new buys if equity down 3% from prior close |
| Drawdown kill switch | at 15% from peak: flatten to cash, halt until manual restart |
| Direction | long-only unless enabled |

Defaults are conservative guesses to be finalised before any paper or live stage.

## 8. Backtest harness and validity controls

### Harness

- **Replay loop:** per trading day: snapshot, pipeline, risk layer, next-open fill via `SimBroker`, portfolio update, score matured picks.
- **Periods (chronological):** train (autoresearch may edit), validation (threshold tuning), holdout (defined as dates after the configured model's training cutoff). With current models the holdout is short; reports say so.
- **Determinism:** LLM calls cached by hash of model, prompt and inputs; a prompt edit invalidates only that agent's entries. Replays are deterministic because identical inputs are served from the cache, not because of sampling settings: no `temperature` is sent (current models reject sampling parameters with HTTP 400).
- **Cost control:** upfront cost estimate with confirmation above a ceiling; hard budget cap; `--dry-run` with a stub model.
- **Resumability:** daily checkpoints.

### Validity controls

1. The backtest is a **plumbing test**. Pre-cutoff results are labelled "not evidence of skill". Forward paper trading is the evidence stage.
2. **Date-only control agent:** sees the date and ticker list but no market data and picks from memory. Its IC is shown next to the real agents'; a similar IC indicates memorisation.
3. **Input anonymisation** (optional, on by default for pre-cutoff periods): opaque ticker IDs with sector labels, prices rebased to 100, relative day index instead of dates. Partial protection only, since macro levels can still identify an era.
4. **Contamination probe** (optional tooling): ask the model to guess date and ticker from anonymised snapshots; the hit rate becomes a contamination score.
5. **Older-model check** (opportunistic): if a model with an earlier cutoff is available, rerun on a period after that cutoff.
6. **Point-in-time data:** macro via real-time vintages (FRED ALFRED); universe fixed in advance of liquid ETFs and large caps; survivorship bias from missing delisted names is noted in the report.
7. **Append-only prediction log:** each day's picks are written and hashed before outcomes are known so later forward results are verifiable.

### Report

Markdown plus CSV and charts:

- Equity curve versus SPY buy-and-hold and an equal-weight basket of traded names; return, volatility, Sharpe, max drawdown, turnover, all net of costs, per period.
- Per-agent table: picks, hit rate, shrunk IC, weight over time, plus the control agent's IC.
- Autoresearch log: every edit, paired-replay result, kept or reverted, holdout performance.
- Null-test distribution and implied thresholds.
- CIO ablation result; contamination score if run; total API cost and cost per day.
- Failure summary table by agent and reason code.
- A plain verdict line. "No evidence of skill beyond the null" is an acceptable outcome.

## 9. Error handling

### LLM failures

- Every failed call writes a failure row: agent, date, **reason code**, error detail, attempt number, and the **raw response (including any partial output) stored verbatim**.
- Reason codes: `malformed_json`, `schema_invalid` (with failing fields), `empty_response`, `truncated`, `refusal`, `api_error`, `rate_limited`, `timeout`.
- **Partial output is recorded but never used:** not scored, not passed to later layers, not traded. The agent is recorded as `no_opinion` for the day.
- Retry policy: one retry for `malformed_json`, `schema_invalid`, `truncated`, `empty_response`. Transient `api_error`, `rate_limited`, `timeout` use exponential backoff. `refusal` is not retried and is flagged in the report.
- Day-level rule: if fewer than a configured minimum number of agents return valid output, the day is skipped and not traded; the skip is logged with failed agents and reason codes.
- Agents with a high failure rate are flagged and excluded from autoresearch selection until fixed, so a broken agent is not mistaken for a low-skill one.

### Other failures

- **Network/API errors:** repeated failures pause the run and checkpoint; decisions are never made on partial data.
- **Data gaps:** a missing price or macro value is marked unavailable and agents are told; nothing is silently forward-filled in backtests; a ticker with a gap is excluded for the day.
- **Risk layer:** fail-closed.
- **Cost guard:** a run-level budget cap stops runaway spending.

## 10. Testing

- **Unit tests:** score and IC math (hand-computed cases), shrinkage and dead-zone rules, weight bounds, each risk limit, sizing, cost application.
- **No-lookahead tests:** `PriceStore` never returns data dated after T; macro lag and vintage logic checked on known revisions. The most important test in the suite.
- **Pipeline tests with a stub model:** deterministic fake agents verify layer ordering, parallelism, schema validation, failure handling and `no_opinion` paths at zero API cost.
- **Property tests:** weights stay within [0.3, 2.5]; gross exposure never exceeds the limit; cash never goes negative.
- **Golden backtest:** a short fixed backtest with the stub model reproduces identical results run to run, verifying determinism and checkpoint-resume.
- **Null test as a test:** shuffled picks produce IC near zero.
- **Live LLM tests:** few, small, opt-in, marked, so normal runs cost nothing.

## 11. Out of scope for this spec

- Paper trading, IBKR order execution and any live trading.
- Agent spawning, PRISM regime cohorts, JANUS and MiroFish integration (all possible future extensions; `src/janus.py` in this repo is reference material only).
- News sentiment agent (considered and removed).
- Shorting, leverage, options, intraday trading.

## 12. Open items to finalise before paper/live stages

- Final risk-limit values.
- Which model(s) and cutoff date define the holdout.
- Final stock/ETF universe.
- Exact window lengths (rolling IC window, replay length) after the null test is run.
