# ATLAS - Self-Improving AI Trading Agents

**Built by [General Intelligence Capital](https://generalintelligencecapital.com)**

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) + [Soros's reflexivity](https://en.wikipedia.org/wiki/Reflexivity_(social_theory)) + [MiroFish swarm simulation](https://github.com/666ghj/MiroFish) applied to financial markets.

The agent prompts are the weights. Sharpe ratio is the loss function. No GPU needed.

⭐ 944 stars · 202 forks

---

## What Is This?

ATLAS is a framework for autonomous AI trading agents that improve their own prompts through market feedback, train on different market regimes, create new agents when they detect knowledge gaps, and simulate reflexive futures to prepare for what's coming.

25+ agents debate markets daily across 4 layers. Every recommendation is scored against real outcomes. The worst-performing agent gets its prompt rewritten. If performance improves, the git commit survives. If not, git revert.

Now running live with real capital.

---

## Architecture

### Layer 1 - Macro (10 agents)

Central bank, geopolitical, China, dollar, yield curve, commodities, volatility, emerging markets, news sentiment, institutional flow.

These agents set the regime. Risk on or risk off? What's the macro backdrop?

### Layer 2 - Sector Desks (7 agents)

Semiconductor, energy, biotech, consumer, industrials, financials, plus a Bloomberg-style relationship mapper that tracks supply chains, ownership, analyst coverage, and competitive dynamics.

### Layer 3 - Superinvestors (4 agents)

* **Druckenmiller** - macro/momentum: what's the big asymmetric trade?
* **Aschenbrenner** - AI/compute: who benefits from the capex cycle?
* **Baker** - deep tech/biotech: who has real IP moats?
* **Ackman** - quality compounder: pricing power + FCF + catalyst?

### Layer 4 - Decision (4 agents)

* **CRO** - adversarial risk officer: attacks every idea, finds correlated risks
* **Alpha Discovery** - finds names nobody else mentioned
* **Autonomous Execution** - converts signals to sized trades
* **CIO** - synthesises all prior layers, weighted by Darwinian agent scores, makes the final call

---

## The Autoresearch Loop

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Same pattern, different domain.

* System identifies worst agent by rolling Sharpe
* Generates one targeted prompt modification
* Runs for 5 trading days
* Checks if agent's Sharpe improved
* Keep (git commit) or revert (git reset)

The agent prompts are the weights being optimised. Each trading day is one training iteration. A $20/month VM replaces the H100.

**Darwinian Weights:** Each agent has a weight between 0.3 (minimum) and 2.5 (maximum). Top quartile agents get weight × 1.05 daily. Bottom quartile get × 0.95. The CIO proportionally weights input by these scores. Good agents get louder. Bad agents get quieter.

---

## 18-Month Backtest Results

**Period:** September 2024 - March 2026 (378 trading days)

* Prompt modifications attempted: 54
* Survived (kept): 16 (30%)
* Reverted: 37 (70%)
* Deployment phase return: **+22% in 173 days**
* Best individual pick: **AVGO at $152, held for +128%**

The system independently discovered its own portfolio manager (CIO) was its weakest component — downweighting it to minimum before we diagnosed the same issue manually.

### Equity Curve

![Equity Curve](results/equity_curve.png)

---

## Agent Spawning

The system detects recurring knowledge gaps in its own debates. When the same blind spot appears 3+ times in 5 days, it autonomously creates a new specialist agent at neutral weight.

During a 6-month spawning test (Jul-Dec 2024):
- 9 agents spawned autonomously — credit markets, earnings calendar, options flow, liquidity conditions, positioning data, earnings guidance, retail sentiment, technical levels
- 3 went extinct (stuck at minimum weight for 20+ days)
- 6 survived Darwinian selection and reached maximum weight
- Zero human involvement in deciding what to create or when

The system grew its own team from 25 to 31 agents based on what it learned it didn't know.

---

## All Seasons (PRISM) — Regime-Specific Training

We don't train one set of agents and hope they work everywhere. We train separate cohorts on distinct market regimes.

Same starting agents. Same vanilla prompts. Different evolutionary environments. Completely different survival instincts.

| Cohort | Period | Return | Modifications Kept | Key Learning |
|--------|--------|--------|--------------------|--------------|
| Bull/Low Vol | 2016-2018 | +7.7% | 180/509 (35%) | Exit vol longs when events resolve peacefully |
| Crisis (COVID) | 2020 Q1-Q2 | -13.1% | 0/3 (0%) | Crashes too fast for autoresearch — agents need to arrive pre-trained |
| Rate Tightening | 2022-2023 | -30.2% | 38/89 (43%) | Don't flip-flop during Fed weeks — 15-day minimum between reversals |
| Recovery | 2020 Q2-Q4 | -29.0% | 0/1 (0%) | Same problem as crisis — too fast for the feedback loop |
| Euphoria | 2021 | +14.3% | 119/243 (49%) | Momentum confirmation before shorts, cap conviction during political crises |

**Convergent Evolution:** All five cohorts independently discovered the same meta-rules — cap conviction, use VIX as regime filter, enforce hard position limits, never override risk management. Nobody programmed caution. Every cohort learned it from losing money when overconfident.

**Divergent Evolution:** The same volatility agent started at 844 bytes in every cohort:
- Bull markets: grew to 121,260 bytes (143x). Learned "exit vol longs immediately when events resolve peacefully."
- Rate tightening: grew to 10,354 bytes. Learned "NEVER buy VXX when VIX is 15-25."
- Euphoria: grew to 1,998 bytes. Learned "require VIX above 30 before going long vol."

Same agent. Same starting prompt. Three completely different survival strategies shaped by three different markets.

---

## JANUS Meta-Layer

Multiple trained cohorts produce different recommendations. JANUS sits above all cohorts and algorithmically weights them by recent accuracy.

The weight differential between cohorts is an emergent regime detector:
- When short-window agents outperform → **NOVEL REGIME**
- When long-window agents outperform → **HISTORICAL REGIME**
- When they're roughly equal → **MIXED**

We didn't build a regime detector. It emerged from tracking which cohort gets things right.

---

## Soros Reflexivity Engine

Markets don't just reflect reality — they change it. We built reflexive feedback loops into the simulation framework.

Five feedback loops modelled:
1. **Price → Fundamentals**: Stock drops >15% trigger credit downgrades, talent flight, capex cuts. Rises >20% trigger cheap capital, talent attraction, customer confidence.
2. **P&L → Behaviour**: Fund drawdown >10% → forced selling cascade. Gains >15% → increased position sizes and concentrated bets.
3. **Narrative → Flows**: 3+ analysts converge on thesis → retail flow follows. Contrarian narratives emerge after extended consensus.
4. **Market → Policy**: Equity drawdown >15% → central bank signals easing. Oil >$130 → strategic reserve releases.
5. **Reflexive Reversal Detection**: Feedback loop running 5+ rounds in one direction → flagged as reflexive extreme. Maximum consensus = maximum fragility.

First detection: Gold bullish consensus appeared in 4 of 5 simulation rounds — flagged as a crowded trade with 32% reversal probability. That's Soros in code.

---

## MiroFish Swarm Simulation Integration

Integrated with [MiroFish](https://github.com/666ghj/MiroFish), a swarm intelligence engine that generates parallel digital worlds populated by thousands of AI agents.

Our trading agents don't just learn from the past — they train on simulated futures:

- Overnight, the system generates branching scenarios (geopolitical escalation, Fed policy, earnings shocks, black swans)
- Thousands of simulated agents (fund managers, central bankers, retail traders, corporate executives) interact with reflexive feedback
- ATLAS trading agents are trained inside these simulated futures using the same Darwinian loop
- Agents that navigate simulated futures well get upweighted
- Predictions scored against actual outcomes to improve simulation accuracy

**First result:** Druckenmiller-style agent scores 1.0 in simulated crashes but 0.22 in melt-ups. Quality compounder agent is the opposite. The system knows which instincts to trust before the regime arrives — not after.

---

## Key Insight

The orchestration layer matters as much as the intelligence layer.

Individual agents improved measurably through autoresearch. But portfolio returns depend on how agent signals are converted to sized positions. The synthesis/decision layer is the bottleneck. Improving individual agent intelligence without improving orchestration yields diminishing returns.

---

## What's Included

* Framework architecture and pipeline structure
* Autoresearch loop design
* Backtest results and equity curve
* All Seasons (PRISM) methodology and results
* Agent spawning mechanism
* JANUS meta-layer design
* Soros reflexivity engine
* MiroFish integration bridge
* Example placeholder prompts (generic, not trained)

## What's NOT Included

* Trained agent prompts (proprietary — evolutionary products of market feedback)
* PRISM evolved prompts per regime
* CIO active management rules
* Agent scorecard data
* Live portfolio positions
* Darwinian weight values
* MiroFish simulation outputs

The trained prompts are the core IP. A competitor starting today is hundreds of iterations behind. That gap widens every day.

---

## Tech Stack

* **Agents:** Claude Sonnet (Anthropic API)
* **Simulation:** MiroFish swarm engine
* **Data:** FMP, Finnhub, Polygon, FRED
* **Infrastructure:** Azure VM ($20/month)
* **Version Control:** Git feature branches for autoresearch tracking
* **Cost:** ~$50-80 for full 18-month backtest, ~$30 for all five PRISM cohorts

---

## Running the backtest system

This repository also contains `atlas/`, a fresh, test-covered implementation of a 12-agent
backtest harness (it is not the proprietary system described above: prompts are generic). It is
built to be honest about what a backtest can show: point-in-time data, a hash-chained prediction
log written before outcomes are scored, a null test, a date-only control agent and a report that
labels pre-cutoff periods as not evidence of skill.

**Prerequisites**

* Python 3.11+, then `pip install -e ".[dev]"` (adds the `atlas` command; `python -m atlas` also works).
* `ANTHROPIC_API_KEY` in the environment for real runs (`--dry-run` needs no key).
* `FRED_API_KEY` in the environment when the run config uses `data.macro.source: fred`.
* Optional: a running IB Gateway / TWS and `pip install -e ".[ibkr]"` for `data.prices.source: ibkr`
  (the default price source is yfinance; `csv` reads local files).

Keys are read from the environment only: never put them in a config file. The CLI never prints
them, and a missing key stops the command before any work (exit code 2).

**Workflow**

```bash
cp atlas/config/run.example.yaml my_run.yaml        # edit periods, budget, model, universe
atlas backtest --config my_run.yaml --dry-run       # plumbing check: zero LLM calls (state in <run_dir>/dry_run/)
atlas backtest --config my_run.yaml                 # pass 1: real model, frozen weights
atlas calibrate --config my_run.yaml                # null test on train picks -> calibration.json
# pass 2: a NEW run_dir whose config sets `calibration: <pass-1 run_dir>/calibration.json`
atlas backtest --config my_run_pass2.yaml           # adaptive Darwinian weights
atlas report --run-dir runs/pass2                   # report.md, equity.png, agents.csv, autoresearch.csv
atlas autoresearch --config my_run_pass2.yaml       # optional: prompt edits judged on train days only
atlas probe --run-dir runs/pass2 --samples 30       # (optional) contamination probe -> contamination.json
```

* Weights stay frozen until a dead zone is calibrated by the null test (pass 1). The calibrated
  pass 2 must use a new run directory: a checkpoint refuses to resume with changed settings.
* A run estimated above `confirm_cost_usd` asks for confirmation (or pass `--yes`); `budget_usd` is
  a hard cap. A paused run (rate limits, budget) keeps its checkpoint: re-run the same command to
  resume.
* `--with-control` also runs the date-only control agent (one extra call per day); its IC appears in
  the report next to the real agents'.
* `autoresearch` needs a clean git working tree, edits only `atlas/prompts/`, never pushes, and needs
  at least 40 train days. `anonymize: true` is refused for now (the prompt files name tickers).
* `probe` (optional) shows the model anonymised snapshots of the run's own days (opaque ids, rebased
  prices, relative day index, sector labels; never a real ticker or date) and asks it to guess the
  calendar date and real ticker. A high hit rate means the model can recognise those periods, so
  trust their results less; it is not a measure of skill. Its spend counts against `budget_usd`,
  and `atlas report` adds a "Contamination probe (optional)" section once `contamination.json`
  exists. The older-model check (optional, `atlas.backtest.oldmodel.run_older_model_check`) reruns
  only the window after an older model's cutoff in `<run_dir>/older_model/<model>/`; the model
  name is yours to supply and availability is not checked.
* Error log: every run writes `<run_dir>/errors.log` (dry runs under `dry_run/`), one JSON object per
  line (time, level, kind, day, agent, reason, message, fields), sanitised, mode 0600, rotated at
  5 MB to `errors.log.1`. It records failed calls, skipped days, pauses and operational errors, and
  logging can never change or stop a run. Read it with `atlas errors --run-dir DIR` (read-only;
  `--kind` filters) or in the report's "Errors" section ("not provided" when the file is absent).
  It never holds raw model replies: those stay in `failures.sqlite`.
* A run directory remembers its validated config (`run_config.json`): re-running, calibrating or
  researching it with a different config is refused. Unknown config keys are rejected.
* Exit codes: 0 success, 1 operational failure (the message says how to fix it), 2 usage, config
  or key error.
* A request the Anthropic API rejects (HTTP 400/401/403/404, e.g. a wrong model name, key or
  parameter) is not retried: the command stops with `error: Anthropic API rejected the request
  (HTTP ...)`, exit 1, checkpoint intact. Rate limits, timeouts and 5xx errors stay transient
  (retried, then the run pauses). `atlas/config/settings.yaml` sets `effort: low` and
  `max_output_tokens: 8192` (thinking tokens count against it; a `truncated` failure means raise
  it). One live smoke test checks the configured model: `pytest --live tests/engine/test_llm_live.py`.
* `budget_usd` is an ESTIMATE: it counts tokens at per-model prices (a built-in table in
  `atlas/config.py`, or `usd_per_mtok_input`/`usd_per_mtok_output` in the run config), not what
  Anthropic bills. A model the table does not know is refused (exit 2) until both prices are set
  (`older_usd_per_mtok_*` for the older-model check). Also set a spend limit on your Anthropic
  account.
* A positive verdict needs the control condition, which exists only with `atlas backtest
  --with-control`: without it the report can never say "Possible evidence of skill".
* `autoresearch` merges kept edits into whatever branch is checked out (it prints the branch name
  first and refuses a detached HEAD): run it from a dedicated branch of a clean checkout.

**Limitations** (also listed in every report)

* `model_cutoff` in `atlas/config/settings.yaml` (2026-06-01) is a PLACEHOLDER GUESS and the
  whole verdict rests on it: set it from the documented training-data cutoff of the exact model
  you use. The CLI prints a warning while it is the shipped default.
* Pricing is configured, not billed: the hard cap is an estimate (see `budget_usd` above).
* No sampling settings are sent (current models reject `temperature`): replays are deterministic
  only because identical inputs are served from the response cache; an uncached re-run may differ.
* Single process only: fixed `.tmp` file names, no locks and a non-thread-safe price memo. Run
  one process per run directory.
* yfinance data can be stale, is adjusted as of today (adjusted-price leakage: later splits and
  dividends are embedded) and may be throttled. The whole run range is fetched once up front and
  a ticker with no data at all stops the run before any LLM spend, but a ticker missing on a
  single day mid-run silently drops out of that day's snapshot.
* The fixed universe of current constituents is survivorship-biased.
* The contamination probe's chance rate is approximate (overstated by about 7/5).
* Looking at the same holdout repeatedly (especially after autoresearch edits and a pass-2 run)
  is multiple testing: each extra look weakens what a pass means.
* The null test permutes convictions across all rows and ignores within-day clustering, so its
  thresholds are anti-conservative when picks cluster by day. It is calibrated on the
  macro/sector/superinvestor agents while the verdict's eligible agents also include the
  decision layer.
* Kept autoresearch edits are reported with "holdout IC not computed" (a known gap versus the
  spec): confirm a kept edit with a new backtest in a new run directory before trusting it.
* No live or paper trading exists (simulated broker only). The IBKR price provider is unverified
  against a live gateway.

A pre-cutoff backtest is a plumbing test, not evidence of skill. Only a holdout after the
model's training cutoff can be, and the report's verdict stays "No evidence of skill beyond the
null" unless that holdout clears every bar. Tests: `pytest -q` (the golden end-to-end suite in
`tests/golden/` runs the CLI on a committed synthetic market with a scripted stub model).

---

## Contact

**Chris Worsey** — CEO & Technical Founder, General Intelligence Capital

[chris@generalintelligencecapital.com](mailto:chris@generalintelligencecapital.com)

[generalintelligencecapital.com](https://generalintelligencecapital.com)
