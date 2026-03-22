# ATLAS AI Investment Committee — System Overview & Build Prompt

> **Purpose:** This document provides full context on the existing ATLAS codebase and the plan to build a fully automated AI investment committee on top of it. Use this in Claude chat to iterate on architecture, agent design, and implementation details.

---

## 1. What Already Exists

ATLAS is a self-improving multi-agent AI trading system. The codebase contains:

### Implemented Code
| File | Lines | What It Does |
|------|-------|-------------|
| `src/janus.py` | 572 | Meta-weighting layer — blends recommendations from multiple trained agent cohorts, applies softmax with constraints (20%–80%), detects market regime (NOVEL/HISTORICAL/MIXED) based on which cohort is more accurate |
| `src/mirofish/mirofish_bridge.py` | 939 | Simulation engine — generates scenarios, runs Claude-based multi-agent simulations (10 rounds, 10+ agent types), models 5 Soros reflexive feedback loops |
| `src/mirofish/mirofish_trainer.py` | 437 | Forward trainer — presents synthetic futures to agents, scores their decisions, updates weights (0.5–2.0 range) |
| `src/mirofish/mirofish_futures_generator.py` | 672 | Synthetic price paths — 5 scenario branches (base/bull/bear/tail_up/tail_down), 7 assets, correlated returns via Cholesky decomposition |
| `src/mirofish/mirofish_seed_generator.py` | 500 | Market data collector — FRED macro, FMP prices, agent debates, portfolio positions, upcoming catalysts |
| `src/mirofish/mirofish_context.py` | 142 | Prompt injection helper — formats MiroFish predictions for agent prompts |

### Documentation
| File | Content |
|------|---------|
| `architecture/overview.md` | 4-layer system diagram, daily execution timeline |
| `architecture/layers.md` | All 25 agents specified with inputs/outputs/data formats |
| `architecture/autoresearch.md` | Self-improvement loop mechanics, scoring methodology, Darwinian weights |
| `prompts/examples/cio.md` | CIO agent prompt template (generic placeholder) |
| `prompts/examples/macro_agent.md` | Macro agent prompt template (generic placeholder) |
| `prompts/examples/sector_desk.md` | Sector desk prompt template |
| `prompts/examples/superinvestor.md` | Superinvestor prompt template |

### Backtest Results
| Metric | Value |
|--------|-------|
| Period | Sep 2024 – Mar 2026 (378 days) |
| Starting capital | $1,000,000 |
| Ending value | $940,937.77 (-5.91%) |
| Deployment phase | +22% in 173 days |
| Autoresearch modifications | 54 attempted, 16 kept (30%) |
| Best pick | AVGO at $152, held for +128% |

### What's NOT Built Yet (What We Need to Create)
- **No orchestrator** — no `eod_cycle.py` or pipeline runner
- **No agent runner** — no code that takes a prompt + data and calls Claude
- **No scoring pipeline** — no code that tracks recommendations vs actual returns
- **No autoresearch loop** — the mechanism is documented but not implemented
- **No config/settings** — API keys, model selection not wired up
- **No data clients** — `MacroClient` and `PriceClient` are imported but not included

---

## 2. The 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: MACRO (10 agents)                  │
│  Central Bank · Geopolitical · China · Dollar · Yield Curve     │
│  Commodities · Volatility · EM · News Sentiment · Inst. Flow    │
│                          ↓                                       │
│              REGIME: RISK_ON / RISK_OFF / NEUTRAL               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: SECTOR DESKS (7 agents)              │
│  Semiconductor · Energy · Biotech · Consumer · Industrials      │
│  Financials · Relationship Mapper                                │
│                          ↓                                       │
│              SECTOR PICKS: Long/Short per sector                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 3: SUPERINVESTORS (4 agents)              │
│  Druckenmiller (macro/mom) · Aschenbrenner (AI/compute)         │
│  Baker (deep tech) · Ackman (quality/FCF)                        │
│                          ↓                                       │
│              FILTERED: Philosophy-aligned picks                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4: DECISION (4 agents)                  │
│  CRO (adversarial) · Alpha Discovery · Autonomous Execution     │
│  CIO (final synthesis, weighted by Darwinian scores)             │
│                          ↓                                       │
│         PORTFOLIO ACTIONS: BUY/SELL/HOLD + position sizing      │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Output Formats

**Layer 1 — Macro Agent Output:**
```json
{
  "signal": "BULLISH | BEARISH | NEUTRAL",
  "conviction": 1-100,
  "rationale": "key reasoning",
  "primary_driver": "what's driving the signal",
  "top_long_theme": "sector or asset class",
  "top_short_theme": "sector or asset class",
  "key_risk": "what could change this view"
}
```

**Layer 2 — Sector Desk Output:**
```json
{
  "sector_view": "OVERWEIGHT | NEUTRAL | UNDERWEIGHT",
  "top_long": {"ticker": "XXXX", "conviction": 80, "thesis": "..."},
  "top_short": {"ticker": "YYYY", "conviction": 60, "thesis": "..."}
}
```

**Layer 3 — Superinvestor Output:**
```json
{
  "portfolio_review": [
    {"ticker": "XXXX", "verdict": "HOLD | ADD | TRIM | EXIT", "rationale": "..."}
  ],
  "missing_name": {"ticker": "YYYY", "thesis": "..."}
}
```

**Layer 4 — CIO Output:**
```json
{
  "market_view": "overall assessment",
  "portfolio_actions": [
    {"ticker": "XXXX", "action": "BUY | SELL | HOLD", "shares": 100, "rationale": "..."}
  ],
  "new_positions": [
    {"ticker": "YYYY", "shares": 50, "thesis": "why entering"}
  ],
  "risk_commentary": "portfolio-level risk assessment",
  "conviction": 1-100
}
```

---

## 3. Three Self-Improvement Mechanisms

### A. Autoresearch Loop (Prompt Evolution)

```
1. Calculate rolling 60-day Sharpe for every agent
2. Select worst performer (lowest Sharpe)
3. Generate ONE targeted prompt modification via Claude
4. Create git feature branch
5. Run for 5 trading days with modified prompt
6. If new_sharpe > old_sharpe → git merge (keep)
7. If not → git reset (revert)
8. Repeat daily
```

**Scoring:**
```python
# Each recommendation is scored:
weighted_return = forward_return * (conviction / 100)
if direction == 'SHORT':
    weighted_return *= -1

# Agent Sharpe = mean(weighted_returns) / std(weighted_returns) * sqrt(252)
```

**Constraint:** Agents cannot be modified more than once per 5 trading days.

### B. Darwinian Weight System

```python
# Daily weight update:
for agent in agents:
    if agent in top_quartile_performers:
        agent.weight = min(2.5, agent.weight * 1.05)
    elif agent in bottom_quartile_performers:
        agent.weight = max(0.3, agent.weight * 0.95)

# CIO uses weighted synthesis:
weighted_signal = sum(agent.rec * agent.weight for agent in agents) / sum(agent.weight)
```

Weight boundaries: 0.3 (near-silent) to 2.5 (maximum influence). Starting: 1.0.

**Key discovery:** The CIO was independently downweighted to 0.3 — the system found its own portfolio manager was the bottleneck before the humans did.

### C. JANUS Meta-Layer (Cohort Blending)

Multiple cohorts trained on different time windows. JANUS:
1. Calculates rolling 30-day accuracy for each cohort
2. Score = 50% hit_rate + 50% normalized_sharpe
3. Applies softmax with constraints (20% min, 80% max per cohort)
4. Blends recommendations — contested tickers get conviction penalty
5. Weight differential between cohorts → emergent regime detection

---

## 4. MiroFish Forward-Looking Training

Agents don't just learn from the past — they train on simulated futures:

```
Seed Generator → collects market state (macro, prices, debates, positions)
       ↓
Scenario Generator → creates 8 branching scenarios
       ↓
Lightweight Simulator → Claude simulates 10+ agent types for 10 rounds
       ↓
   Models 5 reflexive feedback loops:
   1. Price → Fundamentals (drops >15% → downgrades)
   2. P&L → Behavior (drawdown >10% → forced selling)
   3. Narrative → Flows (3+ analyst consensus → retail follows)
   4. Market → Policy (equity drawdown >15% → easing signals)
   5. Reflexive Reversal (5+ rounds same direction → fragility)
       ↓
Futures Generator → synthetic price paths (5 branches, 7 assets, Cholesky correlations)
       ↓
Forward Trainer → presents scenarios to agents, scores decisions, updates weights
```

---

## 5. What We're Building: Full Pipeline Implementation

### Components to Create

```
src/
├── config/
│   └── settings.py              # API keys, model selection, paths
├── data/
│   ├── macro_client.py          # FRED API client (or mock)
│   ├── price_client.py          # FMP/market data client (or mock)
│   └── state/                   # JSON state files (runtime)
├── agents/
│   ├── base_agent.py            # Base class: prompt + Claude call + parse
│   ├── macro/                   # Layer 1: 10 macro agents
│   │   ├── central_bank.py
│   │   ├── geopolitical.py
│   │   ├── china.py
│   │   ├── dollar.py
│   │   ├── yield_curve.py
│   │   ├── commodities.py
│   │   ├── volatility.py
│   │   ├── emerging_markets.py
│   │   ├── news_sentiment.py
│   │   └── institutional_flow.py
│   ├── sector/                  # Layer 2: 7 sector desks
│   │   ├── semiconductor.py
│   │   ├── energy.py
│   │   ├── biotech.py
│   │   ├── consumer.py
│   │   ├── industrials.py
│   │   ├── financials.py
│   │   └── relationship_mapper.py
│   ├── superinvestor/           # Layer 3: 4 superinvestors
│   │   ├── druckenmiller.py
│   │   ├── aschenbrenner.py
│   │   ├── baker.py
│   │   └── ackman.py
│   └── decision/                # Layer 4: 4 decision agents
│       ├── cro.py
│       ├── alpha_discovery.py
│       ├── autonomous_execution.py
│       └── cio.py
├── pipeline/
│   ├── daily_cycle.py           # Full daily cycle orchestrator
│   ├── scoring.py               # Recommendation tracking & scoring
│   ├── darwinian_weights.py     # Weight management (0.3–2.5)
│   └── autoresearch.py          # Self-improvement loop
└── main.py                      # Entry point
```

### Key Design Decisions to Make

1. **Data Sources:** Real APIs (FMP, FRED) vs mock data for testing?
2. **Claude Model:** Haiku for all agents (fast/cheap) vs Sonnet for decision layer?
3. **Parallelism:** Run Layer 1 agents concurrently? (Python async)
4. **State:** JSON files (matching existing pattern) or SQLite?
5. **Agent Prompts:** Inline in Python or separate `.md` files?
6. **Execution Mode:** Paper trading (log-only) vs broker integration?

---

## 6. Backtest Results — Lessons Learned

### Final Darwinian Weights (378-day backtest)

**Top performers (weight = 2.5):**
- Geopolitical, Commodities, Volatility, Energy, Industrials, Aschenbrenner, Ackman

**Bottom performers (weight = 0.3):**
- CIO, Central Bank, Semiconductor, Institutional Flow

### Autoresearch Discoveries (from `results/autoresearch_log.json`)

Most-modified agents and what worked:

| Agent | Modifications | Key Discovery |
|-------|--------------|---------------|
| Emerging Markets | 18 kept of many | DXY-based filter, RSI threshold for shorts, strict asset universe enforcement |
| Financials | 3 kept | Defensive override during sector underperformance, momentum filters |
| Semiconductor | 2 kept | SMH momentum check, high-conviction filter during sector weakness |

### PRISM (All Seasons) Convergent Evolution

Five cohorts trained on different regimes ALL independently discovered:
- Cap conviction during uncertainty
- Use VIX as regime filter
- Enforce hard position limits
- Never override risk management

The same volatility agent grew from 844 bytes to 121,260 bytes in bull markets but only 10,354 bytes in rate tightening — completely different survival strategies.

---

## 7. Open Questions for Discussion

1. **Should the committee operate on real-time data or daily snapshots?**
2. **How should we handle the CIO bottleneck that the backtest revealed?** (Options: better prompt, rules-based execution, split the CIO role)
3. **Should agent spawning be included in v1?** (System creates new agents when it detects knowledge gaps 3+ times in 5 days)
4. **What's the right cadence for autoresearch?** (Daily is aggressive — weekly might be more stable)
5. **How should contested tickers be handled?** (Current: 50% conviction penalty. Alternatives: exclude entirely, reduce position size, flag for human review)
6. **Should we add a human-in-the-loop approval gate before the CIO executes?**

---

## 8. Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| LLM | Claude Sonnet 4 (Anthropic API) | Haiku for training, Sonnet for production |
| Data | FMP + FRED APIs | Or mock data for testing |
| State | JSON files | Matching existing ATLAS pattern |
| Orchestration | Python async | Agents within each layer run in parallel |
| Version Control | Git | Autoresearch tracks prompt evolution |
| Infrastructure | Any machine with Python 3.10+ | $20/month VM sufficient |
| Cost | ~$50-80 per full backtest run | ~$2-5 per daily cycle |

---

## 9. How to Use This Document

**In Claude Chat:**
1. Paste this entire document as context
2. Ask questions like:
   - "Let's design the base agent class — what methods should it have?"
   - "How should we handle the CIO bottleneck? The backtest showed it was downweighted to 0.3."
   - "Design the scoring pipeline — how do we track recommendations vs actual returns?"
   - "What should the autoresearch modification prompt look like?"
   - "How should Layer 1 agents aggregate into a regime signal?"

**In Claude Code (this repo):**
- The implementation work happens here
- Reference this document for architecture decisions
- Build incrementally: base agent → Layer 1 → Layer 2 → Layer 3 → Layer 4 → orchestrator → scoring → autoresearch

---

*This document describes the baseline system. As we build and iterate, it should be updated to reflect design decisions and implementation progress.*
