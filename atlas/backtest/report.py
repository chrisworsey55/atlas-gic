"""Backtest report: report.md, equity.png, agents.csv, autoresearch.csv.

Everything here is descriptive and conservative. Pre-cutoff periods are labelled as not
evidence of skill; the verdict says 'No evidence of skill beyond the null' unless the holdout
(the only stretch after the model's training cutoff) clears every bar. Equity already includes
trading costs, so all strategy metrics are net of costs.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atlas.autoresearch.loop import EditRecord  # noqa: E402
from atlas.backtest.nulltest import NullDistribution  # noqa: E402
from atlas.backtest.runner import BacktestResult  # noqa: E402
from atlas.config import Settings  # noqa: E402
from atlas.scoring.metrics import hit_rate, rank_ic  # noqa: E402
from atlas.scoring.store import ScoreStore  # noqa: E402
from atlas.scoring.weights import shrunk_ic  # noqa: E402

NOT_EVIDENCE = "NOT EVIDENCE OF SKILL (pre-cutoff: the model may have seen these dates)"
HOLDOUT_NOTE = "the only uncontaminated stretch (after the model's training cutoff)"
SHORT_HOLDOUT_DAYS = 60
NO_EVIDENCE = "No evidence of skill beyond the null"
POSSIBLE = "Possible evidence of skill"
DISCLAIMER = "This backtest is not evidence of future profitability."
CONTROL_AGENT = "control"
PAIR_CAVEAT = ("a prompt that abstains selectively inside a day can inflate paired IC; "
               "kept edits must be confirmed on the holdout")
EXTRA_LIMITATIONS = [
    "This backtest is a plumbing test: it checks that the machinery runs end to end.",
    "Pre-cutoff results are not evidence of skill (the model may have seen those dates).",
    "Adjusted-price leakage: provider prices are adjusted as of today, so they embed later "
    "splits and dividends.",
    "Autoresearch pair-dropping caveat: " + PAIR_CAVEAT + ".",
]


# ------------------------------------------------------------------------------ metrics

def _norm(s: pd.Series | None) -> pd.Series:
    """Series indexed by python dates, sorted, NaN-free (empty series if s is None)."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    idx = [pd.Timestamp(i).date() for i in s.index]
    out = pd.Series(np.asarray(s, dtype=float), index=idx).dropna()
    return out[~out.index.duplicated(keep="last")].sort_index()


def sharpe(returns: pd.Series) -> float | None:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 5:
        return None
    sd = float(r.std(ddof=1))
    if not math.isfinite(sd) or sd < 1e-12:
        return None
    return float(r.mean() / sd * math.sqrt(252))


def ann_vol(returns: pd.Series) -> float | None:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * math.sqrt(252))


def max_drawdown(path: pd.Series) -> float:
    """Worst peak-to-trough fall of the path, as a non-positive fraction."""
    v = np.asarray(pd.Series(path, dtype=float).dropna())
    if len(v) == 0:
        return 0.0
    peak = np.maximum.accumulate(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, v / peak - 1.0, 0.0)
    return float(dd.min())


def compute_metrics(series: pd.Series, lo: date, hi: date, prior: float | None) -> dict | None:
    """Metrics for `series` restricted to [lo, hi]. `prior` is the value on the day before the
    period starts (the base for the first return); without it the first in-period value is
    the base. None if no data in the period."""
    s = _norm(series)
    s = s[(s.index >= lo) & (s.index <= hi)]
    if len(s) == 0:
        return None
    path = s if prior is None or not math.isfinite(prior) else pd.concat(
        [pd.Series([float(prior)]), s.reset_index(drop=True)], ignore_index=True)
    vals = np.asarray(path, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pd.Series(vals[1:] / vals[:-1] - 1.0)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    total = float(vals[-1] / vals[0] - 1.0) if vals[0] != 0 else None
    return {"n": len(s), "total_return": total, "vol": ann_vol(rets), "sharpe": sharpe(rets),
            "max_drawdown": max_drawdown(path)}


def _prior(series: pd.Series, lo: date) -> float | None:
    s = _norm(series)
    before = s[s.index < lo]
    return float(before.iloc[-1]) if len(before) else None


# ------------------------------------------------------------------------------ formatting

def _f(x, pct=False, nd=3) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x * 100:.2f}%" if pct else f"{x:.{nd}f}"


def _cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------ periods

def _period_ranges(result: BacktestResult, cutoff: date | None):
    """[(name, lo, hi, contaminated, is_holdout)]. Without configured periods the whole run
    is one period; it counts as pre-cutoff only if it ends on or before the cutoff."""
    if result.periods is not None:
        out = []
        for name, (lo, hi) in result.periods.items():
            pre = cutoff is None or hi <= cutoff  # unknown cutoff: assume contaminated
            out.append((name, lo, hi, pre,
                        cutoff is not None and name == "holdout" and lo > cutoff))
        return out
    eq = _norm(result.equity)
    if len(eq) == 0:
        return []
    lo, hi = eq.index[0], eq.index[-1]
    return [("run", lo, hi, cutoff is None or hi <= cutoff, False)]


def _agent_stats(frame: pd.DataFrame, k: float) -> pd.DataFrame:
    rows = []
    if frame is not None and len(frame):
        f = frame[frame["scored"].astype(bool)]
        for agent, g in f.groupby("agent"):
            ex = g["excess"].astype(float)
            rows.append({"agent": agent, "n_scored": len(g),
                         "hit_rate": hit_rate(g["signed_excess"].astype(float)),
                         "rank_ic": rank_ic(g["conviction"].astype(float), ex)})
    df = pd.DataFrame(rows, columns=["agent", "n_scored", "hit_rate", "rank_ic"])
    if len(df):
        ics = df["rank_ic"].astype(float)
        real = _real(df, "rank_ic")
        gm = float(real.mean()) if len(real) else 0.0
        df["shrunk_ic"] = [shrunk_ic(ic, n, gm, k) if math.isfinite(ic) else float("nan")
                           for ic, n in zip(ics, df["n_scored"])]
    else:
        df["shrunk_ic"] = pd.Series(dtype=float)
    return df


def _real(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df.loc[df["agent"] != CONTROL_AGENT, col].astype(float)
    return s[np.isfinite(s)]


# ------------------------------------------------------------------------------ report

def build_report(result: BacktestResult, scores: ScoreStore, null: NullDistribution | None,
                 edits: list[EditRecord], ablation: dict | None, control_ic: float | None,
                 failures: pd.DataFrame, cost_usd: float, out_dir: Path, *,
                 settings: Settings | None = None,
                 holdout_edit_ics: dict[str, float] | None = None,
                 final_weights: dict[str, float] | None = None,
                 dropped_picks: pd.DataFrame | None = None,
                 turnover: dict[str, float] | None = None,
                 ablation_by_period: dict[str, dict | None] | None = None,
                 null_meta: dict | None = None, null_source: str | None = None,
                 autoresearch_null_meta: list[dict | None] | None = None) -> Path:
    """`ablation_by_period` (optional, period name -> ablation dict or None, plus "all") renders
    the CIO ablation per period with the holdout as the headline; without it the single
    `ablation` dict is shown as before."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = settings.model_cutoff if settings is not None else None
    k = settings.shrinkage_k if settings is not None else 20.0
    failures = failures if failures is not None else pd.DataFrame()
    holdout_edit_ics = holdout_edit_ics or {}

    ranges = _period_ranges(result, cutoff)
    eq, spy, bsk = _norm(result.equity), _norm(result.spy), _norm(result.basket)

    # ---- scores
    try:
        frame = scores.picks_frame(None, 1)
    except Exception:  # empty / missing store must not kill the report
        frame = pd.DataFrame(columns=["agent", "date", "ticker", "direction", "conviction",
                                      "horizon", "excess", "signed_excess", "scored"])
    stats = _agent_stats(frame, k)
    ho = next((r for r in ranges if r[4]), None)
    ho_stats = None
    if ho is not None and len(frame):
        fd = frame[[ho[1] <= pd.Timestamp(d).date() <= ho[2] for d in frame["date"]]]
        ho_stats = _agent_stats(fd, k)
    ho_days = 0
    ho_metrics = None
    if ho is not None:
        ho_metrics = compute_metrics(eq, ho[1], ho[2], _prior(eq, ho[1]))
        ho_days = ho_metrics["n"] if ho_metrics else 0

    # ---- verdict
    warmup = settings.warmup_picks if settings is not None else 0
    ctl_all = ctl_ho = None
    ctl_source = "score store"
    verdict_error = None
    try:
        ctl_all, ctl_ho = _control_ic(stats), _control_ic(ho_stats)
        if ctl_all is None and ctl_ho is None and control_ic is not None:
            ctl_all = ctl_ho = control_ic  # override: the store has no control rows
            ctl_source = "passed-in control_ic (no control rows in the score store)"
        verdict, reasons = _verdict(null, cutoff, ho, ho_days, ho_metrics, spy, ho_stats,
                                    ctl_ho, warmup, k)
    except Exception as e:  # any failure degrades to the conservative verdict
        verdict_error = type(e).__name__
        verdict = NO_EVIDENCE
        reasons = [f"verdict computation failed: {verdict_error}"]

    md: list[str] = ["# Backtest report\n"]
    md += ["## Verdict\n", f"**{verdict}**\n"]
    md += [f"- {r}" for r in reasons] + ["", DISCLAIMER, ""]

    # ---- performance
    md += ["## Performance\n",
           "Net-of-cost figures apply to the strategy (its equity curve already includes "
           "commissions and slippage); SPY and the basket are buy-and-hold without trading "
           "costs. Sharpe = mean/std (ddof=1) of daily returns x sqrt(252).\n"]
    if not ranges:
        md.append("no data\n")
    for name, lo, hi, pre, is_ho in ranges:
        md.append(f"### {name} ({lo} to {hi})\n")
        if pre:
            md.append(f"**{NOT_EVIDENCE}**\n")
        elif is_ho:
            md.append(f"**Holdout: {HOLDOUT_NOTE}.**\n")
            if ho_days < SHORT_HOLDOUT_DAYS:
                md.append(f"**holdout too short for reliable conclusions** "
                          f"({ho_days} trading days < {SHORT_HOLDOUT_DAYS}).\n")
        elif cutoff is not None:
            md.append("This stretch straddles the model cutoff and is not separable: treat it "
                      "as contaminated.\n")
        rows = []
        for label, ser in (("Strategy", eq), ("SPY buy-and-hold", spy),
                           ("Equal-weight basket", bsk)):
            m = compute_metrics(ser, lo, hi, _prior(ser, lo))
            tov = (turnover or {}).get(name) if label == "Strategy" else None
            if m is None:
                rows.append({"series": label, "days": 0, "total return": "no data",
                             "ann. vol": "no data", "Sharpe": "no data",
                             "max drawdown": "no data", "turnover": "not provided" if turnover is None else "n/a"})
            else:
                rows.append({"series": label, "days": m["n"],
                             "total return": _f(m["total_return"], pct=True),
                             "ann. vol": _f(m["vol"], pct=True), "Sharpe": _f(m["sharpe"], nd=2),
                             "max drawdown": _f(m["max_drawdown"], pct=True),
                             "turnover": ("not provided" if turnover is None else _f(tov, nd=2))})
        md += [_table(pd.DataFrame(rows))]

    # ---- agents
    agents = sorted(set(stats["agent"]) | (set(failures["agent"]) if "agent" in failures else set()))
    fcount = (failures.groupby("agent")["count"].sum().to_dict()
              if len(failures) and {"agent", "count"} <= set(failures.columns) else {})
    arows = []
    for a in agents:
        s = stats[stats["agent"] == a]
        arows.append({"agent": a,
                      "n_scored": int(s["n_scored"].iloc[0]) if len(s) else 0,
                      "hit_rate": float(s["hit_rate"].iloc[0]) if len(s) else float("nan"),
                      "rank_ic": float(s["rank_ic"].iloc[0]) if len(s) else float("nan"),
                      "shrunk_ic": float(s["shrunk_ic"].iloc[0]) if len(s) else float("nan"),
                      "final_weight": (final_weights.get(a, float("nan")) if final_weights is not None
                                       else float("nan")),
                      "failures": int(fcount.get(a, 0))})
    adf = pd.DataFrame(arows, columns=["agent", "n_scored", "hit_rate", "rank_ic", "shrunk_ic",
                                       "final_weight", "failures"])
    adf.to_csv(out_dir / "agents.csv", index=False)
    md += ["## Agents\n",
           "Control-agent rows are shown but excluded from every real-agent mean/median. "
           "Hit rate = share of picks with signed_excess > 0 (a successful short is a hit); "
           "rank IC = Spearman(conviction, excess); horizon 1 day.\n"]
    disp = adf.copy()
    for c in ("hit_rate", "rank_ic", "shrunk_ic", "final_weight"):
        disp[c] = [_f(x) for x in disp[c]]
    if final_weights is None:
        disp["final_weight"] = "not provided"
    md += [_table(disp)]

    # ---- control
    ic_real = _real(stats, "rank_ic")
    ic_real_ho = _real(ho_stats, "rank_ic")
    md += ["## Control agent\n"]
    if verdict_error is not None:
        md.append(f"Control agent: n/a (verdict computation failed: {verdict_error}).\n")
    elif ctl_all is None and ctl_ho is None:
        md.append("Control agent: not run.\n")
    else:
        def _mm(x):
            return (f"mean {_f(float(x.mean()))}, median {_f(float(x.median()))}"
                    if len(x) else "mean n/a, median n/a")
        md += [f"Source: {ctl_source}. Same definition and windows as the real agents "
               "(rank IC of conviction vs excess, horizon 1 day).\n",
               f"- All-period: control IC {_f(ctl_all)}; real agents {_mm(ic_real)}",
               f"- Holdout only (used by the verdict): control IC {_f(ctl_ho)}; real agents "
               f"{_mm(ic_real_ho)}",
               "", "A control agent with no information should sit near zero; if it matches "
               "the real agents, their ICs are not skill.\n"]

    # ---- autoresearch
    erows = []
    for e in edits:
        d = asdict(e)
        d["holdout_ic"] = holdout_edit_ics.get(e.agent) if e.decision == "kept" else None
        erows.append(d)
    edf = pd.DataFrame(erows, columns=[*EditRecord.__dataclass_fields__.keys(), "holdout_ic"])
    edf.to_csv(out_dir / "autoresearch.csv", index=False)
    md += ["## Autoresearch\n"]
    if not edits:
        md.append("No edits were attempted.\n")
    for e in edits:
        line = (f"- {e.agent}: **{e.decision}** ({e.reason}); old IC {_f(e.old_ic)}, new IC "
                f"{_f(e.new_ic)}, diff {_f(e.diff)} vs keep margin {_f(e.keep_margin)}")
        if e.decision == "kept":
            h = holdout_edit_ics.get(e.agent)
            line += (f"; holdout IC {_f(h)}" if h is not None else "; holdout IC not computed")
        md.append(line)
    md.append("")

    # ---- null
    md += ["## Null test\n"]
    if null is None:
        md.append("not calibrated — weights were frozen.\n")
    else:
        def _try(fn):
            try:
                return _f(fn())
            except Exception:
                return "n/a"
        try:
            n_shuf = len(null.ics)
        except Exception:
            n_shuf = "n/a"
        md += [f"- Dead zone: {_try(null.dead_zone)}", f"- Keep margin: {_try(null.keep_margin)}",
               f"- Null |IC| 95th percentile: {_try(lambda: null.quantile(0.95))}",
               f"- Number of shuffles: {n_shuf}"]
        md.append(f"- Calibration: {_meta_text(null_meta)}" + (f" (source: {null_source})" if null_source else ""))
        md.append("")
    if autoresearch_null_meta:
        distinct = []
        for m in autoresearch_null_meta:
            if m not in distinct:
                distinct.append(m)
        md.append("Autoresearch keep margins were calibrated with: "
                  + "; ".join(_meta_text(m, with_margin=True) for m in distinct) + ".")
        if len([m for m in distinct if m is not None]) > 1:
            md.append(f"**WARNING: autoresearch used {len(distinct)} different null-test configurations "
                      "(seed shopping makes a lucky keep margin more likely).**")
        md.append("")

    # ---- ablation
    md += ["## CIO ablation\n"]
    if ablation_by_period is not None:
        md += _ablation_by_period(ablation_by_period, ranges)
    elif ablation is None:
        md.append("not run.\n")
    else:
        try:
            beat = "yes" if ablation.get("cio_beats_baseline") else "no"
            md.append(f"CIO total {_f(ablation.get('cio_total'), pct=True)} vs baseline total "
                      f"{_f(ablation.get('baseline_total'), pct=True)}; CIO beat baseline: "
                      f"{beat}.\n")
        except Exception:
            md.append("n/a (ablation result unreadable).\n")

    # ---- failures
    md += ["## Failures\n", "Failed calls by agent and reason:\n", _table(failures),
           "Dropped picks:\n", (_table(dropped_picks) if dropped_picks is not None else "not provided\n")]

    # ---- cost
    n_days = len(result.day_results) or len(eq)
    per_day = f"{cost_usd / n_days:.4f}" if n_days else "n/a"
    md += ["## Cost\n", f"Total API cost: ${cost_usd:.2f} over {n_days} trading days "
           f"(${per_day} per trading day).\n"]

    # ---- limitations
    md += ["## Limitations\n"]
    md += [f"- {x}" for x in list(result.limitations) + EXTRA_LIMITATIONS]
    md.append("")
    if result.halted:
        md.append(f"- Run HALTED: {result.halt_reason}")
    reasons_skipped = pd.Series(list(result.skipped.values())).value_counts().to_dict()
    md.append(f"- Skipped days: {len(result.skipped)}"
              + (" (" + "; ".join(f"{r}: {n}" for r, n in reasons_skipped.items()) + ")"
                 if reasons_skipped else ""))
    if result.risk_errors:
        md.append(f"- Risk errors: {len(result.risk_errors)} "
                  + "; ".join(f"{d}: {m}" for d, m in sorted(result.risk_errors.items())))
    md.append("")

    _plot(eq, spy, bsk, ranges, out_dir / "equity.png")
    path = out_dir / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def _meta_text(m: dict | None, with_margin: bool = False) -> str:
    if not m:
        return "parameters not recorded"
    out = (f"seed {m.get('seed')}, n_shuffles {m.get('n_shuffles')}, window {m.get('window')}, "
           f"train picks up to {m.get('train_max_date')}")
    if with_margin:
        out += f", keep_margin {_f(m.get('keep_margin'))}"
    return out


def _ablation_line(a: dict | None) -> str:
    if a is None:
        return "not computable (fewer than 2 comparable days)"
    try:
        beat = "yes" if a.get("cio_beats_baseline") else "no"
        return (f"CIO total {_f(a.get('cio_total'), pct=True)} vs baseline total "
                f"{_f(a.get('baseline_total'), pct=True)}; CIO beat baseline: {beat}")
    except Exception:
        return "n/a (ablation result unreadable)"


def _ablation_by_period(by: dict, ranges) -> list[str]:
    ho = next((r for r in ranges if r[4]), None)
    if ho is None:
        out = ["Headline: no holdout (no period after the model's training cutoff).\n"]
    else:
        out = [f"Headline: holdout ({HOLDOUT_NOTE}): {_ablation_line(by.get(ho[0]))}.\n"]
    for name, _lo, _hi, pre, is_ho in ranges:
        label = NOT_EVIDENCE if pre else ("holdout" if is_ho else "contaminated: straddles the model cutoff")
        out.append(f"- {name} [{label}]: {_ablation_line(by.get(name))}")
    if "all" in by:
        clean = bool(ranges) and all(r[4] for r in ranges)
        out.append(f"- all periods [{'holdout only' if clean else NOT_EVIDENCE}]: {_ablation_line(by['all'])}")
    out.append("")
    return out


def _control_ic(stats) -> float | None:
    if stats is None or stats.empty:
        return None
    c = stats[stats["agent"] == CONTROL_AGENT]
    if c.empty:
        return None
    v = float(c["rank_ic"].iloc[0])
    return v if math.isfinite(v) else float("nan")


def _verdict(null, cutoff, ho, ho_days, ho_metrics, spy, ho_stats, control_ic, warmup, k):
    """Possible evidence needs every one of conditions (a)-(g); otherwise no evidence."""
    dz = q95 = None
    if null is not None:
        try:
            dz = null.dead_zone()
            q95 = null.quantile(0.95)
        except ValueError:
            dz = q95 = None
    elig = pd.DataFrame(columns=["rank_ic", "n_scored"])
    if ho_stats is not None and len(ho_stats):
        h = ho_stats[ho_stats["agent"] != CONTROL_AGENT]
        elig = h[(h["n_scored"] >= warmup) & h["rank_ic"].astype(float).apply(math.isfinite)]
    ics = elig["rank_ic"].astype(float)
    med = float(ics.median()) if len(ics) else None
    mean_sh = None
    if len(ics):
        gm = float(ics.mean())
        mean_sh = float(np.mean([shrunk_ic(ic, int(n), gm, k)
                                 for ic, n in zip(ics, elig["n_scored"])]))
    excess = None
    if ho is not None and ho_metrics:
        spy_m = compute_metrics(spy, ho[1], ho[2], _prior(spy, ho[1]))
        if (spy_m and ho_metrics["total_return"] is not None
                and spy_m["total_return"] is not None):
            excess = ho_metrics["total_return"] - spy_m["total_return"]
    ctl_ok = control_ic is not None and math.isfinite(control_ic)
    conds = [
        ("(a) at least 3 eligible agents with holdout picks", len(ics) >= 3,
         f"{len(ics)} eligible"),
        ("(b) median raw holdout IC > null |IC| 95th percentile",
         med is not None and q95 is not None and med > q95, f"median {_f(med) if med is not None else 'no data'} vs {_f(q95)}"),
        ("(c) mean shrunk holdout IC > null dead zone",
         mean_sh is not None and dz is not None and mean_sh > dz, f"mean {_f(mean_sh)} vs {_f(dz)}"),
        ("(d) holdout excess return vs SPY > 0 over >= 60 trading days",
         excess is not None and excess > 0 and ho_days >= SHORT_HOLDOUT_DAYS,
         f"excess {_f(excess, pct=True)} over {ho_days} days"),
        ("(e) control IC below the real agents' holdout median",
         ctl_ok and med is not None and control_ic < med, f"control {_f(control_ic)}"),
        ("(f) null calibrated", dz is not None, "calibrated" if dz is not None else "not calibrated"),
        ("(g) model cutoff known and a holdout after it exists", cutoff is not None and ho is not None,
         "known" if cutoff is not None and ho is not None else "model cutoff unknown or no holdout"),
    ]
    lines = [f"{'PASS' if ok else 'FAIL'} {name}: {detail}" for name, ok, detail in conds]
    failing = [name.split(")")[0] + ")" for name, ok, _ in conds if not ok]
    if not failing:
        return POSSIBLE, ["Rule: a positive verdict needs all of conditions (a)-(g), on holdout-dated picks only."] \
            + lines + ["All bars cleared on a single holdout; suggestive at most, needs replication."]
    return NO_EVIDENCE, ["Rule: a positive verdict needs all of conditions (a)-(g), on "
                         "holdout-dated picks only. Failing: " + ", ".join(failing) + "."] + lines


def _plot(eq, spy, bsk, ranges, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, s in (("Strategy", eq), ("SPY buy-and-hold", spy), ("Equal-weight basket", bsk)):
        if len(s) and s.iloc[0] != 0:
            ax.plot(list(s.index), np.asarray(s / s.iloc[0]), label=label)
    for name, lo, _hi, _pre, _ho in ranges:
        ax.axvline(lo, color="grey", linestyle="--", linewidth=0.8)
        ax.text(lo, 1.0, name, rotation=90, va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylabel("normalised equity (start = 1)")
    ax.set_title("Equity vs benchmarks (net of costs)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
