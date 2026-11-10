"""`atlas probe` covers EVERY model in use (hub, spoke, per-agent overrides) on the SAME anonymised
snapshots, under the run's one cap. Offline: recording stub clients, one per model."""
import json
import re
import shutil
import threading

import pytest

from atlas.cli import main, spend_by_model, spend_so_far, load_stored_run, OperationalError
from atlas.engine.llm import LLMResponse
from tests.golden.test_golden_backtest import (  # noqa: F401 (fixtures)
    FAKE_ANTHROPIC, cli, golden,
)
from tests.golden.test_hub_spoke_cli import (  # noqa: F401 (fixtures)
    HUB, HUB_P, SPOKE, SPOKE_P, mixed, repo,
)

WRONG = json.dumps({"date": "1990-01-01", "ticker": "ZZZZ"})
REAL_TICKERS = ("SPY", "SMH", "XLF", "XLE", "XLV", "AAA", "BBB", "CCC")


class ProbeClient:
    """Records every (system, user, tokens) it serves; reply and token counts are scriptable."""

    def __init__(self, reply=WRONG, tokens=None):
        self.reply, self.tokens = reply, tokens
        self.calls: list[tuple[str, str, int, int]] = []
        self._lock = threading.Lock()

    def complete(self, system, user, max_tokens):
        text = self.reply(user) if callable(self.reply) else self.reply
        i, o = self.tokens or (max(1, (len(system) + len(user)) // 4), max(1, len(text) // 4))
        with self._lock:
            self.calls.append((system, user, i, o))
        return LLMResponse(text, i, o, "end_turn")

    @property
    def users(self):
        return [u for _, u, _, _ in self.calls]

    def cost(self, prices):
        return sum((i * prices[0] + o * prices[1]) / 1e6 for _, _, i, o in self.calls)


def pair(hub=None, spoke=None):
    return {HUB: hub or ProbeClient(), SPOKE: spoke or ProbeClient()}


def probe(run, cl, *extra, samples=6, yes=True):
    argv = ["probe", "--run-dir", str(run), "--samples", str(samples), *extra] + (["--yes"] if yes else [])
    return main(argv, llm_client=cl)


def contamination(run):
    return json.loads((run / "contamination.json").read_text())


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    monkeypatch.delenv("FRED_API_KEY", raising=False)


@pytest.fixture
def mrun(mixed, tmp_path):  # a copy of the mixed (hub + spoke) run
    dest = tmp_path / "run"
    shutil.copytree(mixed["run_dir"], dest)
    return dest


@pytest.fixture
def srun(golden, tmp_path):  # noqa: F811  a copy of the single-model run
    dest = tmp_path / "srun"
    shutil.copytree(golden["run_dir"], dest)
    return dest


def edit_settings(run, **over):
    p = run / "run_config.json"
    st = json.loads(p.read_text())
    st["settings"].update(over)
    p.write_text(json.dumps(st))


# ---------------------------------------------------------------------------- which models, which snapshots

def test_default_probes_every_distinct_model_on_the_same_snapshots(mrun, keys):
    cl = pair()
    assert probe(mrun, cl) == 0
    assert len(cl[HUB].calls) == 6 and len(cl[SPOKE].calls) == 6
    assert cl[HUB].users == cl[SPOKE].users  # same snapshots, same order (same seed)
    assert len(set(cl[HUB].users)) == 6
    c = contamination(mrun)
    assert list(c["models"]) == [HUB, SPOKE]  # hub first, stable order
    assert c["models"][HUB]["n"] == 6 and c["models"][SPOKE]["n"] == 6


def test_model_flag_restricts_to_one_model(mrun, keys, capsys):
    cl = pair()
    assert probe(mrun, cl, "--model", SPOKE) == 0
    assert cl[HUB].calls == [] and len(cl[SPOKE].calls) == 6
    c = contamination(mrun)
    assert list(c["models"]) == [SPOKE] and c["model"] == SPOKE and c["n"] == 6
    assert c["models_in_use"] == [HUB, SPOKE] and c["probed_models"] == [SPOKE]
    assert json.loads((mrun / "probe.json").read_text())["by_model"].keys() == {SPOKE}


def test_unknown_model_exits_2_before_any_spend(mrun, keys, capsys):
    cl = pair()
    for bad in ("claude-haiku-5-5", "no-such-model"):  # priced but not in use / not in use at all
        assert probe(mrun, cl, "--model", bad) == 2
        err = capsys.readouterr().err
        assert bad in err and HUB in err and SPOKE in err  # names what IS in use
    assert cl[HUB].calls == [] and cl[SPOKE].calls == []
    assert not (mrun / "contamination.json").exists() and not (mrun / "probe.json").exists()


def test_a_model_without_prices_exits_2_before_any_spend(mrun, keys, capsys):
    edit_settings(mrun, spoke_usd_per_mtok_input=None, spoke_usd_per_mtok_output=None)
    cl = pair()
    assert probe(mrun, cl) == 2
    assert "spoke_usd_per_mtok_input" in capsys.readouterr().err
    assert cl[HUB].calls == [] and cl[SPOKE].calls == [] and not (mrun / "probe.json").exists()
    assert probe(mrun, cl, "--model", HUB) == 0  # the unpriced spoke is simply not selected
    assert cl[SPOKE].calls == []


# ---------------------------------------------------------------------------- single-model compatibility

OLD_KEYS = {"n", "date_hits", "ticker_hits", "any_hits", "rate", "date_rate", "ticker_rate", "chance_date_rate",
            "chance_ticker_rate", "n_trading_days", "universe_size", "by_period", "failed", "budget_exhausted",
            "tolerance_days", "seed", "model", "requested_samples", "spent_usd", "samples"}


def test_single_model_run_probes_one_model_and_json_is_backward_compatible(srun, keys):
    assert cli(["probe", "--run-dir", srun, "--samples", "5", "--yes"], lambda s, u: WRONG) == 0
    c = contamination(srun)
    assert OLD_KEYS <= set(c) and set(c) - OLD_KEYS == {"models", "models_in_use", "probed_models"}
    assert list(c["models"]) == ["golden-stub-model"] == c["probed_models"] == c["models_in_use"]
    m = c["models"]["golden-stub-model"]
    assert (m["n"], m["date_hits"], m["ticker_hits"], m["failed"]) == (c["n"], c["date_hits"], c["ticker_hits"],
                                                                      c["failed"])
    assert m["spend_usd"] == pytest.approx(c["spent_usd"]) and m["incomplete"] is False
    probe_json = json.loads((srun / "probe.json").read_text())
    assert set(probe_json) == {"spent_usd", "by_model"}
    assert probe_json["by_model"] == {"golden-stub-model": pytest.approx(probe_json["spent_usd"])}


def test_contamination_json_has_per_model_scores_and_top_level_hub_fields(mrun, keys):
    n_spoke = {"n": 0}

    def spoke_reply(user):  # the first spoke reply is unusable, the rest are a (wrong) valid guess
        n_spoke["n"] += 1
        return "garbage" if n_spoke["n"] == 1 else WRONG

    cl = pair(spoke=ProbeClient(spoke_reply))
    assert probe(mrun, cl) == 0
    c = contamination(mrun)
    hub, spoke = c["models"][HUB], c["models"][SPOKE]
    assert (hub["failed"], spoke["failed"]) == (0, 1)
    assert c["model"] == HUB and c["failed"] == 0 and c["n"] == hub["n"]  # top level = the hub
    assert hub["incomplete"] is False and spoke["incomplete"] is False
    for m in (hub, spoke):
        assert {"n", "date_hits", "ticker_hits", "date_rate", "ticker_rate", "failed", "chance_date",
                "chance_ticker", "spend_usd", "by_period"} <= set(m)
        assert m["chance_date"] == c["chance_date_rate"] and m["chance_ticker"] == c["chance_ticker_rate"]
    assert c["samples"] == [{"date": d["date"], "ticker": d["ticker"]} for d in c["samples"]]


def test_a_rerun_overwrites_cleanly(mrun, keys):
    assert probe(mrun, pair()) == 0
    assert probe(mrun, pair(), "--model", HUB) == 0
    c = contamination(mrun)
    assert list(c["models"]) == [HUB] and c["probed_models"] == [HUB] and c["models_in_use"] == [HUB, SPOKE]


# ---------------------------------------------------------------------------- money

def estimate(settings, samples, prices):
    return samples * (settings["est_input_tokens_per_call"] * prices[0] + 100 * prices[1]) / 1e6


def test_estimate_and_confirmation_cover_all_models(mrun, keys, capsys):
    st = json.loads((mrun / "run_config.json").read_text())["settings"]
    hub_est, spoke_est = estimate(st, 6, HUB_P), estimate(st, 6, SPOKE_P)
    edit_settings(mrun, confirm_cost_usd=hub_est * 1.01)  # the hub alone fits, hub + spoke does not
    cl = pair()
    assert probe(mrun, cl, yes=False) == 2  # stdin is not a terminal -> refused, BEFORE any call
    assert cl[HUB].calls == [] and cl[SPOKE].calls == []
    assert "--yes" in capsys.readouterr().err
    assert probe(mrun, cl, "--model", HUB, yes=False) == 0  # one model: under the ceiling
    assert cl[SPOKE].calls == [] and len(cl[HUB].calls) == 6
    edit_settings(mrun, confirm_cost_usd=(hub_est + spoke_est) * 1.01)
    cl2 = pair()
    assert probe(mrun, cl2, yes=False) == 0  # both fit
    assert len(cl2[SPOKE].calls) == 6


def test_probe_spend_split_by_model_and_counted_once_in_the_cap(mrun, keys, monkeypatch):
    import atlas.cli as c
    budget = json.loads((mrun / "run_config.json").read_text())["settings"]["budget_usd"]
    seen = []
    real = c.CachedLLM

    class Spy(real):
        def __init__(self, client, cache, model, pricing, budget_usd, **k):
            seen.append((model, budget_usd, k.get("budget")))
            super().__init__(client, cache, model, pricing, budget_usd, **k)

    monkeypatch.setattr(c, "CachedLLM", Spy)
    cl = pair()
    before = sum(spend_so_far(mrun).values())
    assert probe(mrun, cl) == 0
    assert [m for m, _, _ in seen] == [HUB, SPOKE]
    assert all(b == pytest.approx(budget - before) for _, b, _ in seen)  # the whole remainder, each
    assert seen[0][2] is seen[1][2] and seen[0][2] is not None  # ... drawn from ONE shared budget
    pj = json.loads((mrun / "probe.json").read_text())
    want = {HUB: cl[HUB].cost(HUB_P), SPOKE: cl[SPOKE].cost(SPOKE_P)}
    assert pj["by_model"] == {m: pytest.approx(v) for m, v in want.items()} and want[HUB] > want[SPOKE] > 0
    assert pj["spent_usd"] == pytest.approx(sum(want.values()))  # the total counted ONCE
    _, settings, _, specs, _ = load_stored_run(mrun, OperationalError)
    by = spend_by_model(mrun, settings, specs)
    assert sum(by.values()) == pytest.approx(sum(spend_so_far(mrun).values()))
    # a re-run's cap is the budget minus EVERYTHING recorded so far (this probe's earlier spend once)
    seen.clear()
    after = sum(spend_so_far(mrun).values())
    assert probe(mrun, pair(), samples=3) == 0
    assert all(b == pytest.approx(budget - after) for _, b, _ in seen)
    # the earlier probe spend is kept (per model) when the probe is re-run with new snapshots
    seen.clear()
    pj_mid = json.loads((mrun / "probe.json").read_text())
    assert probe(mrun, pair(), samples=9) == 0
    pj2 = json.loads((mrun / "probe.json").read_text())
    # the private budget starts from zero: only NEW spend is added to the recorded one, never twice
    assert seen[0][2].spent_usd == pytest.approx(pj2["spent_usd"] - pj_mid["spent_usd"])
    assert "(unattributed)" not in pj2["by_model"]
    assert pj2["spent_usd"] >= pj["spent_usd"] and all(pj2["by_model"][m] >= pj["by_model"][m] for m in want)
    assert sum(pj2["by_model"].values()) == pytest.approx(pj2["spent_usd"])


def test_old_probe_json_without_by_model_is_attributed_to_the_hub(mrun, keys):
    _, settings, _, specs, _ = load_stored_run(mrun, OperationalError)
    base = spend_by_model(mrun, settings, specs)
    (mrun / "probe.json").write_text(json.dumps({"spent_usd": 3.0}))
    by = spend_by_model(mrun, settings, specs)
    assert by[HUB] - base.get(HUB, 0.0) == pytest.approx(3.0) and by.get(SPOKE) == base.get(SPOKE)
    (mrun / "probe.json").write_text(json.dumps({"spent_usd": 3.0, "by_model": {SPOKE: 1.0, HUB: 2.0}}))
    by2 = spend_by_model(mrun, settings, specs)
    assert by2[SPOKE] - by.get(SPOKE, 0) == pytest.approx(1.0) and by2[HUB] - by[HUB] == pytest.approx(-1.0)
    assert sum(by2.values()) == pytest.approx(sum(spend_so_far(mrun).values()))
    (mrun / "probe.json").write_text(json.dumps({"spent_usd": 3.0, "by_model": {HUB: "x"}}))
    with pytest.raises(OperationalError, match="probe.json is corrupt"):
        spend_by_model(mrun, settings, specs)
    (mrun / "probe.json").write_text(json.dumps({"spent_usd": 1.0, "by_model": {HUB: 2.0}}))
    with pytest.raises(OperationalError, match="probe.json is corrupt"):
        spend_by_model(mrun, settings, specs)
    # and a re-run on top of an old file keeps its spend in the hub bucket
    (mrun / "probe.json").write_text(json.dumps({"spent_usd": 3.0}))
    assert probe(mrun, pair(), "--model", SPOKE) == 0
    pj = json.loads((mrun / "probe.json").read_text())
    assert pj["by_model"][HUB] == pytest.approx(3.0) and pj["by_model"][SPOKE] > 0
    assert pj["spent_usd"] == pytest.approx(3.0 + pj["by_model"][SPOKE])


HUGE = (10, 1_000_000)  # tokens of one call: $15 on the hub's prices, $5 on the spoke's


@pytest.mark.parametrize("stop_at", [HUB, SPOKE])
def test_budget_stop_marks_the_current_model_incomplete_and_skips_the_rest_cleanly(mrun, keys, capsys, stop_at):
    expensive = ProbeClient(tokens=HUGE)
    cl = pair(hub=expensive if stop_at == HUB else None, spoke=expensive if stop_at == SPOKE else None)
    left = 40.0 if stop_at == HUB else 12.0
    (mrun / "control.json").write_text(json.dumps(
        {"spent_usd": 50.0 - spend_so_far(mrun)["backtest"] - left, "days": 0}))
    assert probe(mrun, cl) == 0  # the same exit code as a single-model budget stop
    out = capsys.readouterr().out
    assert "budget" in out.lower()
    c = contamination(mrun)
    stopped, rest = c["models"][stop_at], c["models"][SPOKE if stop_at == HUB else HUB]
    assert stopped["incomplete"] is True and stopped["budget_exhausted"] is True and 0 < stopped["n"] < 6
    if stop_at == HUB:  # the spoke was never called and is marked as skipped, not as a clean 0%
        assert cl[SPOKE].calls == [] and rest["skipped"] is True and rest["incomplete"] is True and rest["n"] == 0
    else:
        assert rest["incomplete"] is False and rest["n"] == 6
    assert sum(spend_so_far(mrun).values()) <= 50.0 + 20.0  # at most the overshoot of the last call
    pj = json.loads((mrun / "probe.json").read_text())
    assert pj["spent_usd"] == pytest.approx(sum(pj["by_model"].values())) and pj["spent_usd"] > 0
    assert "Traceback" not in out


# ---------------------------------------------------------------------------- report

def test_report_has_one_row_per_model_and_never_claims_skill(mrun, keys):
    assert probe(mrun, pair()) == 0
    assert main(["report", "--run-dir", str(mrun)]) == 0
    md = (mrun / "report" / "report.md").read_text()
    sec = md.split("## Contamination probe (optional)")[1].split("\n## ")[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("|") and (HUB in ln or SPOKE in ln)]
    assert len(rows) == 2 and HUB in rows[0] and SPOKE in rows[1]
    assert f"Models probed: {HUB}, {SPOKE}" in sec
    assert "not a measure of skill" in sec and "NOT probed" not in sec
    assert "Possible evidence of skill" not in sec and "skill" not in sec.replace("not a measure of skill", "")
    assert "chance (dates)" in sec and "chance (tickers)" in sec and "sector label" in sec


def test_report_warns_when_a_configured_model_was_not_probed(mrun, keys):
    assert probe(mrun, pair(), "--model", HUB) == 0
    assert main(["report", "--run-dir", str(mrun)]) == 0
    sec = (mrun / "report" / "report.md").read_text().split("## Contamination probe (optional)")[1].split("\n## ")[0]
    assert f"Models probed: {HUB}" in sec and f"NOT probed: {SPOKE}" in sec
    assert "does not cover the whole system" in sec


def test_report_marks_incomplete_and_skipped_models(mrun, keys):
    cl = pair(hub=ProbeClient(tokens=HUGE))
    (mrun / "control.json").write_text(json.dumps({"spent_usd": 50.0 - spend_so_far(mrun)["backtest"] - 40.0}))
    assert probe(mrun, cl) == 0
    assert main(["report", "--run-dir", str(mrun)]) == 0
    sec = (mrun / "report" / "report.md").read_text().split("## Contamination probe (optional)")[1].split("\n## ")[0]
    assert "incomplete" in sec and "skipped" in sec and f"NOT probed: {SPOKE}" in sec


def test_report_escapes_model_names_from_the_file(mrun, keys):
    assert probe(mrun, pair()) == 0
    c = contamination(mrun)
    evil = "x|y\x1b[31m‮\n## Verdict"
    c["models"][evil] = c["models"].pop(SPOKE)
    c["probed_models"][1] = evil
    c["models_in_use"][1] = evil
    (mrun / "contamination.json").write_text(json.dumps(c))
    assert main(["report", "--run-dir", str(mrun)]) == 0
    md = (mrun / "report" / "report.md").read_text()
    assert "\x1b" not in md and "‮" not in md
    assert md.count("\n## Verdict") == 1
    sec = md.split("## Contamination probe (optional)")[1].split("\n## ")[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("| ") and "date" not in ln.split("|")[1]]
    assert len({ln.count("|") - ln.count("\\|") for ln in rows if "---" not in ln}) == 1  # equal cell counts


def test_a_malformed_models_entry_degrades_to_unreadable_not_a_crash(mrun, keys):
    assert probe(mrun, pair()) == 0
    c = contamination(mrun)
    c["models"][SPOKE] = "garbage"
    (mrun / "contamination.json").write_text(json.dumps(c))
    assert main(["report", "--run-dir", str(mrun)]) == 0
    assert "unreadable" in (mrun / "report" / "report.md").read_text()


def verdict(md):
    return md.split("## Verdict")[1].split("\n## ")[0]


def test_verdict_section_unchanged_with_probe(mrun, keys):
    assert main(["report", "--run-dir", str(mrun)]) == 0
    before = verdict((mrun / "report" / "report.md").read_text())
    assert probe(mrun, pair()) == 0
    assert main(["report", "--run-dir", str(mrun)]) == 0
    assert verdict((mrun / "report" / "report.md").read_text()) == before


# ---------------------------------------------------------------------------- anonymity

def test_probe_prompts_still_contain_no_real_ticker_or_date_for_every_model(mrun, keys):
    cl = pair()
    assert probe(mrun, cl) == 0
    c = contamination(mrun)
    for m in (HUB, SPOKE):
        assert len(cl[m].calls) == 6
        for system, user, _, _ in cl[m].calls:
            msg = system + "\n" + user
            assert not re.search(r"\d{4}-\d{2}-\d{2}", msg)
            assert not re.search(rf"\b({'|'.join(REAL_TICKERS)})\b", msg)
            assert {i["sector"] for i in json.loads(user)["instruments"]} == {"Unclassified"}
            for s in c["samples"]:
                assert s["date"] not in msg
