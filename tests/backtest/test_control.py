import json
import re
from datetime import date

from atlas.backtest.control import run_control_agent
from atlas.engine.failures import FailureLog
from atlas.engine.llm import CachedLLM, StubClient

ASOF = date(2024, 3, 1)


def test_control_prompt_contains_no_prices(tmp_path):
    seen = []

    def fn(system, user):
        seen.append(user)
        return json.dumps({"agent": "control", "date": "2024-03-01", "view": "neutral",
                           "confidence": 0.1, "picks": [
                               {"ticker": "AAA", "direction": "long", "conviction": 0.3, "horizon_days": 5}]})

    llm = CachedLLM(StubClient(fn), tmp_path / "c.db", "m", (1.0, 1.0), 10)
    r = run_control_agent(ASOF, ["BBB", "AAA", "SPY"], "PROMPT", llm, FailureLog(tmp_path / "f.db"))
    assert r.status == "ok" and r.agent == "control" and r.output.picks[0].ticker == "AAA"
    assert len(seen) == 1
    user = json.loads(seen[0])
    assert user == {"asof": "2024-03-01", "inputs": {"universe": ["AAA", "BBB", "SPY"]}}
    assert not re.search(r"\d", seen[0].replace("2024-03-01", ""))
    assert not re.search(r"\d+\.\d+|\$", seen[0])


def test_control_date_only_prompt_file_removed():
    import atlas
    from pathlib import Path
    root = Path(atlas.__file__).parent / "prompts"
    assert (root / "control.md").exists() and not (root / "control_date_only.md").exists()
