"""Smoke test: every demo section runs on the real agents and reproduces the frozen submission."""
import contextlib
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data/hidden/Cyber/hidden/records.csv").exists()
                    or not (ROOT / "models/anomaly_detector.joblib").exists(),
                    reason="hidden records or Agent 2 model not present")
def test_demo_all_sections_run_and_match_frozen_submission():
    import demo
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo.main(["all"])
    out = buf.getvalue()
    for title in ("AGENT 1", "AGENT 2", "AGENT 3", "AGENT 4", "AGENT 5", "COMPLETE FIVE-AGENT PIPELINE",
                  "POLICY LIFECYCLE", "RESULTS DASHBOARD"):
        assert title in out
    assert out.count("IDENTICAL") == 3
    assert "[TRIGGER]" in out and "[AFTER EXPIRY]" in out and "Agent 4 skipped" in out
    assert "attack probability of" not in out
    out.encode("ascii")                       # ASCII only: safe on Windows consoles
