"""The benchmark must bound even blocked scientific imports/calculations."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic


SCRIPT = Path(__file__).parents[2] / "benchmarks/water_md/benchmark.py"


def test_supervisor_kills_blocked_worker_and_keeps_partial_summary(tmp_path):
    script = tmp_path / "benchmark.py"
    shutil.copyfile(SCRIPT, script)
    (tmp_path / "_worker.py").write_text(
        "import sys, time, json\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.mkdir(parents=True)\n"
        "(out / 'summary.json').write_text(json.dumps({'partial': True}))\n"
        "time.sleep(60)\n"
    )
    output = tmp_path / "result"
    start = monotonic()
    result = subprocess.run(
        [sys.executable, str(script), "--provider", "xreac", "--output", str(output),
         "--max-seconds", "2"], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert monotonic() - start < 3
    report = json.loads((output / "comparison.json").read_text())
    assert report["runs"]["xreac"]["status"] == "time_limit"
    assert report["runs"]["xreac"]["summary"] == {"partial": True}


def test_supervisor_rejects_more_than_three_minutes(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "result"),
         "--max-seconds", "181"], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 2
    assert "at most 180" in result.stderr
    assert not (tmp_path / "result").exists()
