"""Water MD comparison with a hard wall-time budget (at most three minutes).

The lightweight supervisor starts before scientific imports. Each model gets
an equal share of the remaining budget; workers checkpoint between MD steps.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic


def main():
    started = monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("both", "xreac", "mattersim"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=180)
    args, extra = parser.parse_known_args()
    if not 0 < args.max_seconds <= 180:
        parser.error("max-seconds must be greater than zero and at most 180")
    args.output.mkdir(parents=True, exist_ok=False)
    providers = ["xreac", "mattersim"] if args.provider == "both" else [args.provider]
    deadline = started + args.max_seconds
    report = {"max_seconds": args.max_seconds, "runs": {}}
    failed = False
    for index, provider in enumerate(providers):
        # Reserve one second for final bookkeeping and shutdown.
        budget = max(0, (deadline - monotonic() - 1) / (len(providers) - index))
        folder = args.output / provider
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="",
                   GDPX_BENCHMARK_DEADLINE=str(monotonic() + budget - 0.5))
        command = [sys.executable, str(Path(__file__).with_name("_worker.py")),
                   "--provider", provider, "--output", str(folder), *extra]
        print(f"{provider}: wall-time budget {budget:.1f}s", flush=True)
        entry = {"budget_seconds": budget, "status": "time_limit"}
        if budget > 0:
            process = subprocess.Popen(command, env=env)
            try:
                entry["returncode"] = process.wait(timeout=budget)
                entry["status"] = "complete" if process.returncode == 0 else "failed"
                failed |= process.returncode != 0
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                entry["status"] = "time_limit"
        summary = folder / "summary.json"
        if summary.exists():
            entry["summary"] = json.loads(summary.read_text())
            if entry["summary"].get("time_limit_reached"):
                entry["status"] = "time_limit"
        report["runs"][provider] = entry
        report["elapsed_seconds"] = monotonic() - started
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Finished in {report['elapsed_seconds']:.1f}s; results: {args.output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
