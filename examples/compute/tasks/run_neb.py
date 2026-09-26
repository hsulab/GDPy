"""Run one local NEB path through the reactor worker API."""
from pathlib import Path
import yaml
from ase.io import read
from gdpx.execution.factory import create_worker

root = Path(__file__).resolve().parent
config = yaml.safe_load((root / "neb.yaml").read_text())
endpoints = read(root / "endpoints.xyz", ":")
worker = create_worker(config, directory="neb-demo")
worker.run(endpoints)
worker.inspect(endpoints)
if worker.get_number_of_running_jobs() == 0:
    paths = worker.retrieve(include_retrieved=True)
    print(f"Retrieved {len(paths)} path(s); inspect neb-demo for trajectories.")
