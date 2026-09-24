"""Compare warmed force calls and equal-budget water4 relaxations on CPU.

Run from the repository root with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
and both the reax and mattersim extras installed. Timings are not accuracy
comparisons: the two potentials describe different energy surfaces.
"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from ase.optimize import BFGS

from gdpx.providers import get_provider_manager
from gdpx.structures.builders.random_structure import RandomStructureImprovedModifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
    recipe = yaml.safe_load((root / "explorations/genetic_algorithm/water4.yaml").read_text())
    settings = dict(recipe["population"]["builders"]["random"])
    settings.pop("method")
    frames = RandomStructureImprovedModifier(
        **settings, random_seed=recipe["random_seed"]
    ).run(size=4)
    assert len(frames) == 4
    report = {
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("xreac", "mattersim", "ase", "numpy")},
        "seed": recipe["random_seed"],
        "atoms": len(frames[0]),
        "results": {},
    }
    for provider in ("xreac", "mattersim"):
        config = yaml.safe_load((root / f"runtimes/{provider}.yaml").read_text())
        start = perf_counter()
        runtime = get_provider_manager().resolve_runtime(config)
        calc = runtime.materialization.calculator
        load_seconds = perf_counter() - start
        atoms = frames[0].copy()
        atoms.calc = calc
        atoms.get_forces()  # Warm up before timing, never time cached results.
        calls = []
        for i in range(5):
            atoms.positions[0, 0] += 0.001
            start = perf_counter()
            forces = atoms.get_forces()
            atoms.get_potential_energy()
            calls.append(perf_counter() - start)
            assert np.isfinite(forces).all()
        relaxations = []
        for frame in frames:
            atoms = frame.copy()
            atoms.calc = calc
            calc.reset()
            start = perf_counter()
            with BFGS(atoms, logfile=None) as opt:
                converged = opt.run(fmax=0.05, steps=20)
                elapsed = perf_counter() - start
                relaxations.append({
                    "seconds": elapsed, "steps": opt.nsteps,
                    "converged": bool(converged),
                    "energy_eV": float(atoms.get_potential_energy()),
                    "fmax_eV_A": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                })
        report["results"][provider] = {
            "load_seconds": load_seconds,
            "force_call_seconds": calls,
            "median_force_seconds": float(np.median(calls)),
            "relaxations": relaxations,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(provider, json.dumps(report["results"][provider]), flush=True)


if __name__ == "__main__":
    main()
