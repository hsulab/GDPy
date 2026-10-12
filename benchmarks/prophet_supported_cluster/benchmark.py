"""Time uncached Prophet energy/force calls on the Cu4/alumina fixture."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
from ase.io import read

from gdpx.providers.prophet.manager import ProphetManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 1:
        parser.error("warmups and samples must each be at least 1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fixture = Path(__file__).resolve().parents[2] / "examples/global_optimisation/assets/cu4_alumina111_supported.xyz"
    atoms = read(fixture)
    manager = ProphetManager()
    started = perf_counter()
    manager.register_calculator(dict(model=args.model, device="cpu", use_kernel=False, use_compile=False))
    load_seconds = perf_counter() - started
    atoms.calc = manager.calc
    positions = atoms.positions.copy()
    rng = np.random.default_rng(20261012)
    samples = []
    print(f"Model loaded in {load_seconds:.3f} s", flush=True)
    for index in range(args.warmups + args.samples):
        atoms.positions = positions + rng.normal(scale=0.001, size=positions.shape)
        atoms.calc.reset()
        started = perf_counter()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        elapsed = perf_counter() - started
        assert np.isfinite(energy) and np.isfinite(forces).all()
        samples.append(elapsed)
        print(f"{'warmup' if index < args.warmups else 'sample'} {index + 1}: {elapsed:.3f} s", flush=True)
    result = {
        "model": "Prophet-OAME-MBD",
        "atoms": len(atoms),
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "use_kernel": False,
        "use_compile": False,
        "perturbation_std_angstrom": 0.001,
        "seed": 20261012,
        "load_seconds": load_seconds,
        "warmup_seconds": samples[:args.warmups],
        "evaluation_seconds": samples[args.warmups:],
        "median_ms": float(np.median(samples[args.warmups:]) * 1000),
        "finite_outputs": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("prophet-mlip", "torch", "ase", "e3nn", "torch_geometric", "matscipy")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
