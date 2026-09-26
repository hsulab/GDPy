"""Internal MD worker; launch through benchmark.py for the wall-time limit."""

import argparse
import csv
import json
import os
import platform
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter, monotonic

import numpy as np
from ase import Atoms, units
from ase.build import molecule
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from gdpx.providers import get_provider_manager


def save_report(output, report):
    # A hard timeout must leave the last complete JSON checkpoint readable.
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "summary.json")


def water_box(grid, density, seed):
    """Identical reproducible, randomly oriented grid for both potentials."""
    monomer = molecule("H2O")  # O, H, H order
    monomer.positions -= monomer.positions[0]
    count = grid**3
    # amu / (g cm^-3) to Angstrom^3
    length = (count * monomer.get_masses().sum() * 1.66053906660 / density) ** (1 / 3)
    rng = np.random.default_rng(seed)
    atoms = Atoms(cell=[length] * 3, pbc=True)
    for index in product(range(grid), repeat=3):
        water = monomer.copy()
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        water.positions = water.positions @ rotation + (np.array(index) + 0.5) * length / grid
        atoms += water
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("xreac", "mattersim"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--equil-steps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    args = parser.parse_args()
    if min(args.grid, args.density, args.temperature, args.timestep_fs,
           args.equil_steps, args.steps, args.sample_interval) <= 0 or args.min_steps < 0:
        parser.error("Sizes and physical parameters must be positive; min-steps may be zero.")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = float(os.environ["GDPX_BENCHMARK_DEADLINE"])
    remaining = max(0, deadline - monotonic())
    min_deadline = monotonic() + remaining * 0.2
    warmup_deadline = monotonic() + remaining * 0.4
    start_all = perf_counter()
    atoms = water_box(args.grid, args.density, args.seed)
    write(args.output / "initial.extxyz", atoms)
    parameters = (
        {"model": "bundled:ffield.reax.HO.2015"}
        if args.provider == "xreac" else
        {"model": "MatterSim-v1.0.0-1M", "compute_stress": False}
    )
    start = perf_counter()
    runtime = get_provider_manager().resolve_runtime({
        "schema_version": 3,
        "potential": ({"provider": "reax", "parameters": parameters}
                      if args.provider == "xreac" else
                      {"provider": args.provider, "parameters": parameters}),
        "executor": {"provider": "ase", "method": "spc", "parameters": {}},
    })
    atoms.calc = runtime.materialization.calculator
    load_seconds = perf_counter() - start
    start = perf_counter()
    with BFGS(atoms, logfile=str(args.output / "min.log"), maxstep=0.05) as opt:
        converged = False
        for converged in opt.irun(fmax=0.1, steps=args.min_steps):
            if monotonic() >= min_deadline:
                break
        min_steps = opt.nsteps
    min_seconds = perf_counter() - start
    write(args.output / "relaxed.extxyz", atoms)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature,
                                 force_temp=True, rng=np.random.default_rng(args.seed))
    Stationary(atoms)
    report = {
        "config": {**vars(args), "output": str(args.output)},
        "platform": platform.platform(),
        "versions": {p: version(p) for p in ("ase", "numpy", args.provider)},
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "atoms": len(atoms), "waters": args.grid**3,
        "cell_A": atoms.cell.lengths().tolist(),
        "parameters": parameters, "load_seconds": load_seconds,
        "minimization": {"seconds": min_seconds, "steps": min_steps, "converged": bool(converged)},
        "stages": {},
        "time_limit_reached": min_steps < args.min_steps and not bool(converged),
    }
    save_report(args.output, report)
    for stage, steps in (("nvt", args.equil_steps), ("nve", args.steps)):
        if monotonic() >= deadline:
            report["time_limit_reached"] = True
            break
        stage_deadline = warmup_deadline if stage == "nvt" else deadline
        rows = []
        dyn = (
            Langevin(atoms, args.timestep_fs * units.fs, temperature_K=args.temperature,
                     friction=0.01 / units.fs, rng=np.random.default_rng(args.seed + 1))
            if stage == "nvt" else VelocityVerlet(atoms, args.timestep_fs * units.fs)
        )
        start = perf_counter()
        with Trajectory(args.output / f"{stage}.traj", "w", atoms) as trajectory, \
                (args.output / f"{stage}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "time_fs", "potential_eV", "kinetic_eV", "total_eV",
                             "temperature_K", "max_force_eV_A", "min_OH_A", "max_OH_A", "wall_seconds"))

            def sample():
                epot = float(atoms.get_potential_energy())
                ekin = float(atoms.get_kinetic_energy())
                force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                distances = [atoms.get_distance(i, i + j, mic=True)
                             for i in range(0, len(atoms), 3) for j in (1, 2)]
                row = (dyn.nsteps, dyn.nsteps * args.timestep_fs, epot, ekin, epot + ekin,
                       float(atoms.get_temperature()), force, min(distances), max(distances),
                       perf_counter() - start)
                if not np.isfinite(row).all():
                    raise RuntimeError(f"Nonfinite result in {stage} step {dyn.nsteps}")
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                trajectory.write(atoms)
                if dyn.nsteps % 100 == 0:
                    print(f"{args.provider} {stage} {dyn.nsteps}/{steps} "
                          f"T={row[5]:.1f} K E={row[4]:.6f} eV wall={row[-1]:.1f}s", flush=True)

            dyn.attach(sample, interval=args.sample_interval)
            for _ in dyn.irun(steps):
                if monotonic() >= stage_deadline:
                    break
            if rows[-1][0] != dyn.nsteps:
                sample()
        elapsed = perf_counter() - start
        values = np.array(rows)
        stats = {
            "steps": dyn.nsteps, "requested_steps": steps,
            "duration_fs": dyn.nsteps * args.timestep_fs,
            "seconds": elapsed, "ms_per_step": elapsed / dyn.nsteps * 1000 if dyn.nsteps else None,
            "temperature_mean_K": float(values[:, 5].mean()),
            "temperature_min_K": float(values[:, 5].min()),
            "temperature_max_K": float(values[:, 5].max()),
            "min_OH_A": float(values[:, 7].min()), "max_OH_A": float(values[:, 8].max()),
            "energy_change_meV_atom": float((values[-1, 4] - values[0, 4]) * 1000 / len(atoms)),
            "energy_range_meV_atom": float(np.ptp(values[:, 4]) * 1000 / len(atoms)),
            "energy_slope_meV_atom_ps": float(np.polyfit(values[:, 1] / 1000, values[:, 4], 1)[0] * 1000 / len(atoms)) if len(rows) > 1 else None,
        }
        report["stages"][stage] = stats
        report["time_limit_reached"] |= dyn.nsteps < steps
        report["total_seconds"] = perf_counter() - start_all
        save_report(args.output, report)
        print(json.dumps({stage: stats}), flush=True)
    write(args.output / "final.extxyz", atoms)
    report["total_seconds"] = perf_counter() - start_all
    save_report(args.output, report)


if __name__ == "__main__":
    main()
