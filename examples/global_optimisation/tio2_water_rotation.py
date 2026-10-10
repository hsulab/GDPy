"""Compare parallel and hydrogen-bonded two- or three-water seeds on anatase (101).

Run from the repository root. The default Monti2012 parameters are bundled
with xreac 0.10.0. Optimizer convergence and adsorption geometry are reported
separately; the two seeds need not relax to different minima.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "examples/global_optimisation/assets/tio2_water2_seeds.xyz"
MODEL = "ffield.reax.TiOH.Monti2012"


def geometry_report(atoms):
    """Measure intact adsorption, dipole alignment, and donor-H-acceptor geometry."""
    waters = [list(range(index, index+3)) for index in range(192, len(atoms), 3)]
    ti = np.flatnonzero(atoms.numbers[:192] == 22)
    rows, bisectors = [], []
    for oxygen, *hydrogens in waters:
        oh = atoms.get_distances(oxygen, hydrogens, mic=True)
        vectors = atoms.get_distances(oxygen, hydrogens, mic=True, vector=True)
        bisector = vectors.sum(axis=0)
        bisectors.append(bisector / np.linalg.norm(bisector))
        distances = atoms.get_distances(oxygen, ti, mic=True)
        nearest = int(np.argmin(distances))
        rows.append(dict(oh_angstrom=oh.tolist(), nearest_ti=int(ti[nearest]),
                         ti_o_angstrom=float(distances[nearest]),
                         intact=bool(np.all((oh > 0.7) & (oh < 1.25))),
                         adsorbed=bool(distances[nearest] < 2.8)))
    oo_distances = [float(atoms.get_distance(waters[i][0], waters[j][0], mic=True))
                    for i in range(len(waters)) for j in range(i+1, len(waters))]
    hydrogen_bonds = []
    contacts = []
    for donor, acceptor in [(i, j) for i in range(len(waters)) for j in range(len(waters)) if i != j]:
        oxygen = waters[donor][0]
        other = waters[acceptor][0]
        oo = float(atoms.get_distance(oxygen, other, mic=True))
        for hydrogen in waters[donor][1:]:
            hd = atoms.get_distance(hydrogen, oxygen, mic=True, vector=True)
            ha = atoms.get_distance(hydrogen, other, mic=True, vector=True)
            distance = float(np.linalg.norm(ha))
            angle = float(np.rad2deg(np.arccos(np.clip(np.dot(hd, ha) / (np.linalg.norm(hd)*distance), -1, 1))))
            contact = dict(donor=oxygen, hydrogen=hydrogen, acceptor=other,
                           h_o_angstrom=distance, o_h_o_degrees=angle)
            contacts.append(contact)
            if oo <= 3.5 and distance <= 2.5 and angle >= 150:
                hydrogen_bonds.append(contact)
    alignment = float(min(np.clip(np.dot(bisectors[i], bisectors[j]), -1, 1)
                          for i in range(len(waters)) for j in range(i+1, len(waters))))
    return dict(waters=rows, o_o_angstrom=min(oo_distances), o_o_distances_angstrom=oo_distances,
                dipole_cosine=alignment,
                parallel=bool(alignment > np.cos(np.deg2rad(20))),
                hydrogen_bonds=hydrogen_bonds, water_contacts=contacts,
                molecular_adsorption=all(row["intact"] and row["adsorbed"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/tio2-water-rotation")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--waters", type=int, choices=[2, 3], default=2)
    parser.add_argument("--optimizer", choices=["fire", "bfgs"], default="fire")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seed relaxations in the output directory")
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.fmax) or args.fmax <= 0:
        parser.error("steps and fmax must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if not args.build_only:
        from xreac import ForceField
        from xreac.ase import ReaxFFCalculator
        ff = ForceField.bundled(MODEL)
    reports = {}
    final = []
    seeds = SEEDS.with_name(f"tio2_water{args.waters}_seeds.xyz")
    for label, atoms in zip(["parallel", "hydrogen_bonded"], read(seeds, ":")):
        destination = args.output / label
        destination.mkdir(exist_ok=args.resume)
        same_seed = False
        if args.resume and (destination / "initial.xyz").exists():
            previous = read(destination / "initial.xyz")
            same_seed = (np.array_equal(previous.numbers, atoms.numbers) and
                         np.array_equal(previous.positions, atoms.positions) and
                         np.array_equal(previous.cell, atoms.cell))
        write(destination / "initial.xyz", atoms)
        report = dict(initial=geometry_report(atoms))
        if args.build_only:
            reports[label] = report
            continue
        reused = same_seed and (destination / "relaxed.xyz").exists()
        if reused:
            seed_pbc = atoms.pbc.copy()
            atoms = read(destination / "relaxed.xyz")
            atoms.set_pbc(seed_pbc)
        atoms.info["initial_motif"] = atoms.info.pop("motif", label)
        fixed = np.concatenate([constraint.get_indices() for constraint in atoms.constraints])
        fixed_positions = atoms.positions[fixed].copy()
        atoms.calc = ReaxFFCalculator(ff)
        started = time.monotonic()
        print(f"Relaxing {label}: {len(atoms)} atoms, {len(fixed)} fixed", flush=True)
        converged = reused and np.linalg.norm(atoms.get_forces(), axis=1).max() <= args.fmax
        reused_converged = bool(converged)
        if converged:
            lines = (destination / "optimize.log").read_text().splitlines()
            steps = int(lines[-1].split()[1])
        else:
            options = dict(maxstep=0.1, logfile=str(destination / "optimize.log"),
                           trajectory=str(destination / "relaxation.traj"))
            if args.optimizer == "fire":
                options.update(dt=0.05, dtmax=0.3)
            optimizer_class = FIRE if args.optimizer == "fire" else BFGS
            with optimizer_class(atoms, **options) as optimizer:
                converged = optimizer.run(fmax=args.fmax, steps=args.steps)
                steps = optimizer.nsteps
        assert np.array_equal(atoms.positions[fixed], fixed_positions)
        report.update(converged=bool(converged), steps=steps, energy_ev=float(atoms.get_potential_energy()),
                      max_mobile_force_ev_angstrom=float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                      final=geometry_report(atoms), elapsed_seconds=time.monotonic()-started,
                      reused_completed_relaxation=reused_converged)
        write(destination / "relaxed.xyz", atoms)
        reports[label] = report
        final.append(atoms)
        (args.output / "summary.json").write_text(json.dumps(dict(model=MODEL, force_field_sha256=ff.checksum,
                                                                 waters=args.waters, optimizer=args.optimizer,
                                                                 fmax_ev_angstrom=args.fmax, seeds=reports), indent=2)+"\n")
        print(f"{label}: converged={converged}, E={report['energy_ev']:.8f} eV, "
              f"H-bonds={len(report['final']['hydrogen_bonds'])}, parallel={report['final']['parallel']}", flush=True)
    result = dict(model=MODEL, waters=args.waters, optimizer=args.optimizer, fmax_ev_angstrom=args.fmax, seeds=reports)
    if not args.build_only:
        result["force_field_sha256"] = ff.checksum
        result["hydrogen_bonded_minus_parallel_ev"] = reports["hydrogen_bonded"]["energy_ev"]-reports["parallel"]["energy_ev"]
        result["both_converged"] = all(report["converged"] for report in reports.values())
        result["both_molecularly_adsorbed"] = all(report["final"]["molecular_adsorption"] for report in reports.values())
        result["two_requested_motifs_survive"] = bool(
            result["both_converged"] and result["both_molecularly_adsorbed"] and
            reports["parallel"]["final"]["parallel"] and not reports["parallel"]["final"]["hydrogen_bonds"] and
            reports["hydrogen_bonded"]["final"]["hydrogen_bonds"])
        write(args.output / "relaxed_pair.xyz", final)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Results: {args.output}", flush=True)
    if not args.build_only and not result["both_converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
