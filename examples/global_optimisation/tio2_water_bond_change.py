"""Build and relax molecular/dissociated TiO2 water seeds with bond_change."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

from gdpx.exploration.sampling import parse_operators
from gdpx.exploration.sampling.fragments import FragmentPolicy
from gdpx.exploration.sampling.geometry import prepare_operators
from gdpx.utils.atoms_tags import get_tags_per_species

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ffield.reax.TiOH.Monti2012"
CUTOFFS = {"O-H": 1.25, "Ti-H": 1.25}
POLICY = FragmentPolicy(bond_cutoffs=CUTOFFS)


def adsorption_report(atoms):
    """Classify each mobile H by its nearest bonded oxygen, without imposing bonds."""
    tags = atoms.get_tags()
    oxygens = np.flatnonzero(atoms.numbers == 8)
    hydrogen_sites = []
    counts = {int(i): 0 for i in oxygens if tags[i] > 0}
    for hydrogen in np.flatnonzero((atoms.numbers == 1) & (tags > 0)):
        distances = atoms.get_distances(hydrogen, oxygens, mic=True)
        nearest = int(np.argmin(distances))
        oxygen, distance = int(oxygens[nearest]), float(distances[nearest])
        kind = "free" if distance > 1.25 else ("surface" if tags[oxygen] == 0 else "adsorbate")
        if kind == "adsorbate":
            counts[oxygen] += 1
        hydrogen_sites.append(dict(hydrogen=int(hydrogen), oxygen=oxygen, distance_angstrom=distance, kind=kind))
    ti = np.flatnonzero(atoms.numbers == 22)
    ti_o = {str(oxygen): float(atoms.get_distances(oxygen, ti, mic=True).min()) for oxygen in counts}
    return dict(molecular_waters=sum(count == 2 for count in counts.values()),
                hydroxyls=sum(count == 1 for count in counts.values()),
                surface_hydrogens=sum(site["kind"] == "surface" for site in hydrogen_sites),
                free_hydrogens=sum(site["kind"] == "free" for site in hydrogen_sites),
                hydrogens_per_adsorbate_oxygen=counts, hydrogen_sites=hydrogen_sites,
                ti_o_angstrom=ti_o, all_oxygens_adsorbed=all(distance < 2.8 for distance in ti_o.values()),
                fragments={species: len(groups) for species, groups in get_tags_per_species(atoms).items()
                           if any(tag > 0 for tag, _ in groups)})


def build_seeds(waters=2):
    molecular = read(ROOT / f"examples/global_optimisation/assets/tio2_water{waters}_seeds.xyz")
    molecular.info = dict(initial_state="molecular")
    # Protonation activates torsions involving exactly collinear ideal-slab
    # atoms. Break that numerical degeneracy identically in both seed slabs.
    fixed = set(np.concatenate([c.get_indices() for c in molecular.constraints]).tolist())
    mobile_slab = [i for i in range(192) if i not in fixed]
    molecular.positions[mobile_slab] += np.random.default_rng(20261010).normal(0., 0.001, (len(mobile_slab), 3))
    trial = molecular.copy()
    oxygen, hydrogen = 192, 193
    surface_oxygens = np.flatnonzero((trial.numbers == 8) & (trial.get_tags() == 0))
    distances = trial.get_distances(oxygen, surface_oxygens, mic=True)
    acceptor = int(surface_oxygens[np.argmin(distances)])
    direction = trial.get_distance(oxygen, acceptor, mic=True, vector=True)
    direction /= np.linalg.norm(direction)
    # Point one intact O-H toward the surface before proposing its break.
    trial.positions[hydrogen] = trial.positions[oxygen] + direction
    params = dict(method="bond_change", pairs=[["O", "H"]], mode="break", break_distance=1.5,
                  bond_cutoffs=CUTOFFS, break_fragments=["H2O"], allow_isolated=True)
    op = parse_operators([params])[0][0]
    op._print = op._debug = lambda *args: None
    prepare_operators([op], sorted(set(trial.numbers)))
    for seed in range(100):
        proposal = op.propose(trial, np.random.default_rng(seed))
        if not proposal.valid:
            continue
        report = adsorption_report(trial)
        if proposal.metadata["bond_change"]["pair"] == [oxygen, hydrogen] and report["surface_hydrogens"] == 1:
            proposal.commit()
            trial.info["initial_state"] = "dissociated"
            POLICY.reconcile(trial)
            return [molecular, trial]
        proposal.rollback()
    raise RuntimeError("Could not generate the surface-dissociated seed using bond_change.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waters", type=int, choices=[2, 3], default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/tio2-water-bond-change")
    parser.add_argument("--seed-output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fmax", type=float, default=0.02)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.fmax) or args.fmax <= 0:
        parser.error("steps and fmax must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = build_seeds(args.waters)
    if args.seed_output is not None:
        write(args.seed_output, seeds, format="extxyz")
    if not args.build_only:
        from xreac import ForceField
        from xreac.ase import ReaxFFCalculator
        ff = ForceField.bundled(MODEL)
    result = dict(model=MODEL, waters=args.waters, fmax_ev_angstrom=args.fmax, states={})
    final = []
    for frame in seeds:
        state = frame.info["initial_state"]
        path = args.output / state
        path.mkdir()
        write(path / "initial.xyz", frame)
        report = dict(initial=adsorption_report(frame))
        if not args.build_only:
            fixed = np.concatenate([c.get_indices() for c in frame.constraints])
            fixed_positions = frame.positions[fixed].copy()
            frame.calc = ReaxFFCalculator(ff)
            with BFGS(frame, maxstep=0.1, logfile=str(path / "optimize.log"),
                      trajectory=str(path / "relaxation.traj")) as optimizer:
                converged = optimizer.run(fmax=args.fmax, steps=args.steps)
                steps = optimizer.nsteps
            assert np.array_equal(frame.positions[fixed], fixed_positions)
            POLICY.reconcile(frame)
            forces = frame.get_forces()
            assert np.all(np.isfinite(forces)) and np.isfinite(frame.get_potential_energy())
            report.update(converged=bool(converged), steps=steps, energy_ev=float(frame.get_potential_energy()),
                          max_mobile_force_ev_angstrom=float(np.linalg.norm(forces, axis=1).max()),
                          final=adsorption_report(frame))
            write(path / "relaxed.xyz", frame)
            final.append(frame)
            print(f"{state}: converged={converged}, E={report['energy_ev']:.8f}, "
                  f"surface H={report['final']['surface_hydrogens']}, OH={report['final']['hydroxyls']}", flush=True)
        result["states"][state] = report
        (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    if final:
        result.update(force_field_sha256=ff.checksum,
                      both_converged=all(report["converged"] for report in result["states"].values()))
        write(args.output / "relaxed_pair.xyz", final)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        if not result["both_converged"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
