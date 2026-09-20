"""Restartable local hopping chains with durable accepted-step boundaries."""
import pickle
import shutil

from ase.io import write

from gdpx.sampling import select_operator
from gdpx.sampling.geometry import infer_unique_atomic_numbers, prepare_operators
from ..accepted_state import load_accepted_state, save_accepted_state


def run_hopping_steps(atoms, identifier, driver, operators, probabilities, mcsteps, rng, checkpoint_directory=None):
    """Resume committed hops; an interrupted hop reuses its driver checkpoint."""
    numbers = infer_unique_atomic_numbers(operators, substrates=[atoms])
    for op in operators:
        prepare_operators([op], numbers, getattr(op, "bond_distance_dict", None),
                          getattr(op, "custom_pair_distance_dict", None))
    base_directory = driver.directory
    trajectory = base_directory.parent / "mctrajs" / f"mc-{identifier:>04d}.xyz"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    original_atoms = atoms
    had_mcstep = "mcstep" in atoms.info
    previous_mcstep = atoms.info.get("mcstep")
    states = []
    start = 0

    def checkpoint(step):
        if checkpoint_directory is None:
            return
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        staging = checkpoint_directory / f"pending-{step:06d}"
        staging.mkdir(exist_ok=True)
        save_accepted_state(staging / "accepted.pkl", atoms, atoms.get_potential_energy())
        with (staging / "state.pkl").open("wb") as stream:
            pickle.dump(dict(version=1, step=step, total_steps=mcsteps, states=states, rng=rng.bit_generator.state), stream)
        staging.rename(checkpoint_directory / f"step-{step:06d}")

    try:
        completed = sorted(checkpoint_directory.glob("step-*")) if checkpoint_directory is not None else []
        if completed:
            with (completed[-1] / "state.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            if saved["version"] != 1:
                raise ValueError("Unsupported BH chain checkpoint version.")
            if saved["total_steps"] != mcsteps:
                raise ValueError("BH num_mcmoves changed; resume with the original recipe.")
            start, states = saved["step"], saved["states"]
            atoms = load_accepted_state(completed[-1] / "accepted.pkl")
            rng.bit_generator.state = saved["rng"]
            # Rebuild the display trajectory from committed states so a crash
            # between checkpoint and XYZ writes cannot duplicate or omit a hop.
            frames = []
            for step in [0] + [i + 1 for i, state in enumerate(states) if state == 0]:
                frame = load_accepted_state(checkpoint_directory / f"step-{step:06d}" / "accepted.pkl")
                frame.info["mcstep"] = step
                frames.append(frame)
            write(trajectory, frames)
        else:
            atoms.info["mcstep"] = 0
            checkpoint(0)
            write(trajectory, atoms)
        energy_before = atoms.get_potential_energy()
        for step in range(start + 1, mcsteps + 1):
            if checkpoint_directory is not None:
                driver.directory = base_directory / f"hop-{step:06d}"
            op = select_operator(operators, probabilities, rng)
            proposal = op.propose(atoms, rng)
            if not proposal.valid:
                states.append(2)
                checkpoint(step)
                continue
            with proposal:
                tags = atoms.get_tags()
                driver.run(atoms, read_ckpt=True)
            relaxed = driver.read_trajectory()[-1]
            relaxed.set_tags(tags)
            energy_after = relaxed.get_potential_energy()
            success = op.acceptance.accept(proposal, energy_before, energy_after, rng)
            if success:
                atoms = relaxed
                energy_before = energy_after
                atoms.info["mcstep"] = step
            states.append(0 if success else 1)
            checkpoint(step)
            if success:
                write(trajectory, atoms, append=True)
            if driver.directory.exists():
                shutil.rmtree(driver.directory)
        return atoms, states
    finally:
        driver.directory = base_directory
        if had_mcstep:
            original_atoms.info["mcstep"] = previous_mcstep
        else:
            original_atoms.info.pop("mcstep", None)
        if atoms is not original_atoms:
            atoms.info.pop("mcstep", None)
