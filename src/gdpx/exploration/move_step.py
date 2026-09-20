"""Worker lifecycle for a borrowed trial, shared by sequential explorations."""

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from gdpx.sampling import select_operator
from gdpx.sampling.moves.operator import BaseMCOperator

from .accepted_state import load_accepted_state, save_accepted_state


@dataclass
class MoveOutcome:
    atoms: Atoms
    operator: BaseMCOperator
    diagnostic: str
    energy: float
    accepted: bool | None
    valid: bool = True


def read_pending(path, rng):
    with (path / "proposal.pkl").open("rb") as stream:
        data = pickle.load(stream)
    if data.get("version") != 1:
        raise ValueError("Unsupported pending move checkpoint; start a new run.")
    rng.bit_generator.state = data["rng"]
    return load_accepted_state(path / "accepted.pkl"), data


def run_worker_move(
    atoms: Atoms,
    energy: float,
    operators: list[BaseMCOperator],
    probabilities: list[float],
    rng: np.random.Generator,
    worker,
    pending_path: Path,
    info: dict | None = None,
    attempts_path: Path | None = None,
    resume_context: dict | None = None,
) -> MoveOutcome:
    """Submit once, release the borrowed trial, then inspect/retrieve its result.

    DriverBasedWorker captures inputs in _preprocess before returning. Pending
    work stores the accepted frame and decision metadata, never a live borrow.
    """
    pending = pending_path.exists()
    if pending:
        atoms, data = read_pending(pending_path, rng)
        energy = data["energy"]
        operator = operators[data["operator"]]
    else:
        operator = select_operator(operators, probabilities, rng)
        proposal = operator.propose(atoms, rng)
        if not proposal.valid:
            return MoveOutcome(atoms, operator, proposal.diagnostic, energy, False, False)
        with proposal:
            for key, value in (info or {}).items():
                proposal.set_info(key, value)
            if attempts_path is not None:
                write(attempts_path, atoms, append=True)
            data = dict(
                version=1,
                operator=operators.index(operator),
                metadata=proposal.metadata,
                diagnostic=proposal.diagnostic,
                tags=atoms.get_tags(),
                energy=energy,
                context=resume_context,
            )
            worker.run([atoms], read_ckpt=True)
        data["rng"] = rng.bit_generator.state

    worker.inspect(resubmit=True)
    if worker.get_number_of_running_jobs() != 0:
        if not pending:
            pending_path.mkdir(parents=True)
            save_accepted_state(pending_path / "accepted.pkl", atoms, energy)
            with (pending_path / "proposal.pkl").open("wb") as stream:
                pickle.dump(data, stream)
        return MoveOutcome(atoms, operator, data["diagnostic"], energy, None)

    evaluated = worker.retrieve()[0][-1]
    evaluated.set_tags(data["tags"])
    trial_energy = evaluated.get_potential_energy()
    accepted = operator.acceptance.accept(data["metadata"], energy, trial_energy, rng)
    if pending:
        shutil.rmtree(pending_path)
    return MoveOutcome(evaluated if accepted else atoms, operator, data["diagnostic"], trial_energy, accepted)
