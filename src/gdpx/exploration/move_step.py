"""Worker lifecycle for a borrowed trial, shared by sequential explorations."""

import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from gdpx.sampling import select_operator
from gdpx.sampling.moves.operator import BaseMCOperator

from .accepted_state import load_accepted_state, save_accepted_state
from .checkpoint import load_data, save_data


@dataclass
class MoveOutcome:
    atoms: Atoms
    operator: BaseMCOperator
    diagnostic: str
    energy: float
    accepted: bool | None
    valid: bool = True


def read_pending(path, rng):
    if not (path / "proposal.json").exists():
        raise ValueError("Legacy pending move checkpoint is not supported; start a new run.")
    data = load_data(path / "proposal.json")
    if data.get("version") != 2:
        raise ValueError("Unsupported pending move checkpoint; start a new run.")
    rng.bit_generator.state = data["rng"]
    return load_accepted_state(path / data["structure"]), data


def _store_pending(path, atoms, data):
    destination = path
    initial = not path.exists()
    if initial:
        path = path.with_name(f".{path.name}-staging")
        if path.exists():
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    name = f"structure-{uuid.uuid4().hex}.json"
    save_accepted_state(path / name, atoms, atoms.get_potential_energy())
    save_data(path / "proposal.json", dict(data, structure=name))
    if initial:
        path.rename(destination)
        path = destination
    # Keep only the files referenced by the newly published document.
    import json
    references = {"proposal.json", name}
    for document in (path / name, path / "proposal.json"):
        payload = json.loads(document.read_text())
        if "arrays" in payload:
            references.add(payload["arrays"])
    for old in path.iterdir():
        if old.is_file() and old.name not in references:
            old.unlink()


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
    had_pending = pending_path.exists()
    pending = had_pending
    if pending:
        # A hybrid substep may have advanced beyond the last durable resolution.
        saved = load_data(pending_path / "proposal.json") if (pending_path / "proposal.json").exists() else None
        if saved is not None and saved.get("resolved") and saved["context"] != resume_context:
            pending = False
    if pending:
        atoms, data = read_pending(pending_path, rng)
        energy = data["energy"]
        operator = operators[data["operator"]]
        if data.get("resolved"):
            return MoveOutcome(atoms, operator, data["diagnostic"], data["trial_energy"], data["accepted"])
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
                version=2,
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
            _store_pending(pending_path, atoms, data)
        return MoveOutcome(atoms, operator, data["diagnostic"], energy, None)

    wdir = getattr(worker, "wdir_name", None)
    evaluated = worker.retrieve(include_retrieved=True, given_wdirs=[wdir] if wdir else None)[0][-1]
    evaluated.set_tags(data["tags"])
    trial_energy = evaluated.get_potential_energy()
    accepted = operator.acceptance.accept(data["metadata"], energy, trial_energy, rng)
    if had_pending:
        data.update(resolved=True, accepted=accepted, trial_energy=trial_energy, rng=rng.bit_generator.state)
        _store_pending(pending_path, evaluated if accepted else atoms, data)
    return MoveOutcome(evaluated if accepted else atoms, operator, data["diagnostic"], trial_energy, accepted)
