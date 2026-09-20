"""Restartable BH rounds. Workers own evaluation; BH owns acceptance."""
import copy
import pickle
from contextlib import ExitStack
from dataclasses import dataclass

from ase import Atoms
from ase.io import write

from gdpx.sampling import select_operator
from gdpx.sampling.geometry import infer_unique_atomic_numbers, prepare_operators
from ..accepted_state import load_accepted_state, save_accepted_state
from ..generation import EvaluationStatus


def evaluate_batch(frames, worker, directory, archive=False):
    """Submit an idempotent batch and return ID-matched endpoints, or pending."""
    worker.directory = directory
    worker.run(frames)
    return retrieve_batch([a.info["confid"] for a in frames], worker, archive)


def retrieve_batch(identifiers, worker, archive=False):
    worker.inspect(resubmit=True)
    if worker.get_number_of_running_jobs():
        return None
    expected = set(identifiers)
    if len(expected) != len(identifiers):
        raise ValueError("BH batch candidate IDs must be unique.")
    results = {}
    for trajectory in worker.retrieve(include_retrieved=True, use_archive=archive):
        if not trajectory:
            continue
        atoms = trajectory[-1]
        identifier = atoms.info["confid"]
        if identifier not in expected or identifier in results:
            raise RuntimeError("Worker returned an unexpected or duplicate BH trial ID.")
        results[identifier] = atoms
    if results.keys() != expected:
        return None
    return [results[identifier] for identifier in identifiers]


def _save(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def _load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _load_trial(path):
    values = _load(path)
    constraints = values.pop("constraints", [])
    atoms = Atoms.fromdict(values)
    atoms.set_constraint(constraints)
    return atoms


@dataclass
class HoppingResult:
    status: EvaluationStatus
    endpoints: list[Atoms]
    extinct: bool = False


def _commit(directory, step, atoms, states, rng, total_steps, context=None):
    staging = directory / f"staging-{step:06d}"
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(atoms):
        save_accepted_state(staging / f"accepted-{index:06d}.pkl", frame, frame.get_potential_energy())
    _save(staging / "state.pkl", dict(version=4, step=step, total_steps=total_steps,
                                     count=len(atoms), states=states, rng=rng.bit_generator.state, context=context))
    staging.rename(directory / f"round-{step:06d}")


def _write_trajectories(directory, states, count):
    target = directory.parent / "mctrajs"
    target.mkdir(exist_ok=True)
    for index in range(count):
        for step in [0] + [i + 1 for i, values in enumerate(states) if values[index] in (0, 4)]:
            frame = load_accepted_state(directory / f"round-{step:06d}" / f"accepted-{index:06d}.pkl")
            saved = _load(directory / f"round-{step:06d}" / "state.pkl")
            context = saved["context"]
            frame.info.update(mcstep=step, segment=context["segments"][index],
                              source_confid=context["segment_starts"][index],
                              event="start" if step == 0 else ("restart" if states[step - 1][index] == 4 else "accepted"))
            write(target / f"mc-{index:04d}.xyz", frame, append=step != 0)


def run_hopping_rounds(starts, worker, operators, probabilities, mcsteps, rng, directory,
                       archive=False, record_trial=None, restart_chains=None, random_streams=None):
    """Advance a generation through round barriers, returning when work is pending.

    Pending inputs are durable before submission. Live proposals borrow distinct
    chain structures only until worker.run captures the batch; no borrow crosses
    a wait, acceptance decision, or exception boundary. record_trial returns
    whether the evaluated trial is extinct; restart_chains returns independently
    owned starts in terminated-slot order. All named streams are checkpointed
    when a registry is supplied, including replacement-selection randomness.
    """
    directory.mkdir(parents=True, exist_ok=True)
    completed = sorted(directory.glob("round-*"))
    if completed:
        state = _load(completed[-1] / "state.pkl")
        if state.get("version") != 4 or state["total_steps"] != mcsteps or state["count"] != len(starts):
            raise ValueError("Incompatible BH round checkpoint; use the original recipe or start a new run.")
        atoms = [load_accepted_state(completed[-1] / f"accepted-{i:06d}.pkl") for i in range(len(starts))]
        rng.bit_generator.state = state["rng"]
        step, states = state["step"], state["states"]
        context = state["context"]
        if random_streams is not None:
            random_streams.restore(context["random_states"])
    else:
        atoms = list(starts)
        step, states = 0, []
        context = dict(segments=[0] * len(atoms), segment_starts=[a.info.get("confid") for a in atoms],
                       terminated=[], exhausted=False,
                       random_states=random_streams.snapshot() if random_streams is not None else {})
        _commit(directory, 0, atoms, states, rng, mcsteps, context)
    numbers = infer_unique_atomic_numbers(operators, substrates=atoms)
    for op in operators:
        prepare_operators([op], numbers, getattr(op, "bond_distance_dict", None),
                          getattr(op, "custom_pair_distance_dict", None))
    _write_trajectories(directory, states, len(atoms))

    if context["exhausted"]:
        return HoppingResult(EvaluationStatus.FINISHED, [], extinct=True)
    for step in range(step + 1, mcsteps + 1):
        pending = directory / f"pending-{step:06d}"
        worker.directory = directory.parent / "evaluations" / f"round-{step:06d}"
        if pending.exists():
            data = _load(pending / "proposal.pkl")
            if data.get("version") != 4:
                raise ValueError("Unsupported BH pending round checkpoint; start a new run.")
            rng.bit_generator.state = data["rng"]
            if random_streams is not None:
                random_streams.restore(data["random_states"])
            trials = [_load_trial(pending / f"trial-{i:06d}.pkl")
                      for i, entry in enumerate(data["entries"]) if entry["valid"]]
            if trials:
                worker.run(trials)
        else:
            staging = directory / f"proposing-{step:06d}"
            staging.mkdir(exist_ok=True)
            entries, trials = [], []
            with ExitStack() as stack:
                for index, accepted in enumerate(atoms):
                    energy = accepted.get_potential_energy()
                    op = select_operator(operators, probabilities, rng)
                    proposal = op.propose(accepted, rng)
                    entry = dict(valid=proposal.valid, operator=operators.index(op), energy=energy,
                                 metadata=copy.deepcopy(proposal.metadata), diagnostic=proposal.diagnostic)
                    entries.append(entry)
                    if not proposal.valid:
                        continue
                    stack.enter_context(proposal)
                    proposal.set_info("confid", index)
                    entry["tags"] = accepted.get_tags()
                    _save(staging / f"trial-{index:06d}.pkl", accepted.todict())
                    trials.append(accepted)
                data = dict(version=4, entries=entries, rng=copy.deepcopy(rng.bit_generator.state),
                            random_states=random_streams.snapshot() if random_streams is not None else {})
                _save(staging / "proposal.pkl", data)
                staging.rename(pending)
                if trials:
                    worker.run(trials)
        identifiers = [i for i, entry in enumerate(data["entries"]) if entry["valid"]]
        evaluated = retrieve_batch(identifiers, worker, archive) if identifiers else []
        if evaluated is None:
            return HoppingResult(EvaluationStatus.PENDING, [])
        results = {a.info["confid"]: a for a in evaluated}
        decisions, terminated = [], []
        for index, entry in enumerate(data["entries"]):
            if not entry["valid"]:
                decisions.append(2)
                continue
            trial = results[index]
            trial.set_tags(entry["tags"])
            accepted = operators[entry["operator"]].acceptance.accept(
                entry["metadata"], entry["energy"], trial.get_potential_energy(), rng)
            extinct = False
            if record_trial is not None:
                extinct = record_trial(step, index, trial, accepted, atoms[index].info["confid"],
                                       context["segments"][index], context["segment_starts"][index])
            if accepted and extinct:
                decisions.append(3)  # Termination; never propose from this trial.
                terminated.append(index)
            else:
                decisions.append(0 if accepted else 1)
                if accepted:
                    atoms[index] = trial
        # Refresh only after every result has been recorded so selection sees
        # the same complete round, independent of retrieval or ingestion order.
        context["terminated"] = terminated
        if terminated and step < mcsteps:
            replacements = restart_chains(terminated) if restart_chains is not None else []
            if not replacements:
                context["exhausted"] = True
            else:
                if len(replacements) != len(terminated):
                    raise RuntimeError("BH restart must replace every terminated chain.")
                for index, replacement in zip(terminated, replacements):
                    atoms[index] = replacement
                    context["segments"][index] += 1
                    context["segment_starts"][index] = replacement.info["confid"]
                    decisions[index] = 4
                context["terminated"] = []
        states.append(decisions)
        context["random_states"] = random_streams.snapshot() if random_streams is not None else {}
        _commit(directory, step, atoms, states, rng, mcsteps, context)
        _write_trajectories(directory, states, len(atoms))
        if context["exhausted"]:
            return HoppingResult(EvaluationStatus.FINISHED, [], extinct=True)
    return HoppingResult(EvaluationStatus.FINISHED,
                         [a for i, a in enumerate(atoms) if i not in context["terminated"]])
