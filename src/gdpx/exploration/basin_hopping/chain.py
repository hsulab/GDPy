"""Restartable BH rounds. Workers own evaluation; BH owns acceptance."""
import copy
import json
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass

import ase.db
from ase import Atoms
from ase.io import write

from gdpx.sampling import select_operator
from gdpx.sampling.geometry import infer_unique_atomic_numbers, prepare_operators
from ..accepted_state import load_accepted_state, save_accepted_state, load_structure, save_structure
from ..checkpoint import save_data, load_data, publish_snapshot, read_snapshot, prune_snapshots
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


_save = save_data
_load = load_data
_load_trial = load_structure


@dataclass
class HoppingResult:
    status: EvaluationStatus
    endpoints: list[Atoms]
    extinct: bool = False


def _commit(directory, step, atoms, states, rng, total_steps, context=None):
    staging = directory / f"staging-{step:06d}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for index, frame in enumerate(atoms):
        save_accepted_state(staging / f"accepted-{index:06d}.json", frame, frame.get_potential_energy())
    journal = directory / "events.jsonl"
    with journal.open("r+b" if journal.exists() else "w+b") as stream:
        stream.truncate(context.get("journal_offset", 0))
        stream.seek(0, 2)
        event = dict(step=step, decisions=states[-1] if step else [0] * len(atoms),
                     candidates=context["current_ids"], segments=context["segments"],
                     sources=context["segment_starts"])
        stream.write((json.dumps(event) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
        offset = stream.tell()
    saved_context = dict(context, journal_offset=offset)
    _save(staging / "state.json", dict(version=5, step=step, total_steps=total_steps,
                                      count=len(atoms), rng=rng.bit_generator.state, context=saved_context))
    publish_snapshot(directory, staging, directory / f"round-{step:06d}", "round-")
    context["journal_offset"] = offset
    for pending in directory.glob("pending-*"):
        if int(pending.name.split('-')[1]) <= step:
            shutil.rmtree(pending)


def _read_round(path):
    state = _load(path / "state.json")
    if state.get("version") != 5:
        raise ValueError("Unsupported BH checkpoint version; start a new run.")
    atoms = [load_accepted_state(path / f"accepted-{i:06d}.json") for i in range(state["count"])]
    journal = path.parent / "events.jsonl"
    if journal.stat().st_size < state["context"]["journal_offset"]:
        raise ValueError("Incomplete BH event journal.")
    return state, atoms


def _write_trajectories(directory, count, read_candidate, offset):
    target = directory.parent / "mctrajs"
    target.mkdir(exist_ok=True)
    with (directory / "events.jsonl").open('rb') as stream:
        while stream.tell() < offset:
            event = json.loads(stream.readline())
            for index in range(count):
                decision = event["decisions"][index]
                if event["step"] and decision not in (0, 4):
                    continue
                frame = read_candidate(event["candidates"][index])
                frame.info.update(mcstep=event["step"], segment=event["segments"][index],
                                  source_confid=event["sources"][index],
                                  event="start" if not event["step"] else ("restart" if decision == 4 else "accepted"))
                write(target / f"mc-{index:04d}.xyz", frame, append=event["step"] != 0)


def finalize_checkpoints(directory):
    """Keep history and small final metadata after database finalization."""
    if (directory / 'final.json').exists():
        _load(directory / 'final.json')
    elif (directory / 'current.json').exists():
        _, (state, _) = read_snapshot(directory, _read_round)
        _save(directory / 'final.json', state)
    else:
        return
    for pattern in ('round-*', 'pending-*', 'staging-*', 'proposing-*'):
        for path in directory.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
    (directory / 'current.json').unlink(missing_ok=True)


def run_hopping_rounds(starts, worker, operators, probabilities, mcsteps, rng, directory,
                       archive=False, record_trial=None, restart_chains=None, random_streams=None, read_candidate=None):
    """Advance a generation through round barriers, returning when work is pending.

    Pending inputs are durable before submission. Live proposals borrow distinct
    chain structures only until worker.run captures the batch; no borrow crosses
    a wait, acceptance decision, or exception boundary. record_trial returns
    whether the evaluated trial is extinct; restart_chains returns independently
    owned starts in terminated-slot order. All named streams are checkpointed
    when a registry is supplied, including replacement-selection randomness.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # Standalone coordinators use an ASE history database; the engine supplies
    # its existing candidate database, so scientific structures are stored once.
    history = None
    if read_candidate is None:
        history = ase.db.connect(directory / 'history.db')
        read_candidate = lambda identifier: history.get_atoms(identifier, add_additional_information=True)
    def history_record(frame, key):
        rows = list(history.select(event_key=key))
        return rows[0].id if rows else history.write(frame, event_key=key)

    if (directory / 'current.json').exists():
        checkpoint, (state, atoms) = read_snapshot(directory, _read_round)
        if state["total_steps"] != mcsteps or state["count"] != len(starts):
            raise ValueError("Incompatible BH round checkpoint; use the original recipe or start a new run.")
        rng.bit_generator.state = state["rng"]
        step, states = state["step"], []
        context = state["context"]
        if random_streams is not None:
            random_streams.restore(context["random_states"])
    else:
        if any(directory.glob('round-*')) or any(directory.glob('pending-*')):
            raise ValueError("Legacy BH checkpoint is not supported; start a new run.")
        atoms = list(starts)
        step, states = 0, []
        ids = [history_record(a, f'start:{i}') if history is not None else a.info['confid'] for i, a in enumerate(atoms)]
        context = dict(segments=[0] * len(atoms), segment_starts=[a.info.get("confid") for a in atoms],
                       current_ids=ids, journal_offset=0, terminated=[], exhausted=False,
                       random_states=random_streams.snapshot() if random_streams is not None else {})
        _commit(directory, 0, atoms, states, rng, mcsteps, context)
    prune_snapshots(directory, "round-")
    for pending in directory.glob("pending-*"):
        if int(pending.name.split('-')[1]) <= step:
            shutil.rmtree(pending)
    numbers = infer_unique_atomic_numbers(operators, substrates=atoms)
    for op in operators:
        prepare_operators([op], numbers, getattr(op, "bond_distance_dict", None),
                          getattr(op, "custom_pair_distance_dict", None))
    _write_trajectories(directory, len(atoms), read_candidate, context["journal_offset"])

    if context["exhausted"]:
        return HoppingResult(EvaluationStatus.FINISHED, [], extinct=True)
    for step in range(step + 1, mcsteps + 1):
        pending = directory / f"pending-{step:06d}"
        worker.directory = directory.parent / "evaluations" / f"round-{step:06d}"
        if pending.exists():
            data = _load(pending / "proposal.json")
            if data.get("version") != 5:
                raise ValueError("Unsupported BH pending round checkpoint; start a new run.")
            rng.bit_generator.state = data["rng"]
            if random_streams is not None:
                random_streams.restore(data["random_states"])
            trials = [_load_trial(pending / f"trial-{i:06d}.json")
                      for i, entry in enumerate(data["entries"]) if entry["valid"]]
            if trials:
                worker.run(trials)
        else:
            staging = directory / f"proposing-{step:06d}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
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
                    save_structure(staging / f"trial-{index:06d}.json", accepted)
                    trials.append(accepted)
                data = dict(version=5, entries=entries, rng=copy.deepcopy(rng.bit_generator.state),
                            random_states=random_streams.snapshot() if random_streams is not None else {})
                _save(staging / "proposal.json", data)
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
            identifier = history_record(trial, f"trial:{step}:{index}") if history is not None else trial.info["confid"]
            if accepted and extinct:
                decisions.append(3)  # Termination; never propose from this trial.
                terminated.append(index)
            else:
                decisions.append(0 if accepted else 1)
                if accepted:
                    atoms[index] = trial
                    context["current_ids"][index] = identifier
        # Refresh only after every result has been recorded so selection sees
        # the same complete round, independent of retrieval or ingestion order.
        context["terminated"] = terminated
        if terminated and step < mcsteps:
            replacements = restart_chains(terminated, step) if restart_chains is not None else []
            if not replacements:
                context["exhausted"] = True
            else:
                if len(replacements) != len(terminated):
                    raise RuntimeError("BH restart must replace every terminated chain.")
                for index, replacement in zip(terminated, replacements):
                    atoms[index] = replacement
                    context["current_ids"][index] = (history_record(replacement, f"restart:{step}:{index}") if history is not None
                                                     else replacement.info["confid"])
                    context["segments"][index] += 1
                    context["segment_starts"][index] = replacement.info["confid"]
                    decisions[index] = 4
                context["terminated"] = []
        states = [decisions]
        context["random_states"] = random_streams.snapshot() if random_streams is not None else {}
        _commit(directory, step, atoms, states, rng, mcsteps, context)
        _write_trajectories(directory, len(atoms), read_candidate, context["journal_offset"])
        if context["exhausted"]:
            return HoppingResult(EvaluationStatus.FINISHED, [], extinct=True)
    return HoppingResult(EvaluationStatus.FINISHED,
                         [a for i, a in enumerate(atoms) if i not in context["terminated"]])
