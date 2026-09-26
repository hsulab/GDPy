"""MC progress containers, using only already available state and energies."""

from contextlib import contextmanager
import json

from gdpx.core.output import Box, quiet_logging


@contextmanager
def _progress_box(method, title):
    box = Box(f"{method} | {title}")
    try:
        with box.as_parent(), quiet_logging():
            yield box
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        box.line(f"status: {status} | {type(error).__name__}: {error}")
        raise
    finally:
        box.border("bottom")


@contextmanager
def mc_box(title):
    """Nest worker reports and close the container even on failure."""
    with _progress_box("monte carlo", title) as box:
        yield box


@contextmanager
def hmc_box(title):
    """Create a hybrid-MC progress container."""
    with _progress_box("hybrid monte carlo", title) as box:
        yield box


def report_setup(operators, probabilities, steps, seed):
    with mc_box("setup") as box:
        box.line(f"step budget: {steps} | random seed: {seed}")
        for index, (operator, probability) in enumerate(zip(operators, probabilities)):
            if index:
                box.border("middle")
            params = operator.as_dict()
            box.line(f"operator {index}: {operator.name} | probability: {probability:.6g}")
            box.line(f"particles: {params.get('particles', [])} | temperature [K]: {operator.temperature:g}")
            for key, unit in (("chempots", "eV"), ("max_disp", "Å"), ("rattle_strength", "Å")):
                if key in params:
                    box.line(f"{key} [{unit}]: {params[key]}")
            box.line("region: " + json.dumps(params["region"], sort_keys=True))
            box.line(f"distance checks: {'off' if operator.skip_distance_check else 'on'} | "
                     f"proposal attempts: {operator.MAX_RANDOM_ATTEMPTS}")


def report_hybrid_intro(cycle, cycles, seed, trajectory, move_log):
    """Summarize the hybrid cycle and its primary outputs."""
    with hmc_box("intro") as box:
        box.line(f"cycle budget: {cycles} | random seed: {seed}")
        for index, stage in enumerate(cycle):
            method = stage["method"]
            executor = stage["runtime"]["executor"]
            if method == "molecular_dynamics":
                parameters = executor.get("parameters", {})
                detail = f"{parameters.get('steps', '?')} MD steps"
                if "temp" in parameters:
                    detail += f" at {parameters['temp']:g} K"
            else:
                detail = f"{stage['steps']} MC proposals"
            box.line(
                f"stage {index}: {method} | {detail} | "
                f"{executor.get('provider', '?')}/{executor['method']}"
            )
        box.line(f"outputs: trajectory {trajectory} | MC moves {move_log}")


def report_status(status, detail):
    with mc_box(status) as box:
        box.line(detail)


def report_outcome(box, result, previous_energy):
    box.line(f"operator: {result.operator.name} | {result.diagnostic}")
    if result.accepted is None:
        box.line("decision: waiting for evaluation")
    elif not result.valid:
        box.line("decision: invalid proposal")
    else:
        box.line(f"decision: {'accepted' if result.accepted else 'rejected'}")
        box.line(f"energy [eV]: previous {previous_energy:.6f} | trial {result.energy:.6f}")
        box.line(f"trial energy change [eV]: {result.energy - previous_energy:+.6f}")
    current_energy = result.energy if result.accepted else previous_energy
    box.line(f"current energy [eV]: {current_energy:.6f} | atoms: {len(result.atoms)}")
