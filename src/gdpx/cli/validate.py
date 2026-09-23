import pathlib
from typing import Union

from gdpx.factory.computer import canonicalise_worker
from gdpx.factory.validator import canonicalise_validator


def run_validation(config: dict, directory: Union[str, pathlib.Path], worker):
    """This is a factory to deal with various validations..."""
    # run over validations
    directory = pathlib.Path(directory)

    tasks = config.get("tasks", [])
    if not tasks:
        raise Exception("No tasks found in the configuration.")

    # Assign worker to each task, priority is task-specific > task-global > command
    task_global_worker_params = config.get("worker", {})
    if task_global_worker_params:
        # override the worker from command line
        task_global_worker = canonicalise_worker(task_global_worker_params)
    else:
        task_global_worker = worker

    for task in tasks:
        task_worker = task.get("worker", {})
        if task_worker:
            task["worker"] = task_worker
        elif task_global_worker:
            task["worker"] = task_global_worker
        elif worker:
            task["worker"] = worker

    # Instantiate teh validators
    validators = []
    for task in tasks:
        validator = canonicalise_validator(task)
        validators.append(validator)

    # run the validations sequentially
    for i, validator in enumerate(validators):
        validator.directory = directory / f"v.{i:>02d}.{validator.name}"
        validator.run()

    return
