"""Training committees using the shared worker layout and lifecycle."""
import functools
import shlex

import numpy as np
import yaml

from gdpx.data.loaders.factory import create_dataloader
from gdpx.execution.fingerprint import atomic_write_text, payload_digest
from gdpx.providers import ComponentConfig
from gdpx.providers.training import BasePotentialTrainer

from .catalog import CatalogWorker
from .registry import WORKER_REGISTRY
from .utils import render_concurrent_task_commands, render_worker_root_command


@WORKER_REGISTRY.register
class TrainerBasedWorker(CatalogWorker):
    """Prepare model directories and monitor committee training jobs."""

    TRAIN_PREFIX = "m"
    worker_kind = "training"

    def __init__(self, trainer, scheduler, share_dataset=False, auto_submit=True,
                 directory=None, *args, **kwargs):
        super().__init__(directory=directory, *args, **kwargs)
        self.trainer = trainer
        self.scheduler = scheduler
        self._share_dataset = share_dataset
        self._submit = auto_submit

    def _get_train_params(
        self,
        trainer: BasePotentialTrainer,
        dataset,
        init_model,
        use_shared_dataset: bool = False,
        random_seed=None,
    ) -> dict:
        """"""
        trainer_params = {}

        component_config = getattr(trainer, "component_config", None)
        if component_config is None:
            # Preserve support for callers that construct a trainer directly
            # instead of using the provider factory.  The provider-created path
            # always carries its original method and normalized parameters.
            parameters = trainer.as_dict()
            parameters.pop("name", None)
            component_config = ComponentConfig(
                provider=trainer.name,
                method="default",
                parameters=parameters,
            )
        trainer_params["trainer"] = component_config.to_dict()

        # extra params
        trainer_params["share_dataset"] = use_shared_dataset

        # TODO: we set a random seed for each trainer
        #       as a committee will be trained
        #       it changes the trainer's random state as well...
        trainer_random_seed = int(np.random.randint(0, 10000)) if random_seed is None else random_seed
        trainer_params["trainer"]["parameters"]["random_seed"] = trainer_random_seed
        trainer.set_rng(seed=trainer_random_seed)

        trainer_params["init_model"] = init_model
        trainer_params["dataset"] = dataset.as_dict()

        return trainer_params

    def _prepare_shared_dataset(self, dataset, size, *args, **kwargs):
        if size <= 1 or not self._share_dataset:
            return dataset
        path = self.directory / "shared_dataset"
        configuration = path / "dataset.yaml"
        if configuration.exists():
            return create_dataloader(yaml.safe_load(configuration.read_text()))
        if not hasattr(self.trainer, "_prepare_dataset"):
            return dataset
        previous_directory = self.trainer.directory
        try:
            self.trainer.directory = path
            dataset = self.trainer._prepare_dataset(dataset, *args, **kwargs)
            atomic_write_text(configuration, yaml.safe_dump(dataset.as_dict()))
        finally:
            self.trainer.directory = previous_directory
        return dataset

    def run(self, dataset, size=1, init_models=None, *args, **kwargs):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Training size must be a positive integer.")
        init_models = [None] * size if init_models is None else init_models
        if len(init_models) != size:
            raise ValueError("The number of initial models must match training size.")
        if self.scheduler.concurrent_tasks > 1 and self.scheduler.transport_name != "local":
            raise ValueError("Packed training currently requires a local scheduler transport.")
        super().run(*args, **kwargs)
        dataset = self._prepare_shared_dataset(dataset, size, *args, **kwargs)
        saved = self.metadata.inputs.read()["workers"].get(self.metadata.worker)
        seeds = {}
        if saved is not None:
            definition = self.metadata.calculation_set(self.metadata.inputs.read(), self.metadata.worker)
            seeds = {name: params["trainer"]["parameters"]["random_seed"]
                     for batch in definition["batches"]
                     for name, params in zip(batch["wdir_names"], batch["trainers"])}
        batches = []
        concurrent = self.scheduler.concurrent_tasks
        for start in range(0, size, concurrent):
            names = [f"{self.TRAIN_PREFIX}{i}" for i in range(start, min(start + concurrent, size))]
            trainers = [self._get_train_params(
                self.trainer, dataset, str(init_models[i]) if init_models[i] is not None else None,
                self._share_dataset, random_seed=seeds.get(name),
            ) for i, name in enumerate(names, start)]
            batches.append(dict(version=1, group_number=start // concurrent,
                                wdir_names=names, trainers=trainers))
        self.metadata.freeze_tasks(batches, self.scheduler.machine_prefix)
        for batch in batches:
            uid = self.metadata.prepare_job(batch, self.scheduler.machine_prefix)
            job_name = f"{uid}-group-{batch['group_number']}"
            if self.job_store.get_by_gdir(job_name) is not None:
                continue
            self.job_store.insert(uid, "", job_name, batch["group_number"], batch["wdir_names"],
                                  job_digest=payload_digest(batch))
            job = self.job_store.get_by_gdir(job_name)
            if self._submit:
                self._submit_job(job)
            else:
                self._write_job(job)

    def _write_job(self, job):
        saved = self.metadata.manifest(job.uid)
        batch = saved["input"]
        if batch["wdir_names"] != job.wdir_names or saved["job_digest"] != job.job_digest:
            raise ValueError("Training job metadata does not match its frozen inputs.")
        for name, params in zip(batch["wdir_names"], batch["trainers"]):
            path = self.directory / name
            path.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path / "trainer.yaml", yaml.safe_dump(params))
        self._prepare_scheduler_for_job(job)
        prefix = saved["machine_prefix"].strip()
        launch = (prefix + " " if prefix else "") + "gdp train trainer.yaml > training.log 2>&1"
        command = render_worker_root_command('cd "${workdirs[$task]}" && ' + launch,
                                             self.scheduler.name)
        names = " ".join(shlex.quote(name) for name in batch["wdir_names"])
        self.scheduler.user_commands = f"workdirs=({names})\n" + render_concurrent_task_commands(
            command, len(batch["wdir_names"]), self.scheduler.concurrent_tasks,
        )
        self.scheduler.script.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler.write()

    def _run_models(self, names):
        from gdpx.cli.train import run_trainer
        for name in names:
            path = self.directory / name
            run_trainer(path / "trainer.yaml", path)

    def _submit_job(self, job):
        self._write_job(job)
        callback = (functools.partial(self._run_models, job.wdir_names)
                    if self.scheduler.is_direct and len(job.wdir_names) == 1 else None)
        job_id = self.scheduler.submit(func_to_execute=callback)
        self.job_store.mark_submitted(job.gdir, job_id)
        self._print(f"{self.directory.name} JOBID: {job_id}")

    def _check_job_convergence(self, job):
        for name in job.wdir_names:
            path = self.directory / name
            if not path.exists():
                return False
            self.trainer.directory = path
            if not self.trainer.read_convergence():
                return False
        return True

    def _resubmit_job(self, job):
        if self._submit:
            self._submit_job(job)

    def _do_retrieve(self, include_retrieved=False, *args, **kwargs):
        jobs = self.job_store.get_finished() if include_retrieved else self.job_store.get_unretrieved()
        results = []
        for job in jobs:
            for name in job.wdir_names:
                self.trainer.directory = self.directory / name
                results.append(str(self.trainer.freeze()))
            self.job_store.mark_retrieved(job.gdir)
        return results
