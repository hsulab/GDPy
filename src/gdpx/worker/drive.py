import copy
import functools
import itertools
import json
import pathlib
import shutil
import tarfile
import tempfile
import time
import uuid
from typing import Optional, Union

import numpy as np
import omegaconf
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed

from gdpx.builder.builder import StructureBuilder
from gdpx.computation.driver import BaseDriver
from gdpx.core.register import registers
from gdpx.potential.manager import BasePotentialManager
from gdpx.scheduler import LocalScheduler
from gdpx.scheduler.scheduler import BaseScheduler
from gdpx.utils.profiler import CustomTimer

from .pairing import Pairing
from .store import JobRecord
from .utils import copy_minimal_frames, get_file_md5, split_batches
from .worker import BaseWorker

# ---------------------------------------------------------------------------
# Module-level command-line runners
# ---------------------------------------------------------------------------


def run_computation_in_commandline(
    identifier: str,
    structures: list[Atoms],
    computation_dirnames: list[str],
    rng_states,
    drivers: Union[BaseDriver, list[BaseDriver]],
    driver_indices: Optional[list[int]],
    directory: pathlib.Path,
    share_wdir: bool,
    print_period: int = 100,
    print_func=print,
) -> None:
    """Run computations directly in the commandline.

    Args:
        identifier: MD5 identifier of the input structures.
        structures: A batch of structures.
        computation_dirnames: Working directories for each structure.
        rng_states: Random seeds for each structure.
        drivers: A single driver or list of drivers.
        driver_indices: Which driver each structure uses (None if single driver).
        directory: Root computation directory.
        share_wdir: Whether to share a working directory.
        print_period: Print frequency for progress.
        print_func: Print function.
    """
    is_single_driver = isinstance(drivers, BaseDriver) or len(drivers) == 1

    # Collect all unique driver instances so we can set machine_prefix once
    all_drivers: list[BaseDriver] = []
    if isinstance(drivers, BaseDriver):
        all_drivers = [drivers]
    else:
        all_drivers = drivers

    # Check machine-specific prefix
    machine_prefix_fpath = directory / "_data" / f"MACHINE_{identifier}"
    if machine_prefix_fpath.exists():
        with open(machine_prefix_fpath, "r") as fopen:
            machine_prefix = "".join(fopen.readlines()).strip()
    else:
        machine_prefix = ""

    # Apply machine prefix to all drivers (save originals for restore)
    prev_prefixes = [d.setting.machine_prefix for d in all_drivers]
    if machine_prefix:
        for d in all_drivers:
            d.setting.machine_prefix = machine_prefix

    # Retrieve the right driver for a given index
    def _get_driver(idx: int) -> BaseDriver:
        if is_single_driver:
            return all_drivers[0]
        return all_drivers[idx]

    # Run computations
    with CustomTimer(name="run-driver", func=print_func):
        if not share_wdir:
            for gi, (dirname, atoms, rs) in enumerate(zip(computation_dirnames, structures, rng_states)):
                d_idx = 0 if driver_indices is None else driver_indices[gi]
                curr_driver = _get_driver(d_idx)
                curr_driver.directory = directory / dirname
                prev_random_seed = curr_driver.random_seed
                curr_driver.set_rng(seed=rs)
                print_func(
                    f"{time.asctime(time.localtime(time.time()))} {dirname} {curr_driver.directory.name} is running..."
                )
                curr_driver.reset()
                curr_driver.run(atoms, read_ckpt=True, extra_info=None)
                curr_driver.set_rng(seed=prev_random_seed)
        else:
            # shared working directory mode
            cache_fpath = directory / "_data" / f"{identifier}_cache.xyz"
            if cache_fpath.exists():
                cache_frames = read(cache_fpath, ":")
                cache_wdirs = [a.info["wdir"] for a in cache_frames]
            else:
                cache_wdirs = []

            temp_wdir = directory / "_shared"
            for gi, (dirname, atoms, rs) in enumerate(zip(computation_dirnames, structures, rng_states)):
                if dirname in cache_wdirs:
                    continue
                d_idx = 0 if driver_indices is None else driver_indices[gi]
                curr_driver = _get_driver(d_idx)
                if temp_wdir.exists():
                    shutil.rmtree(temp_wdir)
                curr_driver.directory = temp_wdir
                if gi % print_period == 0 or gi + 1 == len(structures):
                    print_func(
                        f"{time.asctime(time.localtime(time.time()))} {dirname} "
                        f"{curr_driver.directory.name} is running..."
                    )
                curr_driver.set_rng(seed=rs)
                curr_driver.reset()
                curr_driver.run(atoms, read_ckpt=False, extra_info=dict(wdir=dirname))
                new_atoms = curr_driver.read_trajectory()[-1]
                new_atoms.info["wdir"] = atoms.info["wdir"]
                write(
                    directory / "_data" / f"{identifier}_cache.xyz",
                    new_atoms,
                    append=True,
                )

    # Restore machine prefixes
    for d, prev in zip(all_drivers, prev_prefixes):
        d.setting.machine_prefix = prev

    return


# ---------------------------------------------------------------------------
# Unified DriverBasedWorker
# ---------------------------------------------------------------------------


@registers.worker.register
class DriverBasedWorker(BaseWorker):
    """Monitor driver-based jobs.

    Unifies the original DriverBasedWorker, SingleWorker, and
    GridDriverBasedWorker into a single class.  The *pairing* parameter
    controls how N driver configurations map to M input structures.

    Lifetime: queued (running) -> finished -> retrieved

    The database stores each unique job ID and its working directory.
    """

    reserved_keys: list[str] = ["energy", "step", "wdir"]
    is_spawned: bool = False
    print_period: int = 100

    _share_random_seed: bool = False
    _share_wdir: bool = False
    _retain_info: bool = False

    def __init__(
        self,
        # --- backward-compatible args ---
        potter: Optional[BasePotentialManager] = None,
        driver: Optional[Union[BaseDriver, list[BaseDriver]]] = None,
        scheduler_: Optional[BaseScheduler] = None,
        # --- new-style args ---
        scheduler: Optional[BaseScheduler] = None,
        directory: Optional[Union[str, pathlib.Path]] = None,
        batchsize: int = 1,
        pairing: Union[Pairing, str] = Pairing.AUTO,
        *args,
        **kwargs,
    ):
        super().__init__(directory=directory, batchsize=batchsize, *args, **kwargs)

        # Backward compat: scheduler_ -> scheduler
        self.scheduler = scheduler_ if scheduler_ is not None else (scheduler or LocalScheduler())

        # Potter is stored only for serialisation; it is NOT used at runtime.
        self.potter = potter

        # Always internal list of driver instances
        self._drivers: list[BaseDriver] = []
        if driver is not None:
            if isinstance(driver, list):
                self._drivers = driver
            else:
                self._drivers = [driver]

        self._pairing = Pairing(pairing) if isinstance(pairing, str) else pairing

    # ------------------------------------------------------------------
    # Driver access
    # ------------------------------------------------------------------

    @property
    def driver(self) -> BaseDriver:
        if len(self._drivers) == 1:
            return self._drivers[0]
        raise AttributeError(
            f"{self.__class__.__name__} has {len(self._drivers)} drivers; "
            "use `worker.drivers[i]` or `worker.set_drivers()`."
        )

    @driver.setter
    def driver(self, d: BaseDriver):
        self._drivers = [d]

    @property
    def drivers(self) -> list[BaseDriver]:
        return self._drivers

    def set_drivers(self, *drivers: BaseDriver):
        """Set one or more driver configurations."""
        self._drivers = list(drivers)

    def add_driver(self, driver: BaseDriver):
        self._drivers.append(driver)

    # ------------------------------------------------------------------
    # Task planning — how drivers pair with structures
    # ------------------------------------------------------------------

    def _make_task_plan(self, num_structures: int) -> list[tuple[int, int]]:
        """Build (driver_index, structure_index) pairs for each task.

        The length of the returned list is the total number of tasks.
        Each task is a single (driver, structure) computation.
        """
        nd, ns = len(self._drivers), num_structures
        if self._pairing == Pairing.AUTO:
            return self._infer_pairing(nd, ns)
        elif self._pairing == Pairing.BROADCAST:
            assert nd == 1, f"BROADCAST requires 1 driver, got {nd}."
            return [(0, i) for i in range(ns)]
        elif self._pairing == Pairing.REPEAT:
            return [(i, 0) for i in range(nd)]
        elif self._pairing == Pairing.BIJECTION:
            assert nd == ns, f"BIJECTION requires N drivers == M structures, got N={nd}, M={ns}."
            return [(i, i) for i in range(ns)]
        elif self._pairing == Pairing.PRODUCT:
            return list(itertools.product(range(nd), range(ns)))
        elif self._pairing == Pairing.PARTITION:
            return self._partition_across_drivers(nd, ns)
        else:
            raise ValueError(f"Unknown Pairing: {self._pairing}")

    def _infer_pairing(self, nd: int, ns: int) -> list[tuple[int, int]]:
        if nd == 1:
            return [(0, i) for i in range(ns)]
        if ns == 1:
            return [(i, 0) for i in range(nd)]
        if nd == ns:
            return [(i, i) for i in range(ns)]
        raise ValueError(
            f"Cannot infer Pairing from {nd} drivers and {ns} structures. "
            f"Set an explicit Pairing (e.g. PRODUCT, PARTITION)."
        )

    def _partition_across_drivers(self, nd: int, ns: int) -> list[tuple[int, int]]:
        """Split *ns* structures evenly across *nd* drivers."""
        plan = []
        for si in range(ns):
            di = si % nd
            plan.append((di, si))
        return plan

    # ------------------------------------------------------------------
    # Preprocessing (MD5 caching, seed generation)
    # ------------------------------------------------------------------

    def _read_cached_info(self):
        _info_data = []
        for p in (self.directory / "_data").glob("*_info.txt"):
            with open(p, "r") as fopen:
                for line in fopen.readlines():
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split())
        _info_data = sorted(_info_data, key=lambda x: int(x[0]))
        return _info_data

    def _read_cached_xinfo(self):
        info_keys, _info_data = [], []
        for p in (self.directory / "_data").glob("*_xinfo.txt"):
            with open(p, "r") as fopen:
                lines = fopen.readlines()
                info_keys = lines[0].split()[1:]
                for line in lines:
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split()[1:])
        assert info_keys, "info_keys must not be empty."
        return info_keys, _info_data

    def _preprocess(self, builder, *args, **kwargs):
        frames = []
        if isinstance(builder, StructureBuilder):
            frames = builder.run()
        else:
            assert all(isinstance(x, Atoms) for x in frames), "Input should be a list of atoms."
            frames = builder
        prev_frames = frames

        processed_dpath = self.directory / "_data"
        processed_dpath.mkdir(exist_ok=True)

        curr_frames, curr_info = copy_minimal_frames(prev_frames)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            write(tmp.name, curr_frames, columns=["symbols", "positions", "move_mask"])
            with open(tmp.name, "rb") as fopen:
                curr_md5 = get_file_md5(fopen)

        # Generate random seeds from the first driver's seed
        first_driver = self._drivers[0] if self._drivers else None
        if first_driver is not None:
            self._print(f"Driver's random_seed: {first_driver.random_seed}")
            if not self._share_random_seed:
                rng = np.random.Generator(np.random.PCG64(first_driver.random_seed))
                random_seeds = rng.integers(0, 1e8, size=len(curr_frames))
                random_seeds = [int(x) for x in random_seeds]
            else:
                random_seeds = [first_driver.random_seed] * len(curr_frames)
        else:
            random_seeds = [0] * len(curr_frames)

        _info_data = self._read_cached_info()

        stored_fname = f"{curr_md5}.xyz"
        if (processed_dpath / stored_fname).exists():
            self._print(f"Found file with md5 {curr_md5}")
            self._info_data = _info_data
            start_confid = 0
            for x in self._info_data:
                if x[1] == curr_md5:
                    break
                start_confid += 1
            if len(_info_data) > 0 and len(_info_data[0]) > 5:
                random_seeds = [int(x[-1]) for x in _info_data]
        else:
            if self._retain_info:
                info_keys = []
                for a in prev_frames:
                    info_keys.extend(list(a.info.keys()))
                info_keys = sorted(set(info_keys))
                content = f"{'#id':<12s}  " + ("{:<24s}  " * len(info_keys)).format(*info_keys) + "\n"
                for i, a in enumerate(prev_frames):
                    line = f"{i:<24d}  " + "  ".join([f"{str(a.info.get(k)):<24s}" for k in info_keys]) + "\n"
                    content += line
                with open(processed_dpath / f"{curr_md5}_xinfo.txt", "w") as fopen:
                    fopen.write(content)
            write(processed_dpath / stored_fname, curr_frames)
            start_confid = len(_info_data)
            content = "{:<12s}  {:<32s}  {:<12s}  {:<12s}  {:<s}  {:>24s}\n".format(
                "#id", "MD5", "confid", "step", "wdir", "rs"
            )
            for i, ((confid, step, wdir), rs) in enumerate(zip(curr_info, random_seeds)):
                line = "{:<12d}  {:<32s}  {:<12d}  {:<12d}  {:<s}  {:>24d}\n".format(
                    i + start_confid, curr_md5, confid, step, wdir, rs
                )
                content += line
                _info_data.append(line.strip().split())
            self._info_data = _info_data
            with open(processed_dpath / f"{curr_md5}_info.txt", "w") as fopen:
                fopen.write(content)

        return curr_md5, curr_frames, start_confid, random_seeds

    def _prepare_batches(
        self,
        frames: list[Atoms],
        start_confid: int,
        rng_states: Union[list[int], list[dict]],
    ):
        """Create batches from frames and the task plan."""
        num_frames = len(frames)
        task_plan = self._make_task_plan(num_frames)
        num_tasks = len(task_plan)

        # Build wdir names from the task plan
        wdirs = []
        driver_for_wdir = []
        struct_for_wdir = []
        for gi, (di, si) in enumerate(task_plan):
            wdir_name = "cand{}".format(gi)
            wdirs.append(wdir_name)
            driver_for_wdir.append(di)
            struct_for_wdir.append(si)
            if gi < num_frames:
                frames[si].info["wdir"] = wdir_name

        # Save the task plan for fault tolerance
        task_plan_path = self.directory / "_data" / "task_plan.json"
        if not task_plan_path.exists():
            with open(task_plan_path, "w") as fopen:
                json.dump(
                    dict(
                        pairing=self._pairing.name,
                        num_drivers=len(self._drivers),
                        num_structures=num_frames,
                        tasks=[
                            dict(global_index=i, driver_index=di, structure_index=si)
                            for i, (di, si) in enumerate(task_plan)
                        ],
                    ),
                    fopen,
                    indent=2,
                )

        assert len(set(wdirs)) == num_tasks, f"Found duplicated wdirs {len(set(wdirs))} vs. {num_tasks}."

        overwrite_batchsize = False
        if self._share_wdir or self.scheduler.name == "local":
            overwrite_batchsize = True

        if overwrite_batchsize:
            self._print(f"Overwrites batchsize to {num_tasks=} as it uses share_wdir or local scheduler.")
            batchsize_val = num_tasks
        else:
            batchsize_val = self.batchsize

        starts, ends = (
            self._split_groups(num_tasks)
            if batchsize_val == self.batchsize
            else split_batches(num_tasks, batchsize_val)
        )

        batches = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            global_indices = list(range(s, e))
            batch_wdirs = [wdirs[x] for x in global_indices]
            batch_driver_indices = [driver_for_wdir[x] for x in global_indices]
            batch_seeds = []
            for gi in global_indices:
                si = struct_for_wdir[gi]
                if si < len(rng_states):
                    batch_seeds.append(rng_states[si] if isinstance(rng_states, list) else rng_states)
                else:
                    batch_seeds.append(0)
            batch_frames_structs = [
                frames[struct_for_wdir[x]] for x in global_indices if struct_for_wdir[x] < len(frames)
            ]

            for x in global_indices:
                if struct_for_wdir[x] < len(frames):
                    frames[struct_for_wdir[x]].info["group"] = i

            assert len(set(batch_wdirs)) == len(batch_wdirs), f"Found duplicated wdirs in batch {i}."

            batches.append(
                [
                    global_indices,
                    batch_wdirs,
                    batch_driver_indices,
                    batch_seeds,
                    batch_frames_structs,
                ]
            )

        # Keep task plan for later use in inspect/retrieve
        self._task_plan = task_plan

        return batches

    def prepare_batches(self, builder, rng_states=list()):
        identifier, frames, start_confid, new_rng_states = self._preprocess(builder)
        if rng_states:
            new_rng_states = rng_states
        batches = self._prepare_batches(frames, start_confid, new_rng_states)
        return identifier, frames, batches

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, builder=None, rng_states=list(), *args, **kwargs) -> None:
        super().run(*args, **kwargs)

        identifier, frames, batches = self.prepare_batches(builder, rng_states)

        target_batch = kwargs.get("batch", None)

        if not self.is_spawned:
            self._run_by_scheduler(identifier, frames, batches, target_batch=target_batch)
        else:
            self._run_by_commandline(identifier, frames, batches, target_batch=target_batch)

    def _run_by_commandline(self, identifier: str, frames: list[Atoms], batches, target_batch: Optional[int] = None):
        batch_data = batches[target_batch]
        curr_indices, curr_wdirs, driver_indices, rng_states, curr_frames = batch_data

        run_computation_in_commandline(
            identifier,
            curr_frames,
            curr_wdirs,
            rng_states,
            self._drivers,
            driver_indices,
            self.directory,
            self._share_wdir,
            print_period=self.print_period,
            print_func=self._print,
        )

    def _run_by_scheduler(self, identifier: str, frames: list[Atoms], batches, target_batch: Optional[int] = None):
        database_path = (self.directory / f"_{self.scheduler.name}_jobs.json").resolve()
        self._print(f"database_path: {database_path.relative_to(pathlib.Path.cwd())}")

        queued_jobs = self.job_store.get_queued()
        queued_names = [q.gdir[self.UUIDLEN + 1 :] for q in queued_jobs]
        queued_frames = [q.md5 for q in queued_jobs]

        for ig, batch in enumerate(batches):
            batch_name = f"group-{ig}"
            uid = str(uuid.uuid1())
            job_name = uid + "-" + batch_name

            if batch_name in queued_names and identifier in queued_frames:
                self._print(f"{batch_name} at {self.directory.name} was submitted.")
                continue

            if isinstance(target_batch, int):
                if ig != target_batch:
                    self._print(
                        f"{time.asctime(time.localtime(time.time()))} {self.directory.name} batch {ig} is skipped..."
                    )
                    continue

            if identifier not in queued_frames:
                self.job_store.insert(
                    uid=uid,
                    md5=identifier,
                    gdir=job_name,
                    group_number=ig,
                    wdir_names=batch[1],
                )
                worker_input_fpath = self.directory / "_data" / f"worker-{identifier}.json"
                if not worker_input_fpath.exists():
                    worker_input_dict = omegaconf.OmegaConf.create(self.as_dict())
                    worker_input_dict = omegaconf.OmegaConf.to_container(worker_input_dict)
                    with open(worker_input_fpath, "w") as fopen:
                        json.dump(worker_input_dict, fopen, indent=2)
                    with open(self.directory / "_data" / f"MACHINE_{identifier}", "w") as fopen:
                        fopen.write(self.scheduler.machine_prefix)

            self._irun(
                batch_name,
                uid,
                identifier,
                frames,
                batch,
            )

    def _irun(
        self,
        batch_name: str,
        uid: str,
        identifier: str,
        frames: list[Atoms],
        batch,
    ) -> None:
        batch_number = int(batch_name.split("-")[-1])
        worker_input_fpath = str((self.directory / "_data" / f"worker-{identifier}.json").relative_to(self.directory))
        dataset_path = str((self.directory / "_data" / f"{identifier}.xyz").relative_to(self.directory))

        jobscript_fname = f"run-{uid}.script"
        self.scheduler.job_name = uid + "-" + batch_name
        self.scheduler.script = self.directory / jobscript_fname

        self.scheduler.user_commands = "gdp -p {} compute {} --batch {} --spawn\n".format(
            worker_input_fpath,
            dataset_path,
            batch_number,
        )

        curr_indices, curr_wdirs, driver_indices, rng_states, curr_frames = batch

        func_to_execute = functools.partial(
            run_computation_in_commandline,
            identifier=identifier,
            structures=curr_frames,
            computation_dirnames=curr_wdirs,
            rng_states=rng_states,
            drivers=self._drivers,
            driver_indices=driver_indices,
            directory=self.directory,
            share_wdir=self._share_wdir,
            print_period=self.print_period,
            print_func=self._print,
        )

        self.scheduler.write()
        job_id = self.scheduler.submit(func_to_execute=func_to_execute)
        self._print(f"{self.directory.name} JOBID: {job_id}")

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def _check_job_convergence(self, job: JobRecord) -> bool:
        wdir_names = job.wdir_names
        if not self._share_wdir:
            wdir_existence = [(self.directory / x).exists() for x in wdir_names]
            nwdir_exists = sum(1 for x in wdir_existence if x)
            self._print(f"progress: {nwdir_exists}/{len(wdir_existence)}")

            # Need the driver indices to check convergence
            # For single-driver, always use _drivers[0]
            # For multi-driver, look up from wdir name -> task plan index
            for wdir_name in wdir_names:
                if not (self.directory / wdir_name).exists():
                    return False
                # Determine driver index from wdir name
                wdir = self.directory / wdir_name
                try:
                    task_idx = int(wdir_name[4:])  # "cand{idx}"
                except ValueError:
                    return False
                if hasattr(self, "_task_plan") and task_idx < len(self._task_plan):
                    di = self._task_plan[task_idx][0]
                    d = self._drivers[di]
                else:
                    d = self._drivers[0]  # fallback for single-driver mode
                d.directory = wdir
                if not d.read_convergence():
                    self._print(f"Found unfinished computation at {wdir_name}")
                    return False
            return True
        else:
            # share_wdir: check cache file
            # Find the identifier from the job
            cache_fpath = self.directory / "_data" / f"{job.md5}_cache.xyz"
            if cache_fpath.exists():
                cache_frames = read(cache_fpath, ":")
                cache_wdirs = [a.info["wdir"] for a in cache_frames]
                if set(wdir_names) == set(cache_wdirs):
                    return True
                else:
                    self._print(f"Found unfinished computation at cand{len(cache_wdirs)}")
                    return False
            return False

    def _resubmit_job(self, job: JobRecord):
        self._print(f"RESUBMIT: {str(job.gdir)}")
        identifier = job.md5
        curr_batch = job.group_number

        frames = read(self.directory / "_data" / f"{identifier}.xyz", ":")
        cache_identifier, cache_frames, cache_batches = self.prepare_batches(frames)
        assert cache_identifier == identifier, "Inconsistent identifiers for the input structure."

        batch = cache_batches[curr_batch]
        curr_indices, curr_wdirs, driver_indices, rng_states, curr_frames = batch

        func_to_execute = functools.partial(
            run_computation_in_commandline,
            identifier=cache_identifier,
            structures=curr_frames,
            computation_dirnames=curr_wdirs,
            rng_states=rng_states,
            drivers=self._drivers,
            driver_indices=driver_indices,
            directory=self.directory,
            share_wdir=self._share_wdir,
            print_period=self.print_period,
            print_func=self._print,
        )
        job_id = self.scheduler.submit(func_to_execute=func_to_execute)
        self._print(f"{job.gdir} is re-submitted with JOBID: {job_id}...")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def _do_retrieve(
        self,
        include_retrieved: bool = False,
        given_wdirs: list[str] = None,
        use_archive: bool = False,
        *args,
        **kwargs,
    ):
        self._print(f"<<-- {self.__class__.__name__}+retrieve -->>")

        self._info_data = self._read_cached_info()

        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self.job_store.get_unretrieved()
        else:
            unretrieved_jobs = self.job_store.get_finished()

        unretrieved_identifiers = []
        for job in unretrieved_jobs:
            unretrieved_identifiers.append(job.md5)
            unretrieved_wdirs_.extend(self.directory / w for w in job.wdir_names)

        unretrieved_wdirs = []
        if given_wdirs is not None:
            for wdir in unretrieved_wdirs_:
                if wdir.name in given_wdirs:
                    unretrieved_wdirs.append(wdir)
        else:
            unretrieved_wdirs = unretrieved_wdirs_

        results = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            if not self._share_wdir:
                archive_path = (self.directory / "cand.tgz").absolute()
                if not archive_path.exists():
                    results = self._read_results(unretrieved_wdirs)
                else:
                    self._print("read archived data...")
                    results = self._read_results(unretrieved_wdirs, archive_path=archive_path)

                if use_archive and not archive_path.exists():
                    self._print("archive computation folders...")
                    with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
                        for w in unretrieved_wdirs:
                            tar.add(w, arcname=w.name)
                    for w in unretrieved_wdirs:
                        shutil.rmtree(w)
            else:
                cache_frames = []
                for identifier in unretrieved_identifiers:
                    cache_frames.extend(read(self.directory / "_data" / f"{identifier}_cache.xyz", ":"))
                wdir_names = [x.name for x in unretrieved_wdirs]
                results_ = [a for a in cache_frames if a.info["wdir"] in wdir_names]
                results = [[a] for a in results_]
                if self._retain_info:
                    info_keys, info_data = self._read_cached_xinfo()
                    retained_keys = [k for k in info_keys if k not in self.reserved_keys]
                    for i, traj_frames in enumerate(results):
                        retained_dict = {
                            k: v for k, v in zip(info_keys, info_data[i]) if k in retained_keys and v is not None
                        }
                        traj_frames[0].info.update(retained_dict)
                results = results_

        for job in unretrieved_jobs:
            self.job_store.mark_retrieved(job.gdir)

        return results

    def _read_results(
        self,
        unretrieved_wdirs: list[pathlib.Path],
        archive_path: pathlib.Path = None,
    ):
        with CustomTimer(name="read-results", func=self._print):
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(
                    self._drivers,
                    wdir,
                    info_data=self._info_data,
                    archive_path=archive_path,
                )
                for wdir in unretrieved_wdirs
            )

            if self._retain_info:
                info_keys, info_data = self._read_cached_xinfo()
                retained_keys = [k for k in info_keys if k not in self.reserved_keys]
                for i, traj_frames in enumerate(results_):
                    retained_dict = {
                        k: v for k, v in zip(info_keys, info_data[i]) if k in retained_keys and v is not None
                    }
                    traj_frames[0].info.update(retained_dict)

            results = []
            for i, traj_frames in enumerate(results_):
                if traj_frames:
                    results.append(traj_frames)
                else:
                    self._print(f"Found empty calculation at {str(self.directory)} with cand{i}...")

            if results:
                self._print(f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}")

        return results

    @staticmethod
    def _iread_results(drivers, wdir, info_data: dict = None, archive_path: pathlib.Path = None) -> list[Atoms]:
        """Extract results from a single directory.

        Must be a staticmethod — may be pickled by joblib for parallel execution.
        """
        # Determine which driver to use from the wdir name
        if isinstance(drivers, list) and len(drivers) > 1:
            try:
                task_idx = int(wdir.name[4:])
            except ValueError:
                task_idx = 0
            driver = drivers[task_idx % len(drivers)]
        else:
            driver = drivers[0] if isinstance(drivers, list) else drivers

        driver.directory = wdir
        confid_ = int(wdir.name.strip("cand").split("_")[0])
        if info_data is not None and len(info_data) > confid_:
            cache_confid = int(info_data[confid_][2]) if len(info_data[confid_]) > 2 else -1
            confid = cache_confid if cache_confid >= 0 else confid_
        else:
            confid = confid_

        traj_frames = driver.read_trajectory(add_step_info=True, archive_path=archive_path)
        for a in traj_frames:
            a.info["confid"] = confid
            a.info["wdir"] = str(wdir.name)

        return traj_frames

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        worker_params = {}
        if self.potter is not None:
            worker_params["potter"] = self.potter.as_dict()
        if self._drivers:
            worker_params["driver"] = self._drivers[0].as_dict()
        else:
            worker_params["driver"] = {}
        worker_params["scheduler"] = self.scheduler.as_dict()

        worker_params = copy.deepcopy(worker_params)
        worker_params["batchsize"] = self.batchsize
        worker_params["share_wdir"] = self._share_wdir
        worker_params["retain_info"] = self._retain_info

        return worker_params
