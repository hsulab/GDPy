from typing import Optional

from typing import Union

from ase import Atoms

from .driver import BaseDriver
from .lammps.calculator import _read_a_single_trajectory
from .lammps.constants import ASELMPCONFIG


class ReplicaDriver(BaseDriver):
    """Multi-replica simulation driver.

    Wraps a single backend driver to support multi-replica simulations.
    Each replica writes to a ``replica.{i}`` subdirectory under the driver's
    working directory.

    Currently optimized for LAMMPS with ``-partition`` + ``temper`` (replica
    exchange).  When :attr:`num_replicas` > 1 the LAMMPS calculator is
    configured with ``-partition`` and per-replica file paths.

    """

    name = "replica"

    default_task = "md"
    supported_tasks = ["md", "rex"]

    def __init__(
        self,
        calc,
        params: dict,
        directory="./",
        *args,
        **kwargs,
    ):
        self._num_replicas = params.get("num_replicas", 1)
        super().__init__(calc, params, directory=directory, *args, **kwargs)

    @property
    def _padding(self) -> int:
        """Zero-padding width for replica directory indices."""
        n = self._num_replicas
        if n <= 99:
            return 2
        return 4

    # --- directory management ------------------------------------------------

    def _ensure_replica_dirs(self):
        """Create replica subdirectories inside the working directory."""
        for i in range(self._num_replicas):
            rep_dir = self.directory / f"replica.{i}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            # Share the prism file so each replica can read the cell
            prism_src = self.directory / ASELMPCONFIG.prism_filename
            if prism_src.exists():
                prism_dst = rep_dir / ASELMPCONFIG.prism_filename
                if not prism_dst.exists():
                    import shutil

                    shutil.copy2(prism_src, prism_dst)

    # --- checkpoint / restart -----------------------------------------------

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        verified = self.directory.exists()
        if verified:
            checkpoints = list(self.directory.glob("restart.*.data"))
            if not checkpoints:
                verified = False
        return verified

    def _save_checkpoint(self):
        """Lightweight checkpoint — only save restart-critical files."""
        prev_wdirs = sorted(
            self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"),
            key=lambda p: int(p.name[:4]),
        )
        has_restart_files = bool(list(self.directory.glob("restart.*.data")))
        if not has_restart_files:
            return self.directory
        curr_index = len(prev_wdirs)
        curr_wdir = self.directory / f"{str(curr_index).zfill(4)}.run"
        curr_wdir.mkdir()
        for pattern in ("restart.*.data", "in.lammps", ASELMPCONFIG.log_filename):
            for f in self.directory.glob(pattern):
                if f.is_file():
                    import shutil

                    shutil.copy2(f, curr_wdir)
        return curr_wdir

    # --- simulation ----------------------------------------------------------

    def _create_dynamics(self, atoms: Atoms) -> list[str]:
        lines = []
        if self.setting.task == "md":
            use_lammps_vinit = getattr(self.setting, "use_lammps_vinit", True)
            if use_lammps_vinit:
                velocity_seed = self.setting.velocity_seed or self.random_seed
                self._print(f"ReplicaDriver's velocity_seed: {velocity_seed}")
                if self.setting.ensemble == "rex" or self._num_replicas > 1:
                    line = f"velocity        mobile create $t {velocity_seed} dist gaussian "
                else:
                    line = f"velocity        mobile create {self.setting.temp} {velocity_seed} dist gaussian "
                if getattr(self.setting, "remove_translation", True):
                    line += "mom yes "
                if getattr(self.setting, "remove_rotation", True):
                    line += "rot yes "
                if atoms.get_kinetic_energy() > 0.0:
                    if getattr(self.setting, "ignore_atoms_velocities", False):
                        atoms.set_momenta(__import__("numpy").zeros(atoms.positions.shape))
                        lines.append(line)
                    else:
                        lines.append("# Use atoms' velocities.")
                else:
                    atoms.set_momenta(__import__("numpy").zeros(atoms.positions.shape))
                    lines.append(line)
            else:
                self._prepare_velocities(atoms, self.setting.velocity_seed, self.setting.ignore_atoms_velocities)
        dynamics = self.setting.get_simulation_inputs(random_seed=self.random_seed, group="mobile")
        lines.extend(dynamics)
        return lines

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        self._ensure_replica_dirs()

        run_params = self.setting.get_init_params()
        run_params.update(**self.setting.get_run_params(**kwargs))

        self._print(f"num_replicas: {self._num_replicas}")

        is_continue = ckpt_wdir is not None
        if is_continue:
            checkpoints = sorted(
                list(ckpt_wdir.glob("restart.*.data")),
                key=lambda x: int(x.name.split(".")[1]),
            )
            if checkpoints:
                target_steps = run_params["steps"]
                finish_steps = int(checkpoints[-1].name.split(".")[1])
                remain_steps = target_steps - finish_steps
                run_params.update(read_restart=str(checkpoints[-1].resolve()), steps=remain_steps)
                self._print(f"restart from step {finish_steps}, remaining steps {remain_steps}")

        dynamics = self._create_dynamics(atoms)

        self.calc.set(
            task=self.setting.task,
            dump_period=self.setting.dump_period,
            ckpt_period=self.setting.ckpt_period,
            dynamics=dynamics,
            steps=run_params["steps"],
            constraint=run_params.get("constraint", self.setting.constraint),
            etol=run_params.get("etol", 0.0),
            ftol=run_params.get("ftol", 0.0),
            read_restart=run_params.get("read_restart", None),
            extra_fix=run_params.get("extra_fix", []),
            neighbor=run_params.get("neighbor", "2.0 bin"),
            neigh_modify=run_params.get("neigh_modify", "every 10 check yes"),
            halt="",
            num_replicas=self._num_replicas,
        )
        atoms.calc = self.calc

        try:
            _ = atoms.get_forces()
        except Exception:
            import traceback

            self._debug(traceback.format_exc())

    # --- trajectory reading --------------------------------------------------

    def _read_replica_trajectory(self, replica_idx: int) -> list[Atoms]:
        rep_dir = self.directory / f"replica.{replica_idx}"
        return _read_a_single_trajectory(
            wdir=rep_dir,
            mdir=self.directory,
            units=self.calc.units,
            print_func=self._print,
            debug_func=self._debug,
        )

    def read_trajectory(  # type: ignore[override]
        self,
        replica_idx: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Union[list[Atoms], list[list[Atoms]]]:
        if replica_idx is not None:
            return self._read_replica_trajectory(replica_idx)

        all_trajs = []
        for i in range(self._num_replicas):
            traj = self._read_replica_trajectory(i)
            all_trajs.append(traj)
        return all_trajs

    # --- convergence ---------------------------------------------------------

    def read_convergence_from_logfile(self, *args, **kwargs):
        log_fpath = self.directory / ASELMPCONFIG.log_filename
        if log_fpath.exists() and log_fpath.stat().st_size != 0:
            with open(log_fpath) as fopen:
                lines = fopen.readlines()
            end_line = lines[-1].strip()
            if end_line.startswith("Total wall time:"):
                return True
            elif end_line.startswith("Last command: run"):
                with open(self.directory / "EARLYSTOP", "w") as fopen:
                    fopen.write("")
                return True
        return False
