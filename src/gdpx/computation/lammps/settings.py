import dataclasses
from typing import Optional

from ..driver import DriverSetting
from .controllers import controllers, default_controllers


@dataclasses.dataclass
class LmpDriverSetting(DriverSetting):
    units: str = "metal"
    ensemble: str = "nve"
    controller: dict = dataclasses.field(default_factory=dict)
    fix_com: bool = False
    use_lammps_vinit: bool = True
    emax: Optional[float] = 0.0
    fmax: Optional[float] = 0.05
    neighbor: str = "2.0 bin"
    neigh_modify: Optional[str] = "every 10 check yes"
    extra_fix: list[str] = dataclasses.field(default_factory=list)
    plumed: Optional[str] = None
    num_replicas: int = 1
    replica_temperatures: Optional[list[float]] = None
    temper_period: int = 500
    temper_seed: Optional[int] = None
    temper_freq: int = 127

    def __post_init__(self):
        if self.task == "min":
            self._internals.update(etol=self.emax, ftol=self.fmax)
        if self.task == "md":
            self._internals.update(plumed=self.plumed)
        self._internals.update(
            neighbor=self.neighbor,
            neigh_modify=self.neigh_modify,
            extra_fix=self.extra_fix,
            num_replicas=self.num_replicas,
            replica_temperatures=self.replica_temperatures,
            temper_period=self.temper_period,
            temper_seed=self.temper_seed,
        )

    def get_simulation_inputs(self, random_seed: int, group: str = "mobile") -> list[str]:
        _init_params = {}
        if self.task == "min":
            suffix = self.task
        elif self.task == "md":
            suffix = self.ensemble
            _init_params.update(
                timestep=self.timestep,
                temperature=self.temp,
                temperature_end=self.tend if self.tend is not None else self.temp,
                pressure=self.press,
                pressure_end=self.pend if self.pend is not None else self.press,
                fix_com=self.fix_com,
            )
        else:
            suffix = self.task
        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]
        _init_params.update(**self.controller)
        controller = cont_cls(units=self.units, **_init_params)
        _init_placeholders = dict(
            fix_id="controller",
            group=group,
            seed=random_seed,
            steps=self.steps,
        )
        input_line = controller.conv_params["input_line"].format(**_init_placeholders)
        lines = [input_line]
        return lines

    def get_run_params(self, *args, **kwargs):
        fmax_ = kwargs.pop("fmax", self.fmax)
        emax_ = kwargs.pop("emax", self.emax)
        if emax_ is None:
            emax_ = 0.0
        if fmax_ is None:
            fmax_ = 0.0
        steps_ = kwargs.pop("steps", self.steps)
        run_params = dict(
            steps=steps_,
            constraint=kwargs.get("constraint", self.constraint),
            etol=emax_,
            ftol=fmax_,
        )
        run_params.update(**kwargs)
        return run_params
