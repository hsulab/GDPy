import dataclasses
from typing import Optional

from ase.calculators.lammps import unitconvert

from ..driver import Controller


@dataclasses.dataclass
class CGMinimiser(Controller):
    name: str = "cg"

    def __post_init__(self):
        maxstep = self.params.get("maxstep", 0.2)
        maxstep = unitconvert.convert(maxstep, "distance", "metal", self.units)
        input_line = "min_style  cg\n"
        input_line += f"min_modify dmax {maxstep}"
        self.conv_params = dict(input_line=input_line)


@dataclasses.dataclass
class FireMinimizer(Controller):
    name: str = "fire"

    def __post_init__(self):
        integrator = self.params.get("integrator", "verlet")
        tmax = self.params.get("tmax", 4)
        input_line = "min_style  fire\n"
        input_line += f"min_modify integrator {integrator} tmax {tmax}"
        self.conv_params = dict(input_line=input_line)


@dataclasses.dataclass
class MDController(Controller):
    name: str = "md"
    timestep: float = 1.0
    temperature: float = 300.0
    temperature_end: Optional[float] = None
    pressure: float = 1.0
    pressure_end: Optional[float] = None
    fix_com: bool = True

    def __post_init__(self):
        self.timestep = unitconvert.convert(self.timestep, "time", "real", self.units)
        self.temperature = unitconvert.convert(self.temperature, "temperature", "real", self.units)
        if self.temperature_end is not None:
            self.temperature_end = unitconvert.convert(self.temperature_end, "temperature", "real", self.units)
        else:
            self.temperature_end = self.temperature
        if not (self.temperature > 0.0):
            raise Exception(f"MDController temperature `{self.temperature}` must be greater than 0.")
        assert self.temperature_end is not None
        if not (self.temperature_end > 0.0):
            raise Exception(f"MDController temperature_end `{self.temperature_end}` must be greater than 0.")
        self.pressure = unitconvert.convert(self.pressure, "pressure", "metal", self.units)
        if self.pressure_end is not None:
            self.pressure_end = unitconvert.convert(self.pressure_end, "pressure", "metal", self.units)
        else:
            self.pressure_end = self.pressure
        input_line = ""
        if self.fix_com:
            input_line += "fix  fix_com {group} recenter INIT INIT INIT\n"
        input_line += f"\ntimestep {self.timestep}\n"
        self.conv_params = dict(input_line=input_line)


@dataclasses.dataclass
class Verlet(MDController):
    name: str = "verlet"

    def __post_init__(self):
        super().__post_init__()
        input_line = "fix {fix_id:>24s} {group} nve"
        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]


@dataclasses.dataclass
class LangevinThermostat(MDController):
    name: str = "langevin"

    def __post_init__(self):
        super().__post_init__()
        friction = self.params.get("friction", 0.01)
        assert friction is not None
        damp = unitconvert.convert(1.0 / friction, "time", "real", self.units)
        friction_seed = self.params.get("friction_seed", None)
        input_line = "fix {fix_id:>24s}0 {group} nve\n"
        input_line += "fix {fix_id:>24s}1 {group} langevin "
        input_line += f"{self.temperature} {self.temperature_end} {damp} "
        if friction_seed is not None:
            input_line += f"{friction_seed}"
        else:
            input_line += "{seed}"
        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]


@dataclasses.dataclass
class NoseHooverChainThermostat(MDController):
    name: str = "nose_hoover_chain"

    def __post_init__(self):
        super().__post_init__()
        Tdamp = self.params.get("Tdamp", unitconvert.convert(self.timestep * 100.0, "time", self.units, "real"))
        assert Tdamp is not None
        Tdamp = unitconvert.convert(Tdamp, "time", "real", self.units)
        input_line = "fix {fix_id:>24s} {group} nvt temp "
        input_line += f"{self.temperature} {self.temperature_end} {Tdamp}"
        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]


@dataclasses.dataclass
class ParrinelloRahmanBarostat(MDController):
    name: str = "parrinello_rahman"

    def __post_init__(self):
        super().__post_init__()
        Tdamp = self.params.get("Tdamp", unitconvert.convert(self.timestep * 100.0, "time", self.units, "real"))
        assert Tdamp is not None
        Tdamp = unitconvert.convert(Tdamp, "time", "real", self.units)
        Pdamp = self.params.get("Pdamp", unitconvert.convert(self.timestep * 1000.0, "time", self.units, "real"))
        assert Pdamp is not None
        Pdamp = unitconvert.convert(Pdamp, "time", "real", self.units)
        isotropic = self.params.get("isotropic", True)
        assert isotropic is not None
        isotropic = "iso" if isotropic else "aniso"
        input_line = "fix {fix_id:>24s} {group} npt temp "
        input_line += f"{self.temperature} {self.temperature_end} {Tdamp} "
        input_line += f"{isotropic} {self.pressure} {self.pressure_end} {Pdamp}"
        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]


controllers = dict(
    cg_min=CGMinimiser,
    fire_min=FireMinimizer,
    verlet_nve=Verlet,
    langevin_nvt=LangevinThermostat,
    nose_hoover_chain_nvt=NoseHooverChainThermostat,
    parrinello_rahman_npt=ParrinelloRahmanBarostat,
)

default_controllers = dict(
    min=FireMinimizer,
    nve=Verlet,
    nvt=LangevinThermostat,
    npt=ParrinelloRahmanBarostat,
)
