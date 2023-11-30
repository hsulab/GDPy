"""ASE calculator interface module."""

import copy
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from deepmd import (
    DeepPotential,
)

if TYPE_CHECKING:
    from ase import (
        Atoms,
    )

__all__ = ["DP"]


class DP(Calculator):
    """Implementation of ASE deepmd calculator.

    Implemented propertie are `energy`, `forces` and `stress`

    Parameters
    ----------
    model : Union[str, Path]
        path to the model
    label : str, optional
        calculator label, by default "DP"
    type_dict : Dict[str, int], optional
        mapping of element types and their numbers, best left None and the calculator
        will infer this information from model, by default None

    Examples
    --------
    Compute potential energy

    >>> from ase import Atoms
    >>> from deepmd.calculator import DP
    >>> water = Atoms('H2O',
    >>>             positions=[(0.7601, 1.9270, 1),
    >>>                        (1.9575, 1, 1),
    >>>                        (1., 1., 1.)],
    >>>             cell=[100, 100, 100],
    >>>             calculator=DP(model="frozen_model.pb"))
    >>> print(water.get_potential_energy())
    >>> print(water.get_forces())

    Run BFGS structure optimization

    >>> from ase.optimize import BFGS
    >>> dyn = BFGS(water)
    >>> dyn.run(fmax=1e-6)
    >>> print(water.get_positions())
    """

    name = "DP"
    implemented_properties = ["energy", "free_energy", "forces", "virial", "stress"]

    def __init__(
        self,
        model: Union[str, "Path"],
        label: str = "DP",
        type_dict: Dict[str, int] = None,
        **kwargs,
    ) -> None:
        Calculator.__init__(self, label=label, **kwargs)

        self.model_path = str(Path(model).resolve())
        self.type_dict = type_dict

        # - lazy init
        self.dp = None # will init when first calculate

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces", "virial"],
        system_changes: List[str] = all_changes,
    ):
        """Run calculation with deepmd model.

        Parameters
        ----------
        atoms : Optional[Atoms], optional
            atoms object to run the calculation on, by default None
        properties : List[str], optional
            unused, only for function signature compatibility,
            by default ["energy", "forces", "stress"]
        system_changes : List[str], optional
            unused, only for function signature compatibility, by default all_changes
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        if self.dp is None:
            self.dp = DeepPotential(self.model_path)
            if self.type_dict is None:
                self.type_dict = dict(
                    zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
                )

        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        e, f, v = self.dp.eval(coords=coord, cells=cell, atom_types=atype)
        self.results["energy"] = e[0][0]
        # see https://gitlab.com/ase/ase/-/merge_requests/2485
        self.results["free_energy"] = e[0][0]
        self.results["forces"] = f[0]
        self.results["virial"] = v[0].reshape(3, 3)

        # convert virial into stress for lattice relaxation
        if "stress" in properties:
            if sum(atoms.get_pbc()) > 0:
                # the usual convention (tensile stress is positive)
                # stress = -virial / volume
                stress = -0.5 * (v[0].copy() + v[0].copy().T) / atoms.get_volume()
                # Voigt notation
                self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError


class BatchDP:

    name = "DP"
    implemented_properties = ["energy", "free_energy", "forces", "virial", "stress"]

    """Evaluate a batch of structures.

    This is useful in ASE NEB calculation.

    """

    def __init__(
        self,
        model: Union[str, "Path"],
        label: str = "DP",
        type_dict: Dict[str, int] = None,
        **kwargs,
    ) -> None:
        """"""
        self.model_path = str(Path(model).resolve())
        self.type_dict = type_dict

        # - lazy init
        self.dp = None # will init when first calculate

        return
    
    def calculate(
        self,
        frames: Optional[List["Atoms"]] = None,
        properties: List[str] = ["energy", "forces", "virial"],
        system_changes: List[str] = all_changes,
    ):
        """"""
        if self.dp is None:
            self.dp = DeepPotential(self.model_path)
            if self.type_dict is None:
                self.type_dict = dict(
                    zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
                )
        
        # - convert atoms
        # TODO: assume it is not mixed-type
        coords, cells, atypes = [], [], []
        for atoms in frames:
            curr_coord, curr_cell, curr_atype = self._convert_from_ase_to_dp(atoms)
            coords.append(curr_coord)
            cells.append(curr_cell)
            atypes.append(curr_atype)
        coords = np.array(coords)
        cells = np.array(cells)
        atype = atypes[0]

        # - calc
        e, f, v = self.dp.eval(coords=coords, cells=cells, atom_types=atype)

        # - convert results
        self.results = {}
        self.results["energy"] = [x[0] for x in e] # List[float]
        self.results["free_energy"] = [x[0] for x in e] # List[float]
        self.results["forces"] = f.squeeze
        self.results["virial"] = [x.reshape(3, 3) for x in v]

        # convert virial into stress for lattice relaxation
        if "stress" in properties:
            self.results["stress"] = []
            for i, atoms in enumerate(frames):
                if sum(atoms.get_pbc()) > 0:
                    # the usual convention (tensile stress is positive)
                    # stress = -virial / volume
                    stress = -0.5 * (v[i].copy() + v[i].copy().T) / atoms.get_volume()
                    # Voigt notation
                    self.results["stress"].append(stress.flat[[0, 4, 8, 5, 2, 1]])
                else:
                    raise PropertyNotImplementedError
        
        return
    
    def _convert_from_ase_to_dp(self, atoms):
        """"""
        coord = atoms.get_positions().reshape([1, -1])
        if sum(atoms.get_pbc()) > 0:
            cell = atoms.get_cell(complete=True).reshape([1, -1])
        else:
            cell = None
        symbols = atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]

        return coord, cell, atype


if __name__ == "__main__":
    ...