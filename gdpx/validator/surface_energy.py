#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Union, List

import numpy as np

from ase import units
from ase import Atoms

from ..data.array import AtomsNDArray
from .validator import AbstractValidator

EVA2TOJNM2 = (1./(units.kJ/1000.))/((1./units.m)**2)


class SurfaceEnergyValidator(AbstractValidator):

    """This computes the surface energy of symmetric surface energies.

    TODO:
        Asymmetric surfaces.

    """

    def __init__(self, nsides=1, directory: Union[str, pathlib.Path] = "./", *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.nsides = nsides

        return

    def _process_data(self, data) -> List[List[Atoms]]:
        """"""
        data = AtomsNDArray(data)

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2: # assume it is from extract_cache...
            data = data.tolist()
        elif data.ndim == 3: # assume it is from a compute node...
            data_ = []
            for d in data[:]: # TODO: add squeeze method?
                data_.extend(d)
            data = data_
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data
    
    def run(self, dataset, worker=None, *args, **kwargs):
        """"""
        ref_grp = dataset.get("reference", None)
        if ref_grp is not None:
            ref_data = self._compute_surface_energy(
                self._process_data(ref_grp["bulk"]), 
                self._process_data(ref_grp["surfaces"]),
                self.nsides
            )
            self._write_data(ref_data, prefix="ref-")

        pre_grp = dataset.get("prediction", None)
        if pre_grp is not None:
            pre_data = self._compute_surface_energy(
                self._process_data(pre_grp["bulk"]), 
                self._process_data(pre_grp["surfaces"]),
                self.nsides
            )
            self._write_data(pre_data, prefix="pre-")

        return
    
    def _compute_surface_energy(self, bulk: List[List[Atoms]], surfaces: List[List[Atoms]], nsides=1):
        """"""
        assert len(bulk) == 1, "Only support one bulk."
        bulk_ene = bulk[0][-1].get_potential_energy()
        natoms_bulk = len(bulk[0][-1])

        data = []
        for surf in surfaces:
            # TODO: assert z-axis is perpendicular to the surface plane
            surf = [s for s in surf if s is not None]
            natoms_surf = len(surf[0])
            n_units = int(natoms_surf/natoms_bulk) # TODO: check integer units
            cell = surf[0].get_cell(complete=True)
            surf_area = np.linalg.norm(np.cross(cell[0], cell[1]))
            fin_ene = surf[-1].get_potential_energy()
            relaxed_ene = fin_ene - surf[0].get_potential_energy()
            d_ene = fin_ene - bulk_ene*n_units
            surf_ene = d_ene/(2*surf_area) + relaxed_ene/(nsides*surf_area)
            surf_ene_jnm2 = surf_ene*EVA2TOJNM2
            data.append([surf_area, d_ene, relaxed_ene, surf_ene, surf_ene_jnm2])

        return data
    
    def _write_data(self, data, prefix=""):
        """"""
        content = ("#{:>11s}  "+"{:>12s}  "*5+"\n").format(
            "N", "Area [A2]", "Delta [eV]", "RelEne [eV]", "Sigma [eV/A2]", "Sigma [J/m2]" 
        )
        for i, curr_data in enumerate(data):
            content += ("{:>12d}  "+"{:>12.4f}  "*5+"\n").format(i, *curr_data)
        self._print(f"\n{prefix}\n"+content)

        data_fpath = self.directory / f"{prefix}surfene.dat"
        with open(data_fpath, "w") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...