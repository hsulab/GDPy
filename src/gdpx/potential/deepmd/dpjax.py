#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
from typing import Mapping, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from deepmd_jax import utils as djutils
from jax import jit, vmap


def compute_lattice_candidate(boxes, rcut):  # boxes (nframes,3,3)
    N = 2  # This algorithm is heuristic and subject to change. Increase N in case of missing neighbors.
    ortho = not vmap(lambda box: box - jnp.diag(jnp.diag(box)))(boxes).any()
    recp_norm = jnp.linalg.norm((jnp.linalg.inv(boxes)), axis=-1)  # (nframes,3)
    n = np.ceil(rcut * recp_norm - 0.5).astype(int).max(0)  # (3,)
    lattice_cand = jnp.stack(
        np.meshgrid(
            range(-n[0], n[0] + 1),
            range(-n[1], n[1] + 1),
            range(-n[2], n[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    trial_points = jnp.stack(
        np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), np.arange(-N, N + 1)),
        axis=-1,
    ).reshape(-1, 3) / (2 * N)
    is_neighbor = (
        jnp.linalg.norm(
            (lattice_cand[:, None] - trial_points)[None] @ boxes[:, None], axis=-1
        )
        < rcut
    )  # (nframes,l,t)
    lattice_cand = np.array(lattice_cand[is_neighbor.any((0, 2))])
    lattice_max = is_neighbor.sum(1).max().item()
    # print(
    #     "# Lattice vectors for neighbor images: Max %d out of %d condidates."
    #     % (lattice_max, len(lattice_cand))
    # )
    return {
        "lattice_cand": tuple(map(tuple, lattice_cand)),
        "lattice_max": lattice_max,
        "ortho": ortho,
    }


def get_type_sort_and_count(atoms: Atoms, type_map: Mapping[str, int]):
    """"""
    atypes = np.array([type_map[s] for s in atoms.get_chemical_symbols()])
    atype_sort = atypes.argsort(kind="stable")
    atype_rsort = list(range(len(atype_sort)))
    for i in range(len(atype_rsort)):
        atype_rsort[atype_sort[i]] = i
    type_count = tuple(np.array([(atypes == i).sum() for i in range(atypes.max() + 1)]))

    return atype_sort, atype_rsort, type_count


class DPJax(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, model, type_list, label="DPJax", *args, **kwargs):
        """"""
        super().__init__(label=label, *args, **kwargs)

        self.model_fpath = model
        self.type_list = type_list
        self.type_map = {k: v for v, k in enumerate(self.type_list)}

        self._model = None
        self._variables = None

        self._atype_sort = None
        self._atype_rsort = None  # reverse sort
        self._type_count = None

        self._boxes = None
        self._lattice_args = None

        return

    def __init_model__(self):
        """"""
        self._model, self._variables = djutils.load_model(self.model_fpath)

        return

    def prepare_jaxmd_simulation(self, atoms: Atoms):
        """Prepare a jax_md simulation with a flax-based NN potential."""
        self.__init_model__()

        self._atype_sort, self._atype_rsort, self._type_count = (
            get_type_sort_and_count(atoms, self.type_map)
        )
        # print(f"{self._model.params}")

        box = np.array(atoms.get_cell(complete=True))
        rcut = self._model.params["rcut"]

        lattice_args = compute_lattice_candidate(box[None], rcut)

        static_args = nn.FrozenDict({"type_count": self._type_count, "lattice": lattice_args})

        def energy_fn(coord, nbrs_nm):
            return self._model.apply(
                self._variables, coord, box, static_args=static_args, nbrs_nm=nbrs_nm
            )[0]

        return jax.jit(energy_fn), self._atype_sort, self._atype_rsort

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """"""
        super().calculate(atoms, properties, system_changes)
        if self._model is None:
            self.__init_model__()
            self._compute_energy_and_force = functools.partial(
                jit,
                static_argnums=(
                    3,
                    4,
                ),
            )(vmap(self._model.energy_and_force, in_axes=(None, 0, 0, None)))
        else:
            ...

        if atoms is not None:
            # Update atype_sort only when chemical_symbols have been changed
            # print(f"{system_changes =}")
            if "numbers" in system_changes:
                self._atype_sort, self._atype_rsort, self._type_count = (
                    get_type_sort_and_count(atoms, self.type_map)
                )
            else:
                ...

            if "cell" in system_changes:
                self._boxes = np.array(atoms.get_cell(complete=True)).reshape(1, 3, 3)
                self._lattice_args = compute_lattice_candidate(
                    self._boxes, self._model.params["rcut"]
                )
            else:
                ...

            static_args = nn.FrozenDict(
                {"type_count": self._type_count, "lattice": self._lattice_args}
            )

            # TODO: shift coordinates?
            coord = atoms.get_positions().reshape(1, -1, 3)[:, self._atype_sort]
            e, f = self._compute_energy_and_force(
                self._variables, coord, self._boxes, static_args
            )

            self.results["energy"] = float(e[0])
            self.results["free_energy"] = float(e[0])
            self.results["forces"] = np.array(f[0])[self._atype_rsort]
        else:
            raise RuntimeError("This should not happen!!!")

        return


if __name__ == "__main__":
    ...
