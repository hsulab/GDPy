#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, atoms_too_close_two_sets


class MirrorMutation(OffspringCreator):
    """The mirror mutation with tags.

    This mutation mirrors half of the cluster in a
    randomly oriented cutting plane discarding the other half.

    """

    def __init__(
        self,
        blmin,
        n_top,
        reflect=False,
        use_tags=True,
        rng=np.random.default_rng(),
        verbose=False,
    ):
        """Initialise the mutation.

        Args:
            blmin: Dictionary defining the minimum allowed
                distance between atoms.

            n_top: Number of atoms the GA optimizes.

            reflect: Defines if the mirrored half is also reflected
                perpendicular to the mirroring plane.

            use_tags: Whether applying the mutation to tagged fragments.

            rng: The random number generator.

        """
        super().__init__(verbose=verbose)
        self.descriptor = "MirrorMutation"
        self.min_inputs = 1

        self.blmin = blmin
        self.n_top = n_top
        self.reflect = reflect

        self.use_tags = use_tags
        if self.use_tags:
            raise NotImplementedError(
                "MirrorMutataion does not support molecules."
            )

        self.rng = rng

        return

    def get_new_individual(self, parents: list[Atoms]):
        """"""
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: mirror_none"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

        return self.finalize_individual(indi), "mutation: mirror"

    def mutate(self, atoms: Atoms) -> Optional[Atoms]:
        """Do the mutation of the atoms input."""
        previous_tags = atoms.get_tags()

        reflect = self.reflect
        tc = True
        
        slab = atoms[0 : len(atoms) - self.n_top]
        assert isinstance(slab, Atoms)

        top = atoms[len(atoms) - self.n_top : len(atoms)]
        assert isinstance(top, Atoms)

        # The mirror mutation can be applied only to atoms with the same type.
        # TODO: We need figure out how to properly handle the tags if there are
        #       several atom types and the number of atoms are uneven in the 
        #       search region.
        num_atom_types = len(set(top.numbers))
        if num_atom_types > 1:
            return None

        num = top.numbers
        unique_types = list(set(num))
        nu = {u: sum(num == u) for u in unique_types}
        n_tries = 1000
        counter = 0
        changed = False

        tot = None
        while tc and counter < n_tries:
            counter += 1
            cand = top.copy()
            pos = cand.get_positions()

            cm = np.average(top.get_positions(), axis=0)

            # first select a randomly oriented cutting plane
            theta = np.pi * self.rng.random()
            phi = 2.0 * np.pi * self.rng.random()
            n = (
                np.cos(phi) * np.sin(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(theta),
            )
            n = np.array(n)

            # Calculate all atoms signed distance to the cutting plane
            D = []
            for i, p in enumerate(pos):
                d = np.dot(p - cm, n)
                D.append((i, d))

            # Sort the atoms by their signed distance
            D.sort(key=lambda x: x[1])
            nu_taken = {}

            # Select half of the atoms needed for a full cluster
            p_use = []
            n_use = []
            for i, d in D:
                if num[i] not in nu_taken.keys():
                    nu_taken[num[i]] = 0
                if nu_taken[num[i]] < nu[num[i]] / 2.0:
                    p_use.append(pos[i])
                    n_use.append(num[i])
                    nu_taken[num[i]] += 1

            # calculate the mirrored position and add these.
            pn = []
            for p in p_use:
                pt = p - 2.0 * np.dot(p - cm, n) * n
                if reflect:
                    pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
                pn.append(pt)

            n_use.extend(n_use)
            p_use.extend(pn)

            # In the case of an uneven number of
            # atoms we need to add one extra
            for n in nu:
                if nu[n] % 2 == 0:
                    continue
                while sum(n_use == n) > nu[n]:
                    for i in range(int(len(n_use) / 2), len(n_use)):
                        if n_use[i] == n:
                            del p_use[i]
                            del n_use[i]
                            break
                assert sum(n_use == n) == nu[n]

            # Make sure we have the correct number of atoms
            # and rearrange the atoms so they are in the right order
            for i in range(len(n_use)):
                if num[i] == n_use[i]:
                    continue
                for j in range(i + 1, len(n_use)):
                    if n_use[j] == num[i]:
                        tn = n_use[i]
                        tp = p_use[i]
                        n_use[i] = n_use[j]
                        p_use[i] = p_use[j]
                        p_use[j] = tp
                        n_use[j] = tn

            # Finally we check that nothing is too close in the end product.
            cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())

            tc = atoms_too_close(cand, self.blmin)
            if tc:
                continue
            tc = atoms_too_close_two_sets(slab, cand, self.blmin)

            if not changed and counter > n_tries // 2:
                reflect = not reflect
                changed = True

            tot = slab + cand
            tot.set_tags(previous_tags)

        if counter == n_tries:
            return None

        return tot


if __name__ == "__main__":
    ...
