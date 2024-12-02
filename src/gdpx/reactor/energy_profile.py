#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import pathlib
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import BPoly, CubicSpline

plt.style.use("presentation")

from ase import units
from ase.io import read, write
from ase.thermochemistry import HarmonicThermo, IdealGasThermo


def read_vibrations(wdir):
    """"""
    wdir = pathlib.Path(wdir)
    print(f"{str(wdir) =}")

    vibrations = []
    with open(wdir / "cp2k-VIBRATIONS-1.mol", "r") as fopen:
        start_freq = False
        while True:
            line = fopen.readline()
            if line.strip().startswith("[FREQ]"):
                start_freq = True
                line = fopen.readline()
            if line.strip().startswith("[FR-COORD]"):
                break
            if not line:
                break
            if start_freq:
                vibrations.append(float(line))
    vib_energies = np.array(vibrations) * units.invcm  # eV

    # remove imaginary frequencies
    vib_energies = np.array([v for v in vib_energies if v >= 0.0])

    return vib_energies


@dataclasses.dataclass
class ThermoStructure:

    structure: str

    frequency: str = ""

    energy_shift: float = 0.

    def __post_init__(self):
        """"""
        self.atoms = read(self.structure, "-1")
        self.energy = self.atoms.get_potential_energy()
        self.energy += self.energy_shift

        self.free_energy = self.energy

        return

    def compute_free_energies(self, temperature, pressure):
        """

        Args:
            temperature: K
            pressure: bar

        """
        if self.frequency:
            if isinstance(self.frequency, str):
                vib_energies = read_vibrations(self.frequency)
                thermo = HarmonicThermo(vib_energies)
                free_energy_correction = thermo.get_helmholtz_energy(
                    temperature=temperature
                )
            else:  # dict
                freq_wdir = self.frequency["wdir"]
                vib_energies = read_vibrations(freq_wdir)

                geometry = self.frequency.get("geometry", "nonlinear")
                sigma = self.frequency.get("sigma", None)
                spin = self.frequency.get("spin", 0)
                thermo = IdealGasThermo(
                    vib_energies=vib_energies,
                    geometry=geometry,
                    atoms=self.atoms,
                    symmetrynumber=sigma,
                    spin=spin,
                )
                free_energy_correction = thermo.get_gibbs_energy(
                    temperature=temperature, pressure=pressure * 1e5
                )
        else:
            free_energy_correction = 0.0

        self.free_energy = self.energy + free_energy_correction

        return


@dataclasses.dataclass
class ReactionData:

    eps: float = 0.25

    names: List[str] = dataclasses.field(default_factory=list)

    energies: List[float] = dataclasses.field(default_factory=list)

    coordinates: List[float] = dataclasses.field(default_factory=list)

    trajectory: dict = dataclasses.field(default_factory=dict)

    frequencies: list = dataclasses.field(default_factory=list)

    #: Whether the reverse the reaction progress.
    reverse: bool = False

    is_transitions: List[bool] = dataclasses.field(default_factory=list)

    pathways: List[List[int]] = dataclasses.field(default_factory=list)

    # thermodynamics
    temperature: float = 673.0  # K

    pressure: float = 1.0  # bar

    def __post_init__(self):
        """"""
        if self.trajectory:
            self.structures = read(
                self.trajectory["filename"], self.trajectory["index"]
            )
            self._convert_structures()
        else:
            self.free_energies = copy.deepcopy(self.energies)

        if self.reverse:
            self.structures = self.structures[::-1]
            self.energies = self.energies[::-1]
            self.free_energies = self.free_energies[::-1]

        self.npoints = len(self.energies)
        if self.npoints == 2:
            self.is_transitions = [False, False]
            self.pathways = [[0, 1]]
        elif self.npoints == 3:
            self.is_transitions = [False, True, False]
            self.pathways = [[0, 1, 2]]
        else:
            ...

        return

    def _convert_structures(self):
        """Take three structures: IS, TS if it has, and FS."""
        structures = self.structures  # TODO: at least 3 structures
        energies = [a.get_potential_energy() for a in structures]
        # print(f"energies: {energies}")
        imax = 1 + np.argsort(energies[1:-1])[-1]
        indices = [0, imax, -1]  # TODO: imax should not be 0 or -1

        self.energies = [energies[i] for i in indices]

        if self.frequencies:
            self.compute_free_energies(self.temperature, self.pressure)
        else:
            self.free_energies = copy.deepcopy(self.energies)

        return

    def compute_free_energies(self, temperature, pressure):
        """"""
        free_energy_corrections = []
        for freq_wdir in self.frequencies:
            vib_energies = read_vibrations(freq_wdir)
            thermo = HarmonicThermo(vib_energies)
            dF = thermo.get_helmholtz_energy(temperature=temperature)
            free_energy_corrections.append(dF)
        free_energy_corrections = np.array(free_energy_corrections)

        free_energies = np.array(self.energies) + free_energy_corrections
        self.free_energies = free_energies.tolist()

        return

    def append(self, other: "ReactionData", shift: bool = False):
        """"""
        # print(self.names, other.names)
        assert (
            self.names[-1] == other.names[0]
        ), f"Names are inconsistent. {self.names[-1]} vs. {other.names[0]}."

        # - concat
        self.names += other.names[1:]
        if not shift:
            self.energies += other.energies[1:]
            self.free_energies += other.free_energies[1:]
        else:
            end_ene = self.energies[-1]
            self.energies += [e + end_ene for e in other.energies[1:]]

            end_ene = self.free_energies[-1]
            self.free_energies += [e + end_ene for e in other.free_energies[1:]]

        # self.coordinates += other.coordinates[1:]
        self.is_transitions += other.is_transitions[1:]

        prev_npoints = self.npoints
        self.npoints = len(self.energies)

        self.pathways.append(range(prev_npoints - 1, self.npoints))

        # consistent
        num_names, num_energies = len(self.names), len(self.energies)
        assert num_names == num_energies

        return

    def shift(self, shift: float = 0.0):
        """"""
        self.energies = [e + shift for e in self.energies]
        self.free_energies = [e + shift for e in self.free_energies]

        return

    def show(self, ax, start: float = 0.0, color: str = "k", add_text: bool = True):
        """"""
        eps = self.eps
        energies = self.energies
        for i in range(self.npoints):
            pos = i + start
            ax.plot([pos - eps, pos + eps], [energies[i], energies[i]], color=color)

        if add_text:
            for i in range(self.npoints):
                pos = i + start
                ax.text(
                    pos,
                    energies[i],
                    self.names[i],
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        for i in range(self.npoints - 1):
            pos = i + start
            ax.plot(
                [pos + eps, pos + 1 - eps],
                [energies[i], energies[i + 1]],
                color=color,
                linestyle="--",
            )

        return


class EnergyDiagram:

    eps: float = 0.25

    def __init__(self, units: str = "eV") -> None:
        """"""
        self.units = units

        return

    def add(
        self,
        ax,
        names,
        energies,
        pathways,
        label="reaction pathway",
        start=0.0,
        end=None,
        shift: float = 0.0,
        cshift: float = 0,
        color="k",
        add_text: bool = True,
        add_ticks: bool = False,
        smooth_curve: bool = True,
    ):
        """Add a reaction profile to the figure.

        Args:
            cshift: Coordinate shift that does not affect energy results.

        """
        num_points = len(names)

        if end is None:
            end = start - 1.0 + num_points * 1.0

        # - map coordinates
        coordinates = np.linspace(start, end, num=num_points, endpoint=True)
        step = abs(coordinates[0] - coordinates[1])

        eps = step * self.eps

        # --- group elementary steps
        # groups = [
        #    [0, 1, 2],
        #    [2, 3, 4],
        #    [4, 5, 6],
        # ]
        groups = pathways
        mediates, apexes = [], []
        for i, grp in enumerate(groups):
            # ---
            beg, end = grp[0], grp[-1]
            ene_beg, ene_end = energies[beg], energies[end]
            pos_beg, pos_end = coordinates[beg], coordinates[end]
            mediates.extend([beg, end])
            if len(grp) == 2:
                ax.plot(
                    [pos_beg + eps, pos_end - eps],
                    [ene_beg + cshift, ene_end + cshift],
                    color=color,
                    linestyle="-",
                )
            elif len(grp) == 3:
                if smooth_curve:
                    cs = BPoly.from_derivatives(
                        [pos_beg + eps, (pos_beg + pos_end) / 2.0, pos_end - eps],
                        # [[ene_beg, 0.], [energies[grp[1]], 0.], [ene_end, 0.]]
                        [
                            [ene_beg + cshift],
                            [energies[grp[1]] + cshift, 0.0],
                            [ene_end + cshift],
                        ],
                    )
                    x = np.linspace(pos_beg + eps, pos_end - eps, 100)
                    y = cs(x)
                    ax.plot(x, y, color=color, linestyle="-")
                else:
                    pos_mid = (pos_beg + pos_end) / 2.0
                    ene_mid = energies[grp[1]]
                    ax.plot(  # IS -> TS
                        [pos_beg + eps, pos_mid - eps],
                        [ene_beg + cshift, ene_mid + cshift],
                        color=color,
                        linestyle="--",
                    )
                    ax.plot(  # TS
                        [pos_mid - eps, pos_mid + eps],
                        [ene_mid + cshift, ene_mid + cshift],
                        color=color,
                        linestyle="-",
                    )
                    ax.plot(  # TS -> FS
                        [pos_mid + eps, pos_end - eps],
                        [ene_mid + cshift, ene_end + cshift],
                        color=color,
                        linestyle="--",
                    )

                # imax = 1 + np.argsort(y[1:-1])[-1]
                # apexes.append([beg+1, x[imax], y[imax]])
                apexes.append(grp[1])

        energies = np.array(energies)
        ene_min, ene_max = np.min(energies+shift), np.max(energies+shift)
        ylow = (ene_min//0.5)*0.5
        if -1e8 <= ylow < 1e-8:
            ylow -= 0.5
        yhigh = (ene_max//0.5+1)*0.5
        if (yhigh-ene_max-0.5) < 0:
            yhigh += 0.5
        ylimit = yhigh-ylow
        ax.set_ylim([ylow, yhigh])

        mediates = set(mediates)
        lines = []
        for i in mediates:
            pos = coordinates[i]
            ene = energies[i] + shift
            l = ax.plot(
                [pos - eps, pos + eps], [ene + cshift, ene + cshift], color=color
            )
            lines.append(l)
            ax.text(
                pos,
                (ene + cshift - ylow)/ylimit-0.08,
                f"{ene:.2f}",
                transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        ax.scatter(
            [x + start for x in apexes],
            [energies[i] + cshift for i in apexes],
            color=color,
            facecolor="w",
            zorder=100,
        )
        for i in apexes:
            pos = coordinates[i]
            ene = energies[i] + shift
            ax.text(
                pos,
                (ene + cshift - ylow)/ylimit+0.02,
                f"{ene:.2f}",
                transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                color="r",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        if add_text:
            for i in range(num_points):
                pos = coordinates[i]
                ax.text(
                    pos,
                    energies[i] + shift + cshift - 0.02,
                    names[i],
                    horizontalalignment="center",
                    verticalalignment="top",
                    # fontsize=12
                )

        if add_ticks:
            maxlen = max([len(n) for n in names])
            if maxlen <= 6:
                ax.set_xticks(coordinates, names, rotation=15, fontsize=24)
            else:
                ax.set_xticks(coordinates, names, rotation=15, fontsize=18)

        # custom_lines = [matplotlib.lines.Line2D([0], [0], color=color, lw=4)]
        # ax.legend(custom_lines, [label])

        return


if __name__ == "__main__":
    ...
