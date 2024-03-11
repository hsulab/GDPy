#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import pathlib

import numpy as np

from reportlab.platypus import SimpleDocTemplate, Image, Table, Paragraph, PageBreak

from ase.io import read, write

from . import registers
from .selector import AbstractSelector


class CompareSelector(AbstractSelector):

    name: str = "compare"

    default_parameters: dict = dict(comparator_name=None, comparator_params={})

    def __init__(self, directory="./", axis=None, *args, **kwargs) -> None:
        """"""
        super().__init__(directory, axis, *args, **kwargs)

        self.comparator = registers.create(
            "comparator",
            self.comparator_name,
            convert_name=True,
            **self.comparator_params,
        )

        return

    def _mark_structures(self, data, *args, **kwargs) -> None:
        """"""
        super()._mark_structures(data, *args, **kwargs)

        # -
        structures = data.get_marked_structures()
        nstructures = len(structures)

        # - start from the first structure and compare structures by a given comparator
        if not hasattr(self.comparator, "prepare_data"):
            selected_indices, scores = [0], []
            for i, a1 in enumerate(structures[1:]):
                # NOTE: assume structures are sorted by energy
                #       close structures may have a high possibility to be similar
                #       so we compare reversely
                for j in selected_indices[::-1]:
                    self._print(f"compare: {i+1} and {j}")
                    a2 = structures[j]
                    if self.comparator(a1, a2):
                        break
                else:
                    selected_indices.append(i + 1)
                    self._print(f"--->>> current indices: {selected_indices}")
        else:
            fingerprints = self.comparator.prepare_data(structures)

            # - compare structural fingerprint
            selected_indices, unique_groups, scores = [], {}, []
            for i, fp in enumerate(fingerprints):
                for j in selected_indices[::-1]:
                    # TODO: The ditribution maybe too wide?
                    fp_avg = np.average(
                        [fingerprints[x] for x in unique_groups[j]], axis=0
                    )
                    if self.comparator(fp, fp_avg):
                        unique_groups[j].append(i)
                        break
                else:
                    selected_indices.append(i)
                    unique_groups[i] = [i]

            # - merge groups with similar average fingerprint
            unique_fingerprints = {}
            for k, v in unique_groups.items():
                unique_fingerprints[k] = np.average(
                    [fingerprints[x] for x in v], axis=0
                )

            selected_group_indices, merged_groups = [], {}
            for i, fp in unique_fingerprints.items():
                for j in selected_group_indices:
                    # TODO: The ditribution maybe too wide?
                    fp_avg = np.average(
                        [unique_fingerprints[x] for x in merged_groups[j]], axis=0
                    )
                    if self.comparator(fp, fp_avg):
                        merged_groups[j].append(i)
                        break
                else:
                    selected_group_indices.append(i)
                    merged_groups[i] = [i]

            new_selected_indices, new_unique_groups = [], {}
            for k, v in merged_groups.items():
                curr_indices = sorted(
                    list(itertools.chain(*[unique_groups[x] for x in v]))
                )
                # print(curr_indices)
                # print(unique_fingerprints[k])
                new_unique_groups[curr_indices[0]] = curr_indices
                new_selected_indices.append(curr_indices[0])
            selected_indices = new_selected_indices
            unique_groups = new_unique_groups

            # - report data
            unique_wdir = self.directory / "unique"
            unique_wdir.mkdir(parents=True, exist_ok=True)

            # -- save structures per group?
            with open(unique_wdir / "unique.dat", "w") as fopen:
                for k, v in unique_groups.items():
                    fopen.write(f"{k}_{len(v)}: {v}\n")
            #        write(unique_wdir/f"g_{k}.xyz", [structures[i] for i in v])

            # -- write a report?
            story = []
            for i, (k, v) in enumerate(unique_groups.items()):
                curr_frames = [structures[x] for x in v]
                curr_nframes = len(curr_frames)
                if curr_nframes > 1:
                    story.append(Paragraph(f"Group {k}"))
                    story.append(Paragraph(" ".join([str(x) for x in v])))
                    images = []
                    for ia, a in enumerate(curr_frames[:]):
                        write(unique_wdir / f"xxx-{i}_{ia}.png", a)
                        image = Image(unique_wdir / f"xxx-{i}_{ia}.png")
                        image.drawWidth = 108
                        image.drawHeight = 100
                        images.append(image)
                    image_table = []
                    nrows = int(curr_nframes / 4.0)
                    for x in range(nrows):
                        image_table.append(
                            [images[im] for im in range(x * 4, (x + 1) * 4)]
                        )
                    if (curr_nframes - nrows * 4) > 0:
                        image_table.append(
                            [images[im] for im in range(nrows * 4, curr_nframes)]
                        )

                    story.append(Table(image_table))
                    story.append(PageBreak())

            doc = SimpleDocTemplate(str(self.directory / "report.pdf"))
            doc.build(story)

            for x in unique_wdir.glob("*.png"):
                os.remove(x)

        curr_markers = data.markers
        # NOTE: convert to np.array as there may have 2D markers
        selected_markers = np.array([curr_markers[i] for i in selected_indices])
        data.markers = selected_markers

        return


if __name__ == "__main__":
    ...
