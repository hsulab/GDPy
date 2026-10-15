import io

import matplotlib.pyplot as plt
import numpy as np
import PIL
from ase.io import write

try:
    USE_REPORTLAB = 1
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table
except:
    USE_REPORTLAB = 0

from gdpx.analysis.comparators import create_comparator
from gdpx.data.array import AtomsNDArray

from .selector import BaseSelector
from .similarity import group_structures_by_similarity, validate_similarity_comparator


class CompareSelector(BaseSelector):
    name: str = "compare"

    def __init__(
        self,
        comparator=None,
        number=None,
        fmax=None,
        rank_by=None,
        write_report: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialise the selector.

        Args:
            comparator: The comparator configuration.
            number: Maximum number of distinct structures to retain.
            fmax: Maximum atomic force-vector norm for eligible structures.
            rank_by: Candidate ranking; ``energy`` or ``None``.
            write_report: Whether to write a report. Defaults to False.

        """
        obsolete = {"n_top", "ofp", "rmsd"} & kwargs.keys()
        if obsolete:
            raise ValueError(
                f"Removed compare parameters: {sorted(obsolete)}. Use comparator "
                "with method and group_expression; use rmsd_max for displacement."
            )
        super().__init__(*args, **kwargs)

        if self.group_by is not None:
            raise ValueError("Grouping is not supported in comparison.")
        if number is not None and (
            isinstance(number, bool) or not isinstance(number, int) or number <= 0
        ):
            raise ValueError("number must be a positive integer or null.")
        if fmax is not None and (not np.isfinite(fmax) or fmax <= 0):
            raise ValueError("fmax must be positive and finite or null.")
        if rank_by not in (None, "energy"):
            raise ValueError("rank_by must be energy or null.")
        if comparator is None:
            raise ValueError("Comparison requires an explicit comparator configuration.")

        self.comparator = create_comparator(comparator)
        validate_similarity_comparator(self.comparator)
        self.number = number
        self.fmax = fmax
        self.rank_by = rank_by
        self.write_report = write_report

    def _mark_structures(self, data: AtomsNDArray) -> None:
        """"""
        structures = data.get_marked_structures()
        candidate_indices = []
        energies = {}
        for index, atoms in enumerate(structures):
            if self.rank_by == "energy":
                energy = atoms.get_potential_energy()
                if not np.isfinite(energy):
                    raise ValueError("Energy-ranked comparison requires finite energies.")
                energies[index] = energy
            if self.fmax is not None:
                forces = atoms.get_forces(apply_constraint=True)
                if not np.all(np.isfinite(forces)):
                    raise ValueError("Force-filtered comparison requires finite forces.")
                if np.max(np.linalg.norm(forces, axis=1)) > self.fmax:
                    continue
            candidate_indices.append(index)

        if self.rank_by == "energy":
            candidate_indices.sort(key=lambda index: (energies[index], index))
        candidates = [structures[index] for index in candidate_indices]

        selected_indices, unique_groups = group_structures_by_similarity(
            candidates, self.comparator, max_groups=self.number
        )

        if self.number is not None and len(selected_indices) < self.number:
            raise ValueError(
                f"Requested {self.number} distinct structures, found only "
                f"{len(selected_indices)} among {len(candidates)} eligible candidates."
            )

        if self.write_report:
            if USE_REPORTLAB:
                self.report(candidates, unique_groups)
            else:
                self._print("Please install `reportlab` to report comparison.")

        curr_markers = np.argwhere(data.markers)
        selected_markers = np.array(
            [curr_markers[candidate_indices[i]] for i in selected_indices]
        )
        data.markers = selected_markers

        return

    def report(self, structures, unique_groups):
        """"""
        # - report data
        unique_wdir = self.directory / "unique"
        unique_wdir.mkdir(parents=True, exist_ok=True)

        # -- save structures per group?
        with open(unique_wdir / "unique.dat", "w") as fopen:
            for k, v in unique_groups.items():
                fopen.write(f"{k}_{len(v)}: {v}\n")
        #        write(unique_wdir/f"g_{k}.xyz", [structures[i] for i in v])

        # -- write a report?
        min_num_structures_to_show = 0
        num_structures_per_row = 4
        story = []
        for i, (k, v) in enumerate(unique_groups.items()):
            curr_frames = [structures[x] for x in v]
            curr_nframes = len(curr_frames)
            if curr_nframes > min_num_structures_to_show:
                story.append(Paragraph(f"Group {k} num_structures {len(v)}"))
                images = []
                for ia, a in enumerate(curr_frames[:]):
                    image = self._convert_structure_to_image(
                        a,
                        f"{str(v[ia]).zfill(4)}",
                    )
                    images.append(image)
                image_table = []
                nrows = int(curr_nframes / num_structures_per_row)
                for x in range(nrows):
                    image_table.append(
                        [
                            images[im]
                            for im in range(
                                x * num_structures_per_row,
                                (x + 1) * num_structures_per_row,
                            )
                        ]
                    )
                if (curr_nframes - nrows * num_structures_per_row) > 0:
                    image_table.append([images[im] for im in range(nrows * 4, curr_nframes)])

                story.append(Table(image_table))
                story.append(PageBreak())

        doc = SimpleDocTemplate(str(self.directory / "report.pdf"))
        doc.build(story)

        return

    def _convert_structure_to_image(self, atoms, text):
        """Convert ase.Atoms to reportlab.image"""
        buffer = io.BytesIO()
        write(buffer, atoms, format="png")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        img = PIL.Image.open(buffer)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.05,
            1.05,
            text,
            style="italic",
            bbox={"facecolor": "gray", "alpha": 0.5, "pad": 10},
        )
        buffer2 = io.BytesIO()
        fig.savefig(buffer2, transparent=True, bbox_inches="tight")
        plt.close()
        # ---
        image = Image(buffer2)
        image.drawWidth = 108
        image.drawHeight = 100

        return image
