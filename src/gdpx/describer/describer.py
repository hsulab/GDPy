#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc

from sklearn.decomposition import PCA

from gdpx.core.component import BaseComponent

try:
    import chemiscope
except Exception as e:
    chemiscope = None


class AbstractDescriber(BaseComponent):

    cache_features = "features.npy"

    def __init__(self, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        return

    @abc.abstractmethod
    def run(self, dataset, *args, **kwargs):
        """"""

        return

    def _write_chemiscope(self, frames, features):
        """"""
        assert chemiscope is not None

        # Write chemiscope inputs
        pca = PCA(n_components=2).fit_transform(features)
        properties = dict(
            PCA=dict(target="structure", values=pca),
            # energies = dict(
            #    target = "structure",
            #    values = [a.get_potential_energy() for a in frames],
            #    units = "eV"
            # )
        )

        frame_properties = chemiscope.extract_properties(frames, only=["energy"])
        properties.update(**frame_properties)

        chemiscope.write_input(
            self.directory / "my-input.json.gz", frames=frames, properties=properties
        )

        return

    def _plot_results(self, features, groups: dict[str, list[int]]):
        """"""
        # - plot selection
        import matplotlib.pyplot as plt

        try:
            plt.style.use("presentation")
        except Exception:
            ...

        reducer = PCA(n_components=2)
        reducer.fit(features)

        for i, (name, indices) in enumerate(groups.items()):
            proj = reducer.transform(features[indices, :])
            plt.scatter(
                proj[:, 0],
                proj[:, 1],
                alpha=0.5,
                zorder=100 - i,
                label=f"{name} {len(indices)}",
            )

        plt.legend()
        plt.axis("off")
        plt.savefig(self.directory / "pca.png")

        return


if __name__ == "__main__":
    ...
