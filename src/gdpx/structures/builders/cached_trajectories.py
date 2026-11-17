from gdpx.data.trajectories import read_cached_trajectories
from .builder import StructureBuilder


class CachedTrajectoriesBuilder(StructureBuilder):
    """Expose cached compute/exploration trajectories to ``gdp select -s``."""
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.source_directory = str(directory)

    def run(self, *args, **kwargs):
        return [a for trajectory in read_cached_trajectories(self.source_directory) for a in trajectory]

    def as_dict(self):
        return dict(method="cached_trajectories", directory=self.source_directory)
