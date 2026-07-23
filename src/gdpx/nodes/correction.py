"""Workflow adapter for data correction computations."""

import itertools

from ase.io import write

from gdpx.session.registry import workflow_registers as registers
from gdpx.data.correction import merge_results
from gdpx.session.operation import Operation


@registers.operation.register
class correct(Operation):
    def __init__(self, structures, computer, directory="./") -> None:
        super().__init__([structures, computer], directory)

    def forward(self, structures, computer):
        super().forward()
        worker = computer[0]
        worker._share_wdir = True
        statuses = []
        for name, frames in structures:
            worker.directory = self.directory / name
            worker.batchsize = len(frames)
            worker.run(frames)
            worker.inspect(resubmit=True)
            statuses.append(worker.get_number_of_running_jobs() == 0)
        if not all(statuses):
            return None

        self.status = "finished"
        corrected = []
        for name, frames in structures:
            worker.directory = self.directory / name
            correction = worker.retrieve()
            if not worker._share_wdir:
                correction = itertools.chain.from_iterable(correction)
            merged = merge_results(frames, correction)
            write(self.directory / name / "merged.xyz", merged)
            corrected.append([name, merged])
        return corrected
