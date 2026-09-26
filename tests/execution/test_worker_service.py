from gdpx.execution import ExecutionHandle, WorkerExecutionService


class Worker:
    def __init__(self):
        self.inputs = None
        self.running = 1

    def run(self, inputs, **kwargs):
        self.inputs = inputs

    def inspect(self, resubmit=False, **kwargs):
        self.running = 0

    def retrieve(self, **kwargs):
        return [self.inputs]

    def get_number_of_running_jobs(self):
        return self.running


def test_worker_execution_service_adapts_existing_lifecycle():
    worker = Worker()
    service = WorkerExecutionService(worker)

    handle = service.submit(None, "atoms")

    assert isinstance(handle, ExecutionHandle)
    assert service.status(handle).state == "finished"
    assert service.retrieve(handle) == ["atoms"]

