from gdpx.execution.factory import create_worker
from gdpx.execution.lifecycle import create_runtime_workers


def test_schema_v4_creates_a_runtime_backed_worker():
    config = {
        "schema_version": 4,
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {"provider": "ase", "method": "min", "parameters": {"steps": 1}},
        "scheduler": {
            "provider": "direct",
            "parameters": {},
            "transport": {"provider": "local", "parameters": {}},
        },
        "dispatch": {"batch_size": 2},
    }

    worker = create_worker(config)

    assert worker.runtime.config.executor.method == "min"
    assert worker.batchsize == 2
    assert worker.as_dict()["schema_version"] == 4
    assert worker.as_dict()["potential"]["provider"] == "emt"

    assert worker.as_dict()["potential"]["backend"] == "ase"


def test_runtime_service_expands_executor_broadcast():
    config = {
        "potential": {"provider": "emt"},
        "executor": {
            "provider": "ase",
            "method": "min",
            "parameters": {},
            "broadcast": {"steps": [1, 2]},
        },
    }

    workers = create_runtime_workers(config)

    assert [worker.runtime.config.executor.parameters["steps"] for worker in workers] == [1, 2]
