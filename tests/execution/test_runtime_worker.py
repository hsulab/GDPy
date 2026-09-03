from gdpx.factory.computer import create_workers


def test_schema_v2_creates_a_runtime_backed_worker():
    config = {
        "schema_version": 2,
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {"provider": "ase", "method": "min", "parameters": {"steps": 1}},
        "scheduler": {"provider": "local", "parameters": {}},
        "batchsize": 2,
    }

    worker = create_workers(config)[0]

    assert worker.runtime.config.executor.method == "min"
    assert worker.batchsize == 2
    assert worker.as_dict()["schema_version"] == 2
    assert worker.as_dict()["potential"]["provider"] == "emt"

