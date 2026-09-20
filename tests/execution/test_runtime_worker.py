from gdpx.execution.factory import create_worker


def test_schema_v3_creates_a_runtime_backed_worker():
    config = {
        "schema_version": 3,
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {"provider": "ase", "method": "min", "parameters": {"steps": 1}},
        "scheduler": {
            "provider": "direct",
            "parameters": {},
            "transport": {"provider": "local", "parameters": {}},
        },
        "options": {"batch_size": 2},
    }

    worker = create_worker(config)

    assert worker.runtime.config.executor.method == "min"
    assert worker.batchsize == 2
    assert worker.as_dict()["schema_version"] == 3
    assert worker.as_dict()["potential"]["provider"] == "emt"
