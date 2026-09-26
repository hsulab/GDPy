from gdpx.cli import train as train_cli


class _StructuredTrainer:
    requires_dataloader = True

    def __init__(self):
        self.directory = None
        self.dataset = None
        self.init_model = None
        self.frozen = False

    def train(self, dataset, init_model=None):
        self.dataset = dataset
        self.init_model = init_model

    def freeze(self):
        self.frozen = True


def test_run_trainer_preserves_dataloader_for_structured_trainers(monkeypatch, tmp_path):
    trainer = _StructuredTrainer()
    dataloader = object()
    parameters = {
        "trainer": {"provider": "deepmd", "parameters": {"config": {}}},
        "dataset": {"name": "xyz", "dataset_path": "dataset", "batchsize": 4},
        "init_model": "pretrained.pt",
    }
    monkeypatch.setattr(train_cli, "parse_input_file", lambda configuration: parameters)
    monkeypatch.setattr(train_cli, "create_trainer", lambda configuration: trainer)
    monkeypatch.setattr(
        train_cli,
        "create_dataloader",
        lambda configuration: dataloader,
    )

    train_cli.run_trainer("train.yaml", tmp_path)

    assert trainer.directory == tmp_path
    assert trainer.dataset is dataloader
    assert trainer.init_model == "pretrained.pt"
    assert trainer.frozen
