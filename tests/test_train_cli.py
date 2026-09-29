import yaml

from gdpx.cli import train as train_cli
from gdpx.execution.schedulers.factory import canonicalise_scheduler
from gdpx.execution.workers.train import TrainerBasedWorker
from gdpx.utils.parser import parse_input_file
from gdpx.workflow.factory import create_trainer
from gdpx.workflow.nodes.trainer import TrainerVariable


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


class _Dataset:
    def as_dict(self):
        return {
            "name": "xyz",
            "dataset_path": "dataset",
            "train_ratio": 1.0,
            "batchsize": "n_atoms:1024",
        }


def _dpa4_config():
    return {
        "model": {
            "type": "dpa4",
            "type_map": ["H", "O"],
            "descriptor": {"type": "dpa4"},
            "fitting_net": {"neuron": [64, 64]},
        },
        "learning_rate": {"type": "exp", "start_lr": 0.001},
        "loss": {"type": "ener"},
        "training": {
            "training_data": {"systems": [], "batch_size": 1},
            "validation_data": {"systems": [], "batch_size": 1},
        },
    }


def test_workflow_worker_writes_cli_compatible_dpa4_trainer(monkeypatch, tmp_path):
    trainer = TrainerVariable(
        provider="deepmd",
        method="default",
        parameters={
            "config": _dpa4_config(),
            "type_list": ["H", "O"],
            "train_epochs": 10,
            "train_batches": None,
            "print_epochs": 1,
        },
        directory=tmp_path / "trainer-resource",
    )
    worker = TrainerBasedWorker(
        trainer.value,
        canonicalise_scheduler(
            {"provider": "direct", "parameters": {"is_dry_run": True}}
        ),
        auto_submit=False,
        directory=tmp_path,
    )
    monkeypatch.setattr(
        "gdpx.execution.workers.train.np.random.randint",
        lambda *args, **kwargs: 3101,
    )

    worker.run(_Dataset(), size=1, init_models=[None])

    generated = tmp_path / "m0" / "trainer.yaml"
    params = parse_input_file(generated)
    cli_trainer = create_trainer(params["trainer"])

    assert params["trainer"]["provider"] == "deepmd"
    assert params["trainer"]["method"] == "default"
    assert "parameters" in params["trainer"]
    assert "name" not in params["trainer"]
    assert params["trainer"]["parameters"]["random_seed"] == 3101
    assert params["trainer"]["parameters"]["command"] == "dp"
    assert params["trainer"]["parameters"]["freeze_command"] == "dp"
    assert params["init_model"] is None

    command = cli_trainer._resolve_train_command(params["init_model"])
    assert "dp --pt train deepmd.json" in command
    assert "--finetune" not in command
    assert "--use-pretrain-script" not in command
    assert "--restart" not in command

    # Keep this assertion close to the CLI parsing path: the generated file is
    # ordinary portable YAML, not a Python-specific serialization.
    assert yaml.safe_load(generated.read_text()) == params
