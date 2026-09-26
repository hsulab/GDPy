import json
from types import SimpleNamespace

from gdpx.providers.deepmd.training.deepmd import DeepmdTrainer
from gdpx.providers.training import BasePotentialTrainer


def _dpa4_config(compact=False):
    model = {"type": "dpa4", "type_map": ["H", "O"]}
    if not compact:
        model.update(descriptor={"type": "dpa4"}, fitting_net={"neuron": [64, 64]})
    return {
        "model": model,
        "learning_rate": {"type": "exp", "start_lr": 0.001},
        "loss": {"type": "ener"},
        "training": {
            "training_data": {"systems": [], "batch_size": 1},
            "validation_data": {"systems": [], "batch_size": 1},
        },
    }


def test_dpa4_commands_and_artifact(tmp_path):
    trainer = DeepmdTrainer(config=_dpa4_config(), directory=tmp_path)

    assert trainer.model_family == "dpa4"
    assert trainer.checkpoint_name == "model.ckpt.pt"
    assert trainer.frozen_name == "deepmd.pt2"
    assert trainer._resolve_train_command().startswith("dp --pt train deepmd.json")
    assert trainer._resolve_freeze_command().startswith("dp --pt freeze -c model.ckpt.pt -o deepmd")


def test_dpa4_pt_checkpoint_is_used_for_finetuning(tmp_path):
    trainer = DeepmdTrainer(config=_dpa4_config(), directory=tmp_path)
    checkpoint = tmp_path / "pretrained.pt"

    command = trainer._resolve_train_command(init_model=checkpoint)

    assert f"--finetune {checkpoint.resolve()}" in command


def test_dpa4_freeze_skips_legacy_compression(monkeypatch, tmp_path):
    trainer = DeepmdTrainer(config=_dpa4_config(), directory=tmp_path)
    exported = tmp_path / trainer.frozen_name
    monkeypatch.setattr(BasePotentialTrainer, "freeze", lambda self: exported)
    monkeypatch.setattr(
        trainer,
        "_resolve_compress_command",
        lambda: (_ for _ in ()).throw(AssertionError("compression should not run")),
    )

    assert trainer.freeze() == exported


def test_compact_dpa4_config_can_be_written(tmp_path):
    trainer = DeepmdTrainer(
        config=_dpa4_config(compact=True),
        directory=tmp_path,
        train_epochs=2,
        print_epochs=1,
        train_batches=None,
    )
    dataset = SimpleNamespace(
        train_sys_dirs=[str(tmp_path / "train")],
        valid_sys_dirs=[str(tmp_path / "valid")],
        batchsizes=[1],
        cum_batchsizes=2,
    )
    trainer._prepare_dataset = lambda value: value

    trainer.write_input(dataset)

    with open(tmp_path / "deepmd.json") as stream:
        written = json.load(stream)
    assert "descriptor" not in written["model"]
    assert "fitting_net" not in written["model"]
    assert written["training"]["training_data"]["systems"] == [str(tmp_path / "train")]
    assert written["training"]["numb_steps"] == 100


def test_legacy_deepmd_commands_remain_unchanged(tmp_path):
    config = _dpa4_config()
    config["model"].pop("type")
    config["model"]["descriptor"]["type"] = "se_e2_a"
    trainer = DeepmdTrainer(config=config, directory=tmp_path)

    assert trainer.model_family is None
    assert trainer.checkpoint_name == "model.ckpt"
    assert trainer.frozen_name == "deepmd.pb"
    assert trainer._resolve_train_command().startswith("dp train deepmd.json")
    assert trainer._resolve_freeze_command().startswith("dp freeze -o deepmd.pb")
