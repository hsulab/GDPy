import json
from types import SimpleNamespace

import pytest
import yaml

from gdpx.providers.deepmd.training.deepmd import DeepmdTrainer
from gdpx.providers.training import BasePotentialTrainer, FreezingFailed
from gdpx.workflow.factory import create_trainer


def _dpa4_config(compact=False, family="dpa4"):
    model = {"type_map": ["H", "O"]}
    if family == "dpa4":
        model["type"] = "dpa4"
    if not compact:
        model.update(descriptor={"type": family}, fitting_net={"neuron": [64, 64]})
    return {
        "model": model,
        "learning_rate": {"type": "exp", "start_lr": 0.001},
        "loss": {"type": "ener"},
        "training": {
            "training_data": {"systems": [], "batch_size": 1},
            "validation_data": {"systems": [], "batch_size": 1},
        },
    }


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
@pytest.mark.parametrize("type_list", [None, ["H", "O"]])
def test_workflow_trainer_loads_config_file(tmp_path, suffix, type_list):
    config = _dpa4_config()
    path = tmp_path / f"input{suffix}"
    path.write_text(json.dumps(config) if suffix == ".json" else yaml.safe_dump(config))

    trainer = create_trainer({
        "provider": "deepmd", "method": "default",
        "parameters": {"config": str(path), "type_list": type_list},
    })

    assert trainer.config == config
    assert trainer.type_list == ["H", "O"]
    assert trainer.component_config.to_dict()["parameters"]["config"] == config
    assert trainer._resolve_train_command().startswith("dp --pt train deepmd.json")


def test_trainer_loads_path_object(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_dpa4_config()))

    trainer = DeepmdTrainer(config=path)

    assert trainer.config == _dpa4_config()


def test_trainer_rejects_config_file_without_mapping(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('"invalid"')

    with pytest.raises(TypeError, match="DeepMD configuration must be a mapping"):
        DeepmdTrainer(config=path, type_list=["H", "O"])


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


def test_dpa4c_freezes_graph_and_compresses_pt2(monkeypatch, tmp_path):
    trainer = DeepmdTrainer(config=_dpa4_config(family="dpa4c"), directory=tmp_path)
    frozen = tmp_path / trainer.frozen_name
    compressed = tmp_path / trainer.compressed_name
    commands = []

    assert trainer.checkpoint_name == "model.ckpt.pt"
    assert trainer.frozen_name == "deepmd.pt2"
    assert trainer.compressed_name == "deepmd-c.pt2"
    assert trainer._resolve_train_command().startswith("dp --pt-expt train deepmd.json")
    assert "--lower-kind graph" in trainer._resolve_freeze_command()
    assert trainer._resolve_compress_command().startswith(
        "dp --pt-expt compress -i deepmd.pt2 -o deepmd-c.pt2"
    )

    frozen.touch()
    monkeypatch.setattr(BasePotentialTrainer, "freeze", lambda self: frozen)

    class _Process:
        def wait(self):
            compressed.touch()
            return 0

    monkeypatch.setattr(
        "gdpx.providers.deepmd.training.deepmd.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or _Process(),
    )

    assert trainer.freeze() == compressed
    assert commands == [trainer._resolve_compress_command()]


def test_dpa4c_compression_failure_is_not_hidden(monkeypatch, tmp_path):
    trainer = DeepmdTrainer(config=_dpa4_config(family="dpa4c"), directory=tmp_path)
    frozen = tmp_path / trainer.frozen_name
    frozen.touch()
    monkeypatch.setattr(BasePotentialTrainer, "freeze", lambda self: frozen)

    class _Process:
        def wait(self):
            return 1

    monkeypatch.setattr(
        "gdpx.providers.deepmd.training.deepmd.subprocess.Popen",
        lambda *args, **kwargs: _Process(),
    )

    with pytest.raises(FreezingFailed, match="failed to compress"):
        trainer.freeze()


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


def test_write_input_replaces_epoch_duration_and_populates_dataset(tmp_path):
    config = _dpa4_config()
    config["training"] = {"training_data": {}, "num_epochs": 10}
    trainer = DeepmdTrainer(
        config=config,
        directory=tmp_path,
        train_epochs=2,
        print_epochs=1,
        train_batches=None,
    )
    dataset = SimpleNamespace(
        train_sys_dirs=[str(tmp_path / "train")],
        valid_sys_dirs=["None"],
        batchsizes=[2],
        cum_batchsizes=2,
    )
    trainer._prepare_dataset = lambda value: value

    trainer.write_input(dataset)

    with open(tmp_path / "deepmd.json") as stream:
        written = json.load(stream)
    assert "num_epochs" not in written["training"]
    assert written["training"]["numb_steps"] == 100
    assert written["training"]["training_data"] == {
        "systems": [str(tmp_path / "train")],
        "batch_size": [2],
    }
    assert "validation_data" not in written["training"]


def test_dpa4_convergence_uses_normalized_output_without_requiring_export(tmp_path):
    trainer = DeepmdTrainer(config=_dpa4_config(), directory=tmp_path)
    (tmp_path / "deepmd.json").write_text(
        json.dumps({"training": {"num_epochs": 10}})
    )
    (tmp_path / "out.json").write_text(
        json.dumps({"training": {"numb_steps": 5000}})
    )
    (tmp_path / "lcurve.out").write_text(
        "# step loss\n4500 2.0e-3\nnot-a-step\n5000 1.0e-3\n\n"
    )

    assert trainer.read_convergence()


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


@pytest.mark.parametrize("family", ["dpa4", "dpa4c", "legacy"])
@pytest.mark.parametrize("options", ["", "--skip-neighbor-stat --log-level DEBUG"])
def test_train_options_apply_to_fresh_and_restart(tmp_path, family, options):
    config = _dpa4_config(family="dpa4c" if family == "dpa4c" else "dpa4")
    if family == "legacy":
        config["model"].pop("type")
        config["model"]["descriptor"]["type"] = "se_e2_a"
    trainer = DeepmdTrainer(config=config, directory=tmp_path, train_options=options)
    (tmp_path / "checkpoint").write_text("saved checkpoint\n")
    fresh = trainer._resolve_train_command()
    restart = trainer._train_from_the_restart(dataset=None, init_model=None)
    assert f"--restart {trainer.checkpoint_name}" in restart
    for command in [fresh, restart]:
        assert command.count("--skip-neighbor-stat") == (1 if options else 0)
        if options:
            assert options in command
    assert "--skip-neighbor-stat" not in trainer._resolve_freeze_command()
