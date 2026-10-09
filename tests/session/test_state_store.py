import yaml
from ase import Atoms
from ase.io import write

from gdpx.data.loaders.dataset import XyzDataloader, XyzSnapshotDataloader
from gdpx.providers import PotentialConfig
from gdpx.workflow.state_store import decode_state, encode_state


def test_potential_state_preserves_backend_and_references_models_by_path(tmp_path):
    model = tmp_path / "iter.0000" / "0000.train" / "m0" / "model.pb"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-v1")
    potential = PotentialConfig(
        "deepmd",
        "default",
        {"model": [str(model)], "type_list": ["H"]},
        "lammps",
    )

    record = encode_state(potential, tmp_path)
    restored = decode_state(record, tmp_path)

    assert restored.backend == "lammps"
    assert restored.parameters["model"] == (str(model.resolve()),)
    model.write_bytes(b"changed")
    assert decode_state(record, tmp_path).parameters["model"] == (str(model.resolve()),)


def test_initial_xyz_dataset_is_persisted_as_path_snapshot(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    loader = XyzDataloader(dataset, batchsize=4, train_ratio=0.8, random_seed=17)

    record = encode_state(loader, tmp_path)
    restored = decode_state(record, tmp_path)

    assert record["kind"] == "xyz_snapshot"
    assert isinstance(restored, XyzSnapshotDataloader)
    assert restored.sources == (dataset.resolve(),)
    assert restored.batchsize == 4
    assert restored.train_ratio == 0.8
    assert restored.random_seed == 17

    (dataset / "unexpected.txt").write_text("changed", encoding="utf-8")
    assert isinstance(decode_state(record, tmp_path), XyzSnapshotDataloader)


def test_named_potential_state_uses_manifest_without_copying_model(tmp_path):
    model = tmp_path / "iter.0000" / "0000.train" / "model.npz"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"large-model-placeholder")
    potential = PotentialConfig("nnp", None, {"model": [str(model)]}, "ase")

    record = encode_state(potential, tmp_path, state_name="potential", iteration=0)
    restored = decode_state(record, tmp_path)

    manifest = tmp_path / "artifacts" / "models" / "potential" / "iterations" / "0000.yaml"
    assert record["kind"] == "potential_artifact"
    assert manifest.is_file()
    assert restored.parameters["model"] == (str(model.resolve()),)
    assert list((tmp_path / "artifacts" / "models").rglob("*.npz")) == []


def test_named_initial_dataset_manifest_references_source_without_copying(tmp_path):
    source = tmp_path / "input" / "seed-H-bulk"
    source.mkdir(parents=True)
    write(source / "part.xyz", [Atoms("H"), Atoms("H")])

    record = encode_state(
        XyzDataloader(source.parent, train_ratio=1.0),
        tmp_path,
        state_name="training_data",
        iteration=-1,
    )
    restored = decode_state(record, tmp_path)

    copied_shard = (
        tmp_path
        / "artifacts"
        / "datasets"
        / "training_data"
        / "systems"
        / "seed-H-bulk"
        / "initial.xyz"
    )
    assert not copied_shard.exists()
    assert sum(map(len, restored.load_frames().values())) == 2
    assert restored.shards[0]["path"] == (source / "part.xyz").resolve()
    assert record["artifact"] == "artifacts/datasets/training_data/versions/initial.yaml"
    manifest = yaml.safe_load((tmp_path / record["artifact"]).read_text(encoding="utf-8"))
    assert manifest["loader"]["sources"] == ["input"]
    assert manifest["systems"] == {}
