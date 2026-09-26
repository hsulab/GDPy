import pytest

from gdpx.data.loaders.dataset import XyzDataloader, XyzSnapshotDataloader
from gdpx.providers import PotentialConfig
from gdpx.workflow.state_store import decode_state, encode_state


def test_potential_state_preserves_backend_and_verifies_models(tmp_path):
    model = tmp_path / "iter.0000" / "steps" / "train" / "m0" / "model.pb"
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
    with pytest.raises(RuntimeError, match="changed after commit"):
        decode_state(record, tmp_path)


def test_initial_xyz_dataset_is_persisted_as_immutable_snapshot(tmp_path):
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
    with pytest.raises(RuntimeError, match="changed after commit"):
        decode_state(record, tmp_path)
