from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.data.loaders.dataset import XyzDataloader, XyzSnapshotDataloader
from gdpx.workflow.nodes.dataset import transfer
from gdpx.workflow.session.variable import Variable


def _labelled_hydrogen(energy):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=[[0.0, 0.0, 0.0]])
    return atoms


def test_transfer_returns_immutable_dataset_snapshots(tmp_path):
    train_root = tmp_path / "inputs" / "train"
    test_root = tmp_path / "inputs" / "test"
    train_root.mkdir(parents=True)
    test_root.mkdir(parents=True)
    train = XyzDataloader(train_root, train_ratio=1.0)
    test = XyzDataloader(test_root, train_ratio=1.0)
    operation = transfer(
        Variable([_labelled_hydrogen(0.0), _labelled_hydrogen(1.0)]),
        version="deepmd",
        dataset=Variable(train),
        dataset_test=Variable(test),
        split_ratio={"dataset": 0.5, "dataset_test": 0.5},
        prefix="active",
        directory=tmp_path / "iteration" / "steps" / "transfer",
    )

    outputs = operation.forward([_labelled_hydrogen(0.0), _labelled_hydrogen(1.0)], train, test)

    assert set(outputs) == {"dataset", "dataset_test"}
    assert all(isinstance(value, XyzSnapshotDataloader) for value in outputs.values())
    assert not list(train_root.rglob("*.xyz"))
    assert not list(test_root.rglob("*.xyz"))
    assert sum(map(len, outputs["dataset"].load_frames().values())) == 1
    assert sum(map(len, outputs["dataset_test"].load_frames().values())) == 1
