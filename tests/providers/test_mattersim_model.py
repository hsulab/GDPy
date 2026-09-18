from pathlib import Path

import pytest
import yaml

from gdpx.providers.mattersim.manager import canonicalise_mattersim_model


@pytest.mark.parametrize(
    "model",
    [
        "MatterSim-v1.0.0-1M",
        "MatterSim-v1.0.0-1M.pth",
        "MatterSim-v1.0.0-5M",
        "MatterSim-v1.0.0-5M.pth",
    ],
)
def test_pretrained_model_names_are_preserved(model):
    assert canonicalise_mattersim_model(model) == model


def test_local_model_path_is_resolved(tmp_path):
    model = tmp_path / "model.pth"
    model.touch()

    assert canonicalise_mattersim_model(str(model)) == str(model.resolve())


@pytest.mark.parametrize("model", ["", None])
def test_model_must_be_a_nonempty_string(model):
    with pytest.raises(ValueError, match="non-empty"):
        canonicalise_mattersim_model(model)


def test_missing_local_model_is_rejected(tmp_path):
    model = Path(tmp_path, "missing.pth")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        canonicalise_mattersim_model(str(model))


def test_water_cluster_example_uses_the_fastest_pretrained_model():
    path = (
        Path(__file__).parents[2]
        / "examples"
        / "global_optimisation"
        / "water4_mattersim.yaml"
    )
    with path.open() as stream:
        config = yaml.safe_load(stream)

    builder = config["recipe"]["population"]["builders"]["random"]
    assert builder["composition"] == {"H2O": 4}
    assert builder["pbc"] is True
    assert config["recipe"]["operators"]["crossover"]["method"] == (
        "periodic_cut_and_splice"
    )
    assert config["recipe"]["operators"]["mutation"]["method"] == "rattle"
    assert config["runtime"]["potential"] == {
        "provider": "mattersim",
        "parameters": {
            "model": "MatterSim-v1.0.0-1M",
            "compute_stress": False,
        },
    }
