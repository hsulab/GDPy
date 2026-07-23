from ase.io import read

from gdpx import config
from gdpx.factory.components import create_trainer
from gdpx.factory.dataloader import create_dataloader
from gdpx.utils.parser import parse_input_file


def run_trainer(configuration, directory) -> None:
    """"""
    config._print(f"{configuration = }")
    params = parse_input_file(configuration)

    # Instantiate the trainer
    trainer = create_trainer(params["trainer"])
    trainer.directory = directory

    # Process the dataset
    dataset_params = params["dataset"]
    name = dataset_params.pop("name", None)

    if name == "single_xyz" or "dataset_path" in dataset_params:
        from ase.io import read
        dataset_path = dataset_params.get("dataset_path")
        if dataset_path:
            dataset = read(dataset_path, ":")
        else:
            dataloader = create_dataloader(dict(name=name, **dataset_params))
            systems = dataloader.load_frames()
            dataset = [frame for _, frames in systems for frame in frames]
    else:
        dataloader = create_dataloader(dict(name=name, **dataset_params))
        if hasattr(dataloader, "load_frames"):
            systems = dataloader.load_frames()
            dataset = [frame for _, frames in systems for frame in frames]
        else:
            dataset = dataloader

    # Other options
    init_model = params.get("init_model", None)

    # Run the trainer
    trainer.train(dataset, init_model=init_model)

    trainer.freeze()

    return
