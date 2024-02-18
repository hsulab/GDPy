#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import pathlib

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule, ASEAtomsData, load_dataset, AtomsDataFormat

from . import AbstractPotentialManager

def convert_frames(frames):
    """"""
    atoms_list, property_list = [], []
    for atoms in frames:
        propertis = {"energy": np.array([atoms.get_potential_energy()]), "forces": atoms.get_forces()}
        property_list.append(propertis)
        # set nopbc
        atoms.pbc = False
    atoms_list = frames
    #print([a.get_pbc() for a in frames])
    #assert np.all([a.get_pbc() for a in frames] == False)

    return property_list, atoms_list

def create_dataset():
    """"""
    data_paths = [
      "/users/40247882/projects/dataset/cof/H2-H2-molecule",
      "/users/40247882/projects/dataset/cof/N2-N2-molecule",
      "/users/40247882/projects/dataset/cof/O2-O2-molecule",
      "/users/40247882/projects/dataset/cof/CO-CO-molecule",
      "/users/40247882/projects/dataset/cof/H2O-H2O-molecule",
      "/users/40247882/projects/dataset/cof/H2S-H2S-molecule",
      "/users/40247882/projects/dataset/cof/CO2-CO2-molecule",
      "/users/40247882/projects/dataset/cof/NH3-H3N-molecule",
      "/users/40247882/projects/dataset/cof/CH4-CH4-molecule",
      "/users/40247882/projects/dataset/cof/methanol-CH4O-molecule",
      "/users/40247882/projects/dataset/cof/MeSCN-C2H3NS-molecule",
      "/users/40247882/projects/dataset/cof/MI-C4H6N2-molecule",
      "/users/40247882/projects/dataset/cof/benzene-C6H6-molecule",
      "/users/40247882/projects/dataset/cof/phenol-C6H6O-molecule",
      "/users/40247882/projects/dataset/cof/quinol-C6H6O2-molecule",
      "/users/40247882/projects/dataset/cof/imine-C7H7N-molecule",
      "/users/40247882/projects/dataset/cof/imine135-C9H9N3-molecule",
      "/users/40247882/projects/dataset/cof/cof5-C90H66N18O12-molecule",
      "/users/40247882/projects/dataset/cof/cof5+mimescn-C96H75N21O12S-molecule",
      "/users/40247882/projects/dataset/cof/mimescn-C6H9N3S-molecule",
    ]
    data_paths = [pathlib.Path(p) for p in data_paths]

    new_dataset = ASEAtomsData.create(
        "./dp_dataset.db",
        distance_unit="Ang",
        property_unit_dict={"energy": "eV", "forces": "eV/Ang"}
    )

    for p in data_paths:
        print(p.name)
        xyz_files = p.glob("*.xyz")
        frames = []
        for x in xyz_files:
            frames.extend(read(x, ":"))
        property_list, atoms_list = convert_frames(frames)
        new_dataset.add_systems(property_list, atoms_list)

    print('Number of reference calculations:', len(new_dataset))
    print('Available properties:')
    for p in new_dataset.available_properties:
        print('-', p)
    print()    

    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape)
    
    return

def train():
    """"""
    import torch
    import torchmetrics
    import pytorch_lightning as pl

    # - create data module
    new_dataset = AtomsDataModule(
        datapath = "./new_dataset.db",
        distance_unit="Ang",
        property_units={"energy": "eV", "forces": "eV/Ang"},
        batch_size=32,
        num_train=224,
        num_val=32,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(
                "energy", remove_mean=True, remove_atomrefs=False
            ),
            trn.CastTo32()
        ],
        num_workers=1,
        pin_memory=False
    )
    new_dataset.prepare_data()
    new_dataset.setup()

    # - build model
    cutoff = 5
    n_atom_basis = 30

    # - schnet
    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    #schnet = spk.representation.SchNet(
    #    n_atom_basis=n_atom_basis, n_interactions=3,
    #    radial_basis=radial_basis,
    #    cutoff_fn=spk.nn.CosineCutoff(cutoff)
    #)
    schnet = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    # - components
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy")
    pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
            ]
    )

    output_energy = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    forcetut = './forcetut'
    if not os.path.exists(forcetut):
        os.makedirs(forcetut)

    logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(forcetut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=forcetut,
        max_epochs=5, # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=new_dataset)

    return

def use_model():
    """"""
    calculator = spk.interfaces.SpkCalculator(
        model_file="/mnt/scratch2/users/40247882/porous/schtrain/r0/_ensemble/m1/runs/ca9e9f60-fb47-11ed-b995-b07b25d4c022/best_model",
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        energy_key="energy",
        force_key="forces",
        stress_key="stress",
        energy_unit="eV",
        position_unit="Ang",
    )

    frames = read("/users/40247882/projects/dataset/cof/methanol-CH4O-molecule/init.xyz", ":")
    energies = np.array([a.get_potential_energy() for a in frames])
    print(energies)

    atoms = frames[0]
    atoms.calc = calculator

    #_ = atoms.get_potential_energy()
    _ = atoms.get_forces()
    #print("energy: ", atoms.get_potential_energy())
    #print("forces: ", atoms.get_forces())
    #print("stress: ", atoms.get_stress())

    print("results: ", atoms.calc.results)

    return


class SchnetManager(AbstractPotentialManager):

    name = "schnet"

    implemented_backends = ["ase"]

    valid_combinations = (
        ("ase", "ase")
    )

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)
        self.calc = self._create_calculator(self.calc_params)

        return

    def _create_calculator(self, calc_params: dict) -> Calculator:
        """"""
        calc_params = copy.deepcopy(calc_params)

        # - some shared params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i
        
        # --- model files
        model_ = calc_params.get("model", [])
        if not isinstance(model_, list):
            model_ = [model_]

        models = []
        for m in model_:
            m = pathlib.Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))

        # - create specific calculator
        if self.calc_backend == "ase":
            # return ase calculator
            calc = spk.interfaces.SpkCalculator(
                model_file=models[0],
                neighbor_list=trn.ASENeighborList(cutoff=5.0),
                energy_key="energy",
                force_key="forces",
                stress_key="stress",
                energy_unit="eV",
                position_unit="Ang",
            )
        elif self.calc_backend == "lammps":
            raise NotImplementedError(f"SchNet has not supported LAMMPS.")

        return calc


if __name__ == "__main__":
    #create_dataset()
    #train()
    use_model()
    ...