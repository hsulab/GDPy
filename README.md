<p align="center">
  <img src="./assets/logo.png" width="400" height="300">
</p>

# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [License](#license)

# Overview
Documentation: https://gdpyx.readthedocs.io  

Generating Deep Learning Potential with Python (GDPy/GDP¥) is a bundle of codes for machine learning interatomic potential (MLIP) developments, especially for its application in heterogeneous catalysis.

It mainly includes methods for structure exploration and a unified interface to various MLIPs. The target system can be metal oxides, supported clusters, and solid-liquid interfaces.

# Features
- A unified interface to various MLIPs.
- Versatile exploration algorithms to construct a general dataset.
- Automation workflows for dataset construction and MLIP training.

# Modules
## Driver/Worker
Here, a <strong>`driver`</strong> is defined as a unit (<strong>`AbstractDriver`</strong>) to perform basic dynamics tasks, namely minimisation, molecular dynamics, and transition-state search. It has a ASE <strong>`calculator`</strong> to carry out actual calculation. Through a driver, it allows us to use the same input file to perform a task but with rather different backends (ASE, LAMMPS, LASP, VASP ...). Furthermore, if attached a <strong>`scheduler`</strong>, a <strong>`driver`</strong> becomes a <strong>`worker`</strong> that automatically submit jobs and retrieve results locally or by high-performance clusters (HPCs).

## Scheduler
Currenly, only SLURM is supported.

## Potential
We have supported a few MLIP formulations through <strong>`AbstractPotential`</strong> to use <strong>`driver`</strong>, <strong>`expedition`</strong>, and <strong>`training`</strong> in workflows.

| MLIPs                                                     | Representation                              | Regressor    | Implemented Backend    |
| --------------------------------------------------------- | ------------------------------------------- | ------------ | ---------------------- |
| [eann](https://github.com/zhangylch/EANN)                 | (Rescursive) Embedded Atom Descriptor       | NN/PyTorch   | ASE/Python, ASE/LAMMPS |
| [lasp](http://www.lasphub.com/#/lasp/laspHome)            | Atom-Centered Symmetry Functions            | NN/LASP      | ASE/LASP               |
| [deepmd](https://github.com/deepmodeling/deepmd-kit)      | Deep Potential Descriptors                  | NN/Tensorflow| ASE/Python, ASE/LAMMPS |

*NOTE: we use a modified eann package to train and utilise.*

We have supported a few *ab-initio* package to label explored configurations at density-functional-theory(DFT)-accuracy.

| Ab-Initio   | Description    |
| ----------- | -------------- |
| VASP        | Plane-Wave DFT |

## Expedition
We take advantage of codes in well-established packages (ASE and LAMMPS) to perform basic minimisation and dynamics. Meanwhile, we have implemented several complicated alogirthms in GDPy itself.
| Name                                       | Description                             | Backend     |
| ------------------------------------------ | --------------------------------------- | ----------- |
| Molecular Dynamics (MD)                    | Brute-Force/Biased Dynamics             | ASE, LAMMPS |
| Genetic Algorithm (GA)                     | Evolutionary Global Optimisation        | ASE/GDPy    |
| Grand Cononical Monte Carlo (GCMC)         | Monte Carlo with Variable Composition   | GDPy        |
| Adsorbate Configuration Graph Search (Ads) | Adsorption Site on Graph                | GDPy        |
| Artificial Force Induced Reaction (AFIR)   | Reaction Exploration with Biased Forces | GDPy        | 

## Workflow
There are two kinds of workflows according to the way how they couple the expedition and the training. Offline workflow as the major category separates the expedition and the training, which collects a large number of structures from several expeditions and then train the MLIP with the collective dataset. This process is highly parallelised, and is usually aimed for a general dataset. Online workflow, a really popular one, adopts an on-the-fly strategy to build dataset during the expedition, where a new MLIP is trained to continue exploration once new candidates are selected (sometimes only one structure every time!). Thus, it is mostly used to train a MLIP for a very specific system.

| Type    | Supported Expedition |
| ------- | -------------------- |
| Offline | MD, GA, Ads, AFIR    |
| Online  | MD                   |

# License
GDPy project is under the GPL-3.0 license.
