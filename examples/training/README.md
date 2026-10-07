# DeepMD training examples

These examples train `se_e2_a`, DPA4, and DPA4C on the same small, reproducible
Cu3Au1 dataset. They demonstrate dataset conversion, training, and model export;
100 steps on EMT labels do not produce a validated materials potential.

## Prepare the dataset

Use the existing `gdp3` mamba environment, with GDPy, ASE, dpdata, and DeepMD-kit
3.2.0 installed. The `se_e2_a` example uses the default TensorFlow backend,
DPA4 uses PyTorch, and
DPA4C uses the exportable PyTorch backend. Model export also needs the compiler
toolchain required by the installed DeepMD/PyTorch packages.

From the repository root:

```shell
mamba run -n gdp3 python examples/training/prepare.py
```

The script writes 24 periodic Cu3Au1 structures to
`examples/training/dataset/init-Cu3Au1-bulk/data.xyz`. Each structure has
a slightly strained cell and displaced atoms, with energy, forces, stress,
and virial labels from ASE's EMT calculator. The fixed seed is 1112; rerunning
the script replaces this generated dataset with the same structures.
The source XYZ file lives directly in its system folder; no extra set folder
is needed. During training, DeepMD-format conversion creates `set.000`
directories containing NumPy arrays in the output `train/` and `valid/` systems.

## Train and export

Limit CPU thread counts and disable CUDA for these short examples:

```shell
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

(cd examples/training/se_e2_a && mamba run -n gdp3 gdp -d _train train train.yaml)
(cd examples/training/dpa4 && mamba run -n gdp3 gdp -d _train train train.yaml)
(cd examples/training/dpa4c && mamba run -n gdp3 gdp -d _train train train.yaml)
```

Run from each model directory because configuration and dataset paths are
relative to the working directory. Use a fresh output directory for a fresh
training run. Generated datasets and the `_train` directories are ignored by
Git.

| Example | Training backend | Deployment artifact in `_train/` |
| --- | --- | --- |
| `se_e2_a` | `dp` (default TensorFlow) | `deepmd-c.pb` |
| DPA4 | `dp --pt` | `deepmd.pt2` |
| DPA4C | `dp --pt-expt` | `deepmd-c.pt2` |

GDPy also retains checkpoints, converted `train/` and `valid/` datasets,
`deepmd.json`, and `lcurve.out` in each output directory. DPA4 exports directly;
DPA4C exports a graph and compresses it. The `.pt2` artifacts target the CPU
used for export; export on the device type intended for inference.
Training compilation is disabled, but `.pt2` export still compiles an
AOTInductor package and can take several minutes on the first run.

All YAML files use batch size 4, train ratio 0.9, `train_epochs: 10`, and
`print_epochs: 1`. GDPy splits this dataset into 20 training and four validation
frames and rounds the training duration to **100 steps**. `train_batches: null`
disables the trainer's default 200,000-step minimum. The 100-step rounding is
part of GDPy's checkpoint/display scheduling, so the tiny run lasts longer
than ten exact passes over the training data.

Check one held-out frame from each exported artifact with:

```shell
(cd examples/training/se_e2_a/_train && mamba run -n gdp3 dp test -m deepmd-c.pb -s valid/init-Cu3Au1-bulk -n 1 -d inference)
(cd examples/training/dpa4/_train && mamba run -n gdp3 dp --pt test -m deepmd.pt2 -s valid/init-Cu3Au1-bulk -n 1 -d inference)
(cd examples/training/dpa4c/_train && mamba run -n gdp3 dp --pt-expt test -m deepmd-c.pt2 -s valid/init-Cu3Au1-bulk -n 1 -d inference)
```

These commands report errors and write reference/predicted values to
`inference.*.out` in each output directory.

## Configuration provenance and real datasets

The DPA4 configuration comes from `DPA4-Mini-OMat24-v20260805.json`, with the
`SeZM` model alias written as `dpa4`. The DPA4C configuration comes from
`DPA4C-Mini-OMat24-v20260819.json`. Both retain the supplied descriptor
architecture, fitting layer sizes, learning rate, loss, and HybridMuon
optimizer. For this CPU demonstration,
their element maps are reduced to Cu and Au; OMat24 statistics-file dependencies are
removed; AMP, compilation, TF32, EMA, and distributed training are disabled;
and training/checkpoint settings are shortened.

The DPA4C fitting network additionally uses `resnet_dt: true`. The supplied
`false` setting selects a compact compressed export path whose canonical
force operator is unavailable in the `gdp3` CPU build. Residual timesteps
select the general graph export path, allowing CPU compression and inference
while keeping the supplied layer sizes and float32 precision. This changes
the fitting network parameterization, so the CPU example is not directly
compatible with checkpoints using the original fitting configuration.
These examples train from scratch and do not require pretrained weights or
OMat24 files.

To train your own data, change `dataset.dataset_path` and the JSON model's
`type_map` together. Use system directories such as
`dataset/init-Cu3Au1-bulk/data.xyz`, with consistent composition in each
system. Supply labeled extended XYZ files with energies and forces and, for
these virial loss settings, virials (in eV) in the XYZ header. Keep the complete
element order when adapting a pretrained model. Choose a suitable training
duration and validation dataset for the scientific task instead of relying
on these smoke-run settings.
