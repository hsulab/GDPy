# DPA4C active learning with EMT labels

Run this local CPU example in `gdp3`, with DeepMD's experimental PyTorch backend
and DPA4C support installed. It uses the same 24 EMT-labeled Cu₃Au₁ structures
as the training examples. The four-atom periodic cell and short training runs
are intended to demonstrate the workflow.

From this directory:

```console
mamba activate gdp3
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export NUM_WORKERS=0
python prepare.py
gdp workflow validate bootstrap.yaml
gdp workflow run bootstrap.yaml
gdp workflow validate workflow.yaml
gdp workflow run workflow.yaml
gdp workflow status workflow.yaml
```

`prepare.py` reuses `../../training/prepare.py` to generate
`dataset/init-Cu3Au1-bulk/data.xyz` and writes an unlabeled `cu3au1.xyz` for MD.
The bootstrap trains two independently seeded models and exports their
compressed `deepmd-c.pt2` files. Its
`bootstrap/0001.save/potential.yaml` supplies the initial committee, so run
the bootstrap before validating the repeat workflow.

Each of two iterations performs 20 steps of DPA4C MD at 300 K, filters sampled
frames by `max_devi_f >= 0.001` eV/Å, randomly selects at most four eligible frames,
labels them with EMT, and adds an immutable shard to the cumulative dataset.
It then trains and compresses a new two-model committee, which supplies the next
iteration's MD. GDPy's committee uses the first model for MD forces and computes
force uncertainty across both models. `max_devi_f` is the largest standard
deviation of a Cartesian force component.

Training uses ten epochs with no additional batch floor and fits energy and
forces; virial loss is disabled. Each committee is trained from scratch on all
accumulated data. The sampling state carries compressed inference models; this
example does not carry training checkpoints between iterations.

For plain weight initialization in the installed `gdp3` DeepMD backend, use
`dp --pt-expt train input.json --init-model previous/model.ckpt.pt`. This starts
a fresh optimizer and step counter without the fine-tuning energy-bias adjustment.
GDPy's current `.pt` initialization maps to `--finetune`, so this checkpoint
initialization behavior is not wired into the example.

The threshold is a demonstration setting, not a calibrated accuracy criterion.
If no frames pass selection, the selection step exits without retraining or
committing that iteration. To exercise the full loop with every finite
nonnegative uncertainty eligible, start a fresh run with:

```console
gdp -d /tmp/dpa4c-al workflow run workflow.yaml --set force_deviation_min=0.0
```

The normal run lives in `workflow/`; each iteration's final `save` step writes
`potential.yaml`. Dataset and model manifests live under `workflow/artifacts/`.
Repeating the same run command resumes the saved state. Use a fresh run directory
after changing configuration or parameters. Generated datasets and run outputs
are ignored by Git.
