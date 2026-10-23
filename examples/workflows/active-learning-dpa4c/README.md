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

## Simulated-annealing sampling

`sa.yaml` uses population SA instead of ordinary MD. It reuses `common.yaml`
and `dpa4c.json`. Prepare the data and bootstrap a committee as above, then run
from this directory:

```console
gdp workflow validate sa.yaml
gdp workflow run sa.yaml
gdp workflow status sa.yaml
```

`parameters.initial_models` points to the compressed bootstrap models. Update
those paths when using an existing committee or an older bootstrap directory
layout. The default paths use the current flat step layout. As with the ordinary
MD example, DPA4C inference and training require the experimental PyTorch backend.

Each of two learning iterations starts a fresh SA search from `cu3au1.xyz`,
using the current committee for both MD and minimisation. It minimises the
initial structure, launches two 20-step trajectories cooling from 600 to 300 K,
and minimises every second saved MD snapshot plus the endpoint. The MD save
period is five integration steps; minimisation also saves intermediate frames
with a period of five steps. These short budgets demonstrate the workflow.

The `sample` operation exposes `dynamics` and `minimisation`
separately. The latter includes both generation-0 relaxation and minimisations
of selected MD snapshots. `md_trajectories` and `min_trajectories` extract each
source separately, including intermediate minimisation frames. No worker
methods are mixed before selection. The population comparator does not filter
these trajectories; SA's `minimisation.selector` only controls which MD frames
are relaxed. The default `sample.minimised_candidates` head contains all successful final
minima as structures, including duplicates; it does not contain workers.

Each branch has its own committee scoring and uncertainty-selection steps.
`md_label_budget` and `min_label_budget` default to four, independently limiting
the number of labels from each source. Both selectors use the threshold in
`force_deviation_min`. Selected structures are labeled with EMT in separate
`label_md` and `label_min` steps. `collect_md` appends the MD labels to the
existing dataset; `collect_min` appends the minimisation labels to that result.
Their dataset shards use `active_md` and `active_min` prefixes to distinguish
the sources. The resulting dataset is used for retraining and becomes the next
iteration's dataset state. No cross-source deduplication is performed.

Only reference-labeled data enter the training dataset; the latest trained
committee is used in the next learning iteration. No SA checkpoint is carried
between models.

The default threshold comes from `common.yaml`. If either branch selects no frames, that
iteration stops before training/state commit; the other branch does not borrow
its unused budget. To make every finite nonnegative
uncertainty eligible in a fresh run:

```console
gdp -d /tmp/dpa4c-sa-al workflow run sa.yaml --set force_deviation_min=0.0
```

The default run directory is `sa/`. YAML validation and a shortened EMT-only
SA-to-separated-extraction smoke test have been checked; full DPA4C inference and retraining
were not run for this example.
