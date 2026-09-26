(active-learning-workflow)=

# repeated active-learning workflow

The complete migration of the former `active.yaml` is available at
`examples/workflows/active-learning.yaml`. It preserves the original
five-iteration exploration, labeling, training, and validation graph while
using the current `resources`, `steps`, `inputs`, and `options` schema.

This is a site-specific reference workflow, not a portable demo. Before
running it, provide the initial structures and DeepMD models, install the
DeepMD, LAMMPS, VASP, descriptor, and SSH dependencies, and replace the
Princeton-specific paths and scheduler settings.

The changing potential and datasets are explicit loop-carried state. The
workflow never updates an external `shared/` directory.

## Complete YAML

The documentation includes the checked-in example directly, so the displayed
configuration is always the same file used by the validation commands below.

```{literalinclude} ../../../examples/workflows/active-learning.yaml
:language: yaml
:caption: examples/workflows/active-learning.yaml
:linenos:
```

## Execution graph

The two validation branches are workflow targets. Following their dependencies
executes the same computational steps as the original configuration without a
user-defined join step:

| Step | Purpose | Depends on |
| --- | --- | --- |
| `read_stru` | Read the initial configurations. | — |
| `model_to_explore` | Assemble the runtime chain used for exploration. | exploration runtime |
| `run_nvt` | Run MD at 300, 400, 500, and 600 K. | initial structures, exploration model |
| `est_devi` | Evaluate committee model deviation. | MD trajectories |
| `select_devi` | Keep structures in the requested deviation interval. | deviation estimates |
| `select_desc` | Apply SOAP/CUR diversity selection. | deviation selection |
| `run_vasp` | Label selected structures with VASP. | descriptor selection |
| `sift_forces` | Remove structures with forces above the configured limit. | VASP results |
| `transfer` | Split structures into training and test datasets. | force-filtered structures |
| `train` | Train four DeepMD models, continuing across iterations. | updated training dataset |
| `model_spc` | Assemble the runtime used for validation. | trained potential |
| `test_spc_train` | Validate the training dataset. | updated dataset, validation runtime |
| `test_spc_test` | Validate the held-out dataset. | test dataset, validation runtime |

Shared upstream steps execute once per iteration even though the workflow has
two validation endpoints. The compiler creates the required internal barrier
for multiple targets automatically.

## Resources and provider configuration

The original variables are represented as typed resources:

- `potential`, `executor`, and `scheduler` resources form complete runtime
  resources for exploration, uncertainty estimation, and VASP labeling.
- Executor `broadcast` expands the four MD temperatures without duplicating
  the workflow step.
- Runtime `dispatch` controls batch size and shared working directories.
- Initial `dataset`, `selector`, `trainer`, and `validator` resources hold the
  reusable objects consumed by steps.
- `current_potential`, `training_data`, and `test_data` are committed state
  values used to rebuild dependent resources for every iteration.
- Dataset transfer writes immutable iteration deltas and returns named training
  and test snapshots; it never modifies the initial dataset directories.

The training step initializes from the models in `current_potential`. Its output
becomes the next iteration's potential, so there is no `active` flag, model-path
discovery, or shared potential symlink.

## Site parameters

Paths and commands that commonly change are declared together at the top of
the file:

```yaml
parameters:
  initial_structures: ./ini.xyz
  deepmd_config: /path/to/input-fp32.json
  current_models:
    - ./models/m0/deepmd.pb
    - ./models/m1/deepmd.pb
    - ./models/m2/deepmd.pb
    - ./models/m3/deepmd.pb
  remote_workdir: /remote/workdir
  vasp_command: /path/to/vasp_std
  vasp_incar: /path/to/INCAR_LABEL
  vasp_pp_path: /path/to/pseudopotentials
```

Edit these values directly, define a site profile, or override an individual
declared parameter:

```console
gdp workflow run examples/workflows/active-learning.yaml \
  --set initial_structures=./seed.xyz \
  --set remote_workdir=/scratch/$USER/project
```

Shell expansion occurs before the value reaches GDPy. Quote values when they
contain spaces or YAML punctuation.

## Inspect before running

Validation checks the schema, registered node types, constructor arguments,
references, and cycles without submitting calculations:

```console
gdp workflow validate examples/workflows/active-learning.yaml
gdp workflow plan examples/workflows/active-learning.yaml
gdp workflow graph examples/workflows/active-learning.yaml > active-learning.dot
```

Validation does not verify that external model files, executables, remote
hosts, scheduler accounts, or pseudopotentials exist. Those are checked only
when the corresponding resources and calculations are used.

## Repeated execution

The workflow uses:

```yaml
workflow:
  mode: repeat
  targets:
    - test_spc_train
    - test_spc_test
  max_iterations: 5

state:
  current_potential:
    initial: deepmd
    update: train
  training_data:
    initial: dataset
    update: {node: transfer, output: dataset}
  test_data:
    initial: dataset_test
    update: {node: transfer, output: dataset_test}
```

Each pass is stored under `iter.0000` through `iter.0004`. Validation steps can
report convergence and stop the repeated workflow early. Otherwise, all five
iterations run. Re-running the same command resumes from the saved iteration
state rather than replacing completed calculation directories.
The state manifest is committed before an iteration is marked finished, so an
interrupted iteration cannot publish a partial dataset or potential.
