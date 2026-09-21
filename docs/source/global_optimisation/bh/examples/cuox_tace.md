(bh-cuox-tace-example)=

# Cu₄O₄ with TACE

This self-contained example searches for low-energy Cu₄O₄ clusters with the
small TACE-OAM-7M foundation model. Four random structures are relaxed in a
20 Å periodic vacuum cell, then two basin-hopping chains each attempt eight
moves. Every valid trial is locally minimized before acceptance at 500 K.
This is a short demonstration, not a converged global-minimum search.

## Input

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../../examples/global_optimisation/runtimes/tace_min_300.yaml
:language: yaml
```

All eight atoms are independently movable. Unlike the MatterSim extinction
example, this recipe applies no O–O extinction rule. The acceptance temperature
is a search parameter, not a molecular-dynamics temperature.

## Run

Install the {ref}`TACE extra <potential-tace>` and run from the repository root:

```shell
conda activate catorch3
python -m pip install -e '.[tace]'
OMP_NUM_THREADS=1 gdp -d ./run-cu4o4-bh-tace \
    --runtime ./examples/global_optimisation/runtimes/tace_min_300.yaml explore \
    ./examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
```

The first calculation downloads the approximately 29 MB checkpoint to
`~/.cache/tace/`. The runtime explicitly uses CPU and float32, without compilation
or CUDA extensions. Use `device: cuda` on a compatible GPU installation.

Under `run-cu4o4-bh-tace/expedition-0`, inspect `candidates.db` for relaxed
structures and acceptance metadata, `results/lineage/gen0001.png` for the search
history, and `tmp_folder/gen1/rounds/events.jsonl` for committed rounds.
Running the same command with the same directory resumes the search; a completed
search does not add duplicate calculations. See {doc}`cluster` for trajectory
export instructions.

## Model choice and validation

The 7M models are the small current general-purpose foundation models in the
upstream registry. On macOS arm64 with one Torch CPU thread, float32, the default
matscipy neighbor list, and the same eight-atom Cu₄O₄ geometry, median warmed
energy-and-force evaluations were:

| Model | Time per evaluation |
| --- | --- |
| TACE-OAM-7M | 86.7 ms |
| TACE-OMat24-7M | 82.1 ms |

Each timing uses 20 uncached evaluations after five warmups, excluding loading
and downloading. The difference is below 10%, so this example uses OAM-7M.
These measurements compare the two small models on this system; they do not
establish a universal speed ranking across hardware or the larger models.

Validated with TACE 0.2.2 at commit
`90e241bc9c74f7ed5c1e0be42fe7aee4bf5e9896`, Torch 2.14.0, NumPy 2.0.2,
ASE 3.27.0, matscipy 1.2.0, and Lightning 2.6.6. Both models matched direct
upstream energy and force calculations, and local checkpoint loading worked.
A symmetric Cu₄O₄ test structure relaxed to a maximum force of 0.045 eV/Å.
MatterSim 1.2.3 also passed an energy/force smoke test in the same environment.

The complete recipe produced four initial relaxed candidates and 16 relaxed
trials across all eight rounds, with a best energy of approximately
−36.1774 eV. Every stored relaxed structure had finite energy and forces.
The lineage figure was generated, and rerunning the completed search left the
database rows unchanged. Numerical differences can change the search trajectory.
