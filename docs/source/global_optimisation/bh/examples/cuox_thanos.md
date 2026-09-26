(bh-cuox-thanos-example)=

# Cu<sub>4</sub>O<sub>4</sub> cluster with O–O extinction

This example exercises Thanos on a gas-phase Cu₄O₄ cluster using MatterSim.
The default periodic setting uses a 20 × 20 × 20 Å periodic vacuum cell, avoiding
MatterSim's automatic supercell construction for nonperiodic inputs.
It generates random structures, relaxes them, and rattles the surviving chains.
Structures with an O–O contact shorter than 1.6 Å are stored in the database but
excluded from the population. An accepted trial with such a contact terminates
its chain segment and triggers a replacement when moves remain.

The example deliberately **allows O–O bonds to form before extinction**. There
is no O–O prohibition in the builder or move. The ordinary move distance check
prevents severe overlap; Thanos applies its separate rule after minimization.

## Input

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu4o4_thanos.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../../examples/global_optimisation/runtimes/mattersim.yaml
:language: yaml
```

The random builder creates four candidates. Two chains each attempt eight rattle
moves, giving at most 20 relaxed structures including initialization. Each
atomic particle is independently selected with probability 0.8, with displacement
components uniform within ±1.8 Å. Generation 1 is the default final generation.

The Thanos rule uses `scope: all_atoms` so O–O detection does not depend on
particle tags. `distance.max: 1.6` and `max: 0` forbid any O–O pair below that
distance. This identifies an O₂ unit geometrically, including a unit bound to
copper; it does not identify its charge state or imply a detached neutral O₂
molecule.

The deliberately high acceptance temperature, 100,000 K, makes this a short
extinction demonstration: an uphill O–O-containing trial can be accepted and
then terminate its chain. It is an acceptance parameter, not a physical cluster
temperature. Lower it for a production search; then an O–O trial may be rejected
without terminating the current chain.

## Run and verify

Install MatterSim following {ref}`potential-mattersim`. The example uses the
1M checkpoint and was verified on CPU; its first use may download the model. From the repository
root, run:

```shell
OMP_NUM_THREADS=1 gdp -d ./run-cu4o4-bh-thanos \
    --runtime ./examples/global_optimisation/runtimes/mattersim.yaml explore \
    ./examples/global_optimisation/explorations/basin_hopping/cu4o4_thanos.yaml

python ./examples/global_optimisation/verify_cu4o4_bh_thanos.py \
    ./run-cu4o4-bh-thanos
```

The shared MatterSim runtime now limits relaxation to 20 steps; the previous
verified extinction sequence below used 150 steps. The shorter demo may not
trigger the same events or pass the verifier.

The verifier requires an actual minimized O–O-containing trial that was accepted,
marked extinct, replaced by an eligible candidate, and followed by another trial
in the new segment. It fails if the run only extinguishes initialization
structures or MC-rejected trials.

In the earlier 150-step CPU run with seed 7, chain 1 produced candidate 20 at round 6
with an O–O distance of approximately 1.38 Å. It was accepted and marked extinct;
the chain restarted and continued in segment 1. Initialization also produced
extinct candidates with O–O distances near 1.30 Å. Numerical differences between
model or library versions may change the trajectory; the verifier checks the
observed events rather than assuming a particular candidate ID.

Inspect these outputs under the working directory:

- `results/oo_extinct.xyz`: the actual relaxed structures containing forbidden
  O–O contacts, including initialization and trial structures.
- `results/thanos_verification.json`: verified extinction/restart events with
  candidate IDs, chain and round numbers, O–O distances, and replacement IDs.
- `candidates.db`: all minimized candidates, including extinct and rejected
  trials; trial metadata records `accepted`, `outcome`, `segment`, and parents.
- `tmp_folder/gen1/rounds/events.jsonl`: decision `4` marks chain replacement.

Use `oo_extinct.xyz` to view the offending O₂ units. Normal chain trajectory
exports contain replacement starts rather than the extinct endpoints.
