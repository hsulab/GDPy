# explore with a genetic algorithm

The `explore` operation combines an exploration strategy with a runtime used
to evaluate or relax generated candidates. The exploration strategy proposes
structures; execution remains responsible for running the configured
potential and executor.

Define the candidate builder and genetic-algorithm settings as exploration
inputs, then pass one complete runtime for candidate relaxation. Downstream
`select` and `compute` operations can consume the default
`minimised_candidates` head directly. To extract relaxation trajectories, pass
`{node: search, output: minimisation}` to `extract`, where `search` is the
exploration step. For MD trajectories, select the `dynamics` head instead.

Multiple evaluation levels must be represented as an explicit runtime chain.
gdpx does not infer pairings between lists of potentials and executors.
