# EMT sampling and NNP retraining

This directory is a complete, locally runnable workflow example. It contains
the Cu starting structure, an EMT-labeled seed dataset, and a compatible
version-2 seed NNP.

Run it from this directory:

```console
gdp workflow validate workflow.yaml
gdp workflow run workflow.yaml
gdp workflow status workflow.yaml
```

The run is written to `workflow/`. Repeating the run command resumes from the
latest committed iteration.

The loop performs short EMT molecular dynamics, labels every sampled frame with
EMT, appends an immutable dataset shard, and warm-starts NNP training from the
model produced by the preceding iteration. It stops after three iterations.
The step directories are `0000.read`, `0001.sample`, `0002.label`,
`0003.collect`, and `0004.train`.

Collected data and model metadata are centralized under `workflow/artifacts/`.
Dataset versions are cumulative manifests over extxyz shards grouped by system;
model manifests reference the original model files in each iteration rather
than copying potentially large files.
