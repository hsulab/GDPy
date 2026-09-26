# Minimal EMT and NNP loop

This small example demonstrates transactional active-learning state without a
cluster scheduler or an external `shared/` directory. EMT samples and labels Cu
structures, each iteration adds an immutable dataset shard, and the NNP trainer
warm-starts from the model committed by the preceding iteration.

```{literalinclude} ../../../examples/workflows/active-learning-emt-nnp/workflow.yaml
:language: yaml
```

The example directory includes all three inputs:

- `cu.xyz`, the structure from which each short EMT trajectory starts;
- `dataset/seed-Cu-bulk/*.xyz`, an initial EMT-labeled training set;
- `seed-nnp.npz`, a version-2 NNP archive trained from the bundled data.

Enter the self-contained directory, then validate and run the workflow without
any overrides:

```console
cd examples/workflows/active-learning-emt-nnp
gdp workflow validate workflow.yaml
gdp workflow run workflow.yaml
gdp workflow status workflow.yaml
```

This is intentionally the smallest closed data-and-model loop. It labels every
sampled frame and does not include uncertainty selection. A production workflow
would normally insert selection between `sample` and `label`.

The loop runs three iterations. Its durable outputs are centralized without
duplicating trained models:

```text
workflow/
├── artifacts/
│   ├── datasets/training_data/
│   │   ├── systems/active-Cu4-bulk/{0000,0001,0002}.xyz
│   │   └── versions/{initial,0000,0001,0002}.yaml
│   └── models/potential/
│       ├── initial.yaml
│       └── iterations/{0000,0001,0002}.yaml
├── state/{initial,current}.yaml
└── iter.0000/ ... iter.0002/
```

The initial dataset version references the `dataset/` root in the example
directory; it neither enumerates nor copies its files. Each later version keeps
that root and lists only the extxyz shards collected by the workflow. Model
versions likewise point to models in their original locations, so neither large
datasets nor large model files are copied or hashed.
