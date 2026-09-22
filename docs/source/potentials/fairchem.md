(potential-fairchem)=

# fairchem

The `fairchem` provider wraps FAIR-Chem pretrained predictors as potential calculators.

## Requirements

Install PyTorch and FAIR-Chem exposing `FAIRChemCalculator` and `pretrained_mlip.get_predict_unit` from `fairchem.core`. Make the requested pretrained model available to that loader.

## Configuration

```yaml
potential:
  provider: fairchem
  parameters:
    model: ./fairchem-cache/uma-s-1p1.pt
    head: omat
```

The model string has special semantics: gdpx uses its parent directory as
`FAIRCHEM_CACHE_DIR` and its filename stem as the upstream pretrained-model
name. The example requests `uma-s-1p1` from that loader; it is not a general
local-checkpoint loader. Choose a name and `head` supported by your installed
FAIR-Chem package and model access permissions.

`head` is passed as `task_name`. Only the first model entry is used, even if
`model` is a list. gdpx chooses CUDA when available and otherwise CPU.
