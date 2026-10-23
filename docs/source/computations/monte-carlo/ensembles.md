# Ensembles

Standard MC, biased MC, HMC, and hybrid biased MC define thermodynamic settings under
`executor.parameters.ensemble`, using the same mapping format as
{doc}`MD <../molecular-dynamics/ensembles>`.

| Method | Required fields | Allowed changes |
| --- | --- | --- |
| `nvt` | `temperature` | Positions and orientations at fixed composition and cell |
| `uvt` | `temperature`, `chemical_potentials` | Particle count and composition at fixed cell |
| `suvt` | `temperature`, `chemical_potentials` | Species identities at fixed total particle count and cell |

MC and HMC do not currently implement `nve` or `npt` sampling. Selecting an
ensemble validates and supplies acceptance settings; the configured operators
determine which changes are proposed. See the {ref}`MC sampling limitations
<monte-carlo>` for the limitations of individual proposals and the
{doc}`HMC guide <hmc>` for its MD/MC acceptance behavior.

The fragments below belong under `executor.parameters`.

## NVT

```yaml
ensemble:
  method: nvt
  temperature: 1200
```

Use composition-preserving proposals such as `move`, `rotate`, or `swap`.
`swap` exchanges particle positions, preserving species counts.
See the {doc}`canonical example <mc/examples/canonical>`.

## UVT

```yaml
ensemble:
  method: uvt
  temperature: 1200
  chemical_potentials:
    Cu: 2.0
```

Particle-exchange operators use the reservoir chemical potentials for insertion
and removal. Values must use the same energy reference as the potential;
the value above is an example input, not a calibrated reservoir.
See the {doc}`grand-canonical example <mc/examples/grand-canonical>`.

## SUVT

```yaml
ensemble:
  method: suvt
  temperature: 1200
  chemical_potentials:
    Cu: 0.0
    Ni: 0.2
```

Use `swap_type` for species changes while preserving total atom count.
Supply a chemical potential for every participating species; acceptance uses
the difference between the new and old species values. Particle insertion and
removal are rejected for this ensemble. See the
{doc}`semi-grand-canonical example <mc/examples/semi-grand-canonical>`.

## Shared settings and validation

Temperature is in K and must be finite and positive. Chemical potentials are
finite values in eV per particle, keyed by species or particle name. NVT does
not accept chemical potentials. Define these settings on the ensemble, not on
individual MC operators. The former `custom` ensemble is no longer supported;
use {doc}`basin hopping <../../global_optimisation/basin_hopping>` for structure
search with per-operator acceptance settings.

## HMC inheritance

Declare `parameters.ensemble` only on the HMC parent. Keep the MD and MC
executors in the ordered `parameters.setup` list. For NVT, both children use
the parent temperature. For UVT and SUVT, MD runs NVT at the current composition
and shared temperature; MC handles particle exchange or species changes.

Children cannot redeclare the ensemble, add regulator targets, or introduce
temperature/pressure ramps. MD retains its own timestep and regulator algorithm;
MC retains its own operators and proposal count. The potential is shared.
A parent broadcast at `ensemble.temperature` changes MD and MC together.
See the {doc}`complete HMC input <hmc>`.

## Biased Monte Carlo

{doc}`Biased Monte Carlo <biased-mc>` also accepts `nvt`, `uvt`, and `suvt`
with the same thermodynamic fields and operator restrictions. It uses minimised
endpoint energies for acceptance and permits geometry filters. These settings
define biased search acceptance, not equilibrium sampling of the original
potential. Standard MC and HMC still reject enabled distance checks; HMC
does not accept biased-MC children.
