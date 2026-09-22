# units

Use these units for gdpx configuration parameters unless a parameter's guide
specifies otherwise:

| Quantity | Unit |
| --- | --- |
| Time | femtosecond (`fs`) |
| Length | ångström (`Å`) |
| Energy | electronvolt (`eV`) |
| Force | `eV/Å` |
| Temperature | kelvin (`K`) |

For example, `timestep: 1.0` specifies a 1 fs molecular-dynamics time step,
and `fmax: 0.05` specifies a force threshold of 0.05 eV/Å.

Native software input files and provider-specific parameters can use their
own unit conventions. See the relevant {doc}`potential guide
<potentials/providers>` and {doc}`task guide <computations/tasks/index>`
for those settings.
