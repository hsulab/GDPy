# Add Corrections to Computed Structures

The `correct` operation evaluates a dataset with a dedicated runtime and
merges the resulting energy and force correction into each structure.

Define the correction model as a potential component, the single-point
calculation as an executor component, and combine them in a runtime. For
example, a DFT-D3 correction uses a DFT-D3 potential with an ASE `spc`
executor. The operation receives `structures` and that runtime.

Keeping correction execution in a separate runtime makes its software,
scheduler, and artifacts independent of the runtime that produced the original
dataset.
