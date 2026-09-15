(operations)=

# Operations

Operations consume variables or earlier operation results. Computation-facing
operations accept a complete `runtime` or `runtime_chain` variable rather
than separate execution components.

`compute` executes structures, `extract` reads trajectories, `select`
chooses structures, `react` runs path calculations, `train` creates model
artifacts, and `validate` evaluates model quality. Each operation stores its
state beneath its session directory so a session can be resumed.

References use the normal session syntax, for example
`runtime: ${vx:local_relaxation}` or `structures: ${op:build}`.
