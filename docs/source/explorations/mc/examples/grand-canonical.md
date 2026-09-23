# Grand-canonical: Cu insertion and removal

The toy periodic 8 Å box begins with eight Cu atoms and one Au atom. Only Cu
exchanges with the reservoir. The Au atom remains at its initial position and
keeps the structure nonempty even if every Cu atom is removed. This example
uses exchange moves alone; it is not a model of a bulk phase or a relaxed
adsorption surface.

```{literalinclude} ../../../../../examples/monte_carlo/grand-canonical.yaml
:language: yaml
```

The explicit lattice region covers the full box. Its volume, $V=512$ Å³,
enters the acceptance rule; it is not just a selector. For atomic exchange,
the implemented probabilities are

$$
P_\mathrm{insert}=\min\left(1,
\frac{V}{(N+1)\Lambda^3}
\exp[-(\Delta E-\mu)/(k_\mathrm{B}T)]\right),
$$

$$
P_\mathrm{remove}=\min\left(1,
\frac{N\Lambda^3}{V}
\exp[-(\Delta E+\mu)/(k_\mathrm{B}T)]\right).
$$

Here $N$ counts exchangeable Cu particles in the region, not the Au atom,
and $\Lambda$ is the thermal de Broglie wavelength calculated from particle
mass and temperature. One `exchange` operator accepts exactly one particle
type and one chemical potential. Increasing `chempots[0]` favours insertion.
Unlike the semi-grand case, the absolute chemical potential matters and must
use the same energy reference as the potential. The value 2.0 eV is chosen
for this EMT demonstration, not calibrated to an experimental reservoir or
pressure. Using a different potential requires reconsidering this value.

## Run

From the repository root, use the shared {doc}`EMT single-point runtime <index>`:

```shell
gdp -d run-mc-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/grand-canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-grand-canonical
```

See the {ref}`Monte Carlo guide <monte-carlo>` for sampling limitations,
output interpretation, and restart instructions.
