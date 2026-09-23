(hmc-examples)=

# Examples

These EMT examples alternate **20 NVT MD steps and five MC proposals** per
cycle, for five cycles at 1200 K. Each input includes initialization, MD, and
MC runtimes and uses an included periodic 32-atom structure. Run the commands
on each example page from the GDPy repository root.

Read the {ref}`Hybrid Monte Carlo guide <hybrid-monte-carlo>` for procedure
configuration, sampling limitations, output interpretation, and restart details.

```{toctree}
:maxdepth: 1

canonical
semi-grand-canonical
```
