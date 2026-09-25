(hmc-examples)=

# Examples

The EMT examples alternate **20 NVT MD steps and five MC proposals** per cycle,
for five cycles at 1200 K. Each input includes initialization, MD, and MC
runtimes. Run the commands on each example page from the GDPy repository root.

For Cu(111) oxidation with xreac, use the short
{doc}`grand-canonical MC example <../../mc/examples/cu111-oxidation-xreac>`.

Read the {ref}`Hybrid Monte Carlo guide <hybrid-monte-carlo>` for cycle
configuration, sampling limitations, output interpretation, and restart details.

```{toctree}
:maxdepth: 1

canonical
semi-grand-canonical
```
