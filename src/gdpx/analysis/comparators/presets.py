"""Concise definitions that expand into independent comparison criteria."""

from .combined import CombinedComparator


def energetic_oganov_fingerprint(
    ene_diff_tol=1.0,
    group_expression=None,
    cos_dist_max=5e-3,
    rcut=20.0,
    binwidth=0.05,
    sigma=0.02,
    nsigma=4,
    pbc=True,
    maxdims=None,
    **kwargs,
):
    """Return energy AND Oganov comparison with a flat union of parameters.

    Component options are forwarded to the combination and both children.
    Group selection affects fingerprints; energy uses the complete structure.
    """
    return CombinedComparator(comparators=[
        {"method": "energy", "ene_diff_tol": ene_diff_tol, **kwargs},
        {
            "method": "oganov_fingerprint",
            "group_expression": group_expression,
            "cos_dist_max": cos_dist_max,
            "rcut": rcut,
            "binwidth": binwidth,
            "sigma": sigma,
            "nsigma": nsigma,
            "pbc": pbc,
            "maxdims": maxdims,
            **kwargs,
        },
    ], **kwargs)
