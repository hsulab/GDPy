"""Shared grouping of structures by comparator-defined similarity."""


def validate_similarity_comparator(comparator):
    """Require the prepared-representation comparator interface."""
    if not all(
        callable(getattr(comparator, method, None))
        for method in ("prepare_data", "compare_fingerprints")
    ):
        raise TypeError(
            "Similarity comparator must implement prepare_data and "
            "compare_fingerprints."
        )


def group_structures_by_similarity(structures, comparator, max_groups=None):
    """Group candidates greedily, keeping the first member as representative.

    Candidates are processed in the supplied order.  Each candidate joins the
    most recently created matching group, or starts a new group.  Processing
    stops once ``max_groups`` representatives have been found.
    """
    validate_similarity_comparator(comparator)
    representations = comparator.prepare_data(structures)
    if len(representations) != len(structures):
        raise ValueError("Comparator must prepare one representation per candidate.")

    representatives = []
    groups = {}
    for index, representation in enumerate(representations):
        for representative in reversed(representatives):
            if comparator.compare_fingerprints(
                representation, representations[representative]
            ):
                groups[representative].append(index)
                break
        else:
            representatives.append(index)
            groups[index] = [index]
            if max_groups is not None and len(representatives) == max_groups:
                break

    return representatives, groups
