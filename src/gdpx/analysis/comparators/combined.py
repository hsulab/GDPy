"""Composition of independently configured comparison criteria."""

from .comparator import BaseComparator


class CombinedComparator(BaseComparator):
    """Require every child comparator to consider the candidates similar.

    Direct comparisons short-circuit in configuration order. Prepared data
    contains one tuple per candidate, with one representation per criterion.
    Comparators without a preparation interface receive copied Atoms objects.
    """

    name = "combined"

    def __init__(self, comparators, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(comparators, (list, tuple)) or not comparators:
            raise ValueError("combined requires a nonempty list of comparators.")
        from . import create_comparator

        self.comparators = [create_comparator(item) for item in comparators]

    @staticmethod
    def _supports_preparation(comparator):
        return all(callable(getattr(comparator, method, None))
                   for method in ("prepare_data", "compare_fingerprints"))

    def prepare_data(self, structures):
        structures = list(structures)
        prepared = []
        for comparator in self.comparators:
            if self._supports_preparation(comparator):
                data = comparator.prepare_data(structures)
            else:
                data = []
                for atoms in structures:
                    candidate = atoms.copy()
                    candidate.calc = atoms.calc
                    data.append(candidate)
            if len(data) != len(structures):
                raise ValueError("Each comparator must prepare one representation per candidate.")
            prepared.append(data)
        return list(zip(*prepared))

    def compare_fingerprints(self, first, second):
        if len(first) != len(self.comparators) or len(second) != len(self.comparators):
            raise ValueError("Prepared criteria do not match the configured comparator count.")
        for comparator, left, right in zip(self.comparators, first, second):
            compare = comparator.compare_fingerprints if self._supports_preparation(comparator) else comparator.looks_like
            if not compare(left, right):
                return False
        return True

    def looks_like(self, first, second):
        return all(comparator.looks_like(first, second) for comparator in self.comparators)
