import collections
import copy

from ase import Atoms


def get_tags_per_species(
    atoms: Atoms,
) -> dict[str, list[tuple[int, list[int]]]]:
    """Get tags per species.

    Args:
        atoms: An Atoms object with tags.

    Returns:
        A dict with chemical_formula as the key and the nested dict with
        atom indices to form the molecule.

    Example:

        .. code-block:: python

            >>> atoms = Atoms("PtPtPtCOCO")
            >>> tags = [0, 0, 0, 1, 1, 2, 2]  # or incontiguous [0,0,0,1,2,1,2]
            >>> atoms.set_tags(tags)
            >>> get_tags_per_species(atoms)
            >>> {'Pt3': [(0, [0,1,2])], 'CO': [(1, [3,4]), (2, [5,6])]}

    """
    # Get tags which is all zero for default
    tags = atoms.get_tags()

    # Group all indices by tags
    tag_to_indices: dict[int, list[int]] = {}
    for idx, tag in enumerate(tags):
        tag = int(tag)
        if tag not in tag_to_indices:
            tag_to_indices[tag] = []
        tag_to_indices[tag].append(idx)

    # Sort tags so output is deterministic
    tags_dict: dict[str, list[tuple[int, list[int]]]] = {}
    for tag in sorted(tag_to_indices.keys()):
        atomic_indices = sorted(tag_to_indices[tag])

        # Build sub-Atoms object
        entity = atoms[atomic_indices]
        formula = entity.get_chemical_formula()

        if formula not in tags_dict:
            tags_dict[formula] = []

        tags_dict[formula].append((tag, atomic_indices))

    return tags_dict


def reassign_tags_by_species(atoms: Atoms) -> Atoms:
    """"""
    tags_dict = get_tags_per_species(atoms)

    # Find substrate which has tag 0
    substrate: str = ""
    num_atoms_in_substrate: int = 0
    for k, v in tags_dict.items():
        num_instances = len(v)
        v_ = sorted(v, key=lambda x: x[0])  # Make sure we have the entry that has tag=0 at the first
        if v[0][0] == 0:
            assert num_instances == 1, f"`{atoms}` must have only one substrate (tag==0)."
            substrate = k
            num_atoms_in_substrate = len(v[0][1])  # type: ignore
            break
    else:
        tag_min = atoms.get_tags().min()
        assert tag_min > 0, f"`{atoms}` must have tags greater than 0 if no substrate (tag==0) is found."

    new_tags = [0] * num_atoms_in_substrate
    new_indices = list(range(num_atoms_in_substrate))

    current_tag = 1
    valid_keys = sorted([k for k in tags_dict.keys() if k != substrate])
    for species in valid_keys:
        for k, v in tags_dict[species]:  # type: ignore
            new_indices.extend(v)
            new_tags.extend([current_tag] * len(v))
            current_tag += 1

    new_atoms: Atoms = atoms[new_indices]  # type: ignore
    new_atoms.set_tags(new_tags)

    # Inherit info
    new_atoms.info = copy.deepcopy(atoms.info)

    return new_atoms


def sort_structures_by_tags(frames: list[Atoms]) -> list[Atoms]:
    """Sort atomic orders by their tags."""
    new_frames = []
    for atoms in frames:
        new_atoms = reassign_tags_by_species(atoms)
        new_frames.append(new_atoms)
    frames = new_frames

    return new_frames


def get_structure_chemical_notation(atoms: Atoms, chemical_types: list[str], padding_length: int = 4) -> str:
    """Get the chemical notation of a structure that can be sorted easily.

    Args:
        atoms: Atoms object.
        chemical_types: A list of chemical types sorted alphabetically.
        padding_length: The padding length of the number of each chemical type.

    Returns:
        A string of chemical notation.

    """
    counter = collections.Counter(atoms.get_chemical_symbols())

    notation = ""
    for k in chemical_types:
        num = counter.get(k, 0)
        if num >= 10**padding_length:
            raise RuntimeError(f"Too many atoms {num} for the padding length {padding_length}.")
        notation += f"{num:>0{padding_length}d}"

    return notation


def sort_structures_by_natoms_per_type(frames: list[Atoms], chemical_types: list[str]) -> list[Atoms]:
    """"""
    frames = sorted(
        frames,
        key=lambda a: get_structure_chemical_notation(a, chemical_types, padding_length=4),
    )

    return frames
