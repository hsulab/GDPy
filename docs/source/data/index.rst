Data Array Design
=================

``AtomsNDArray`` is an N-dimensional container for ASE ``Atoms`` objects
with sparse storage, labeled axes, and selection tracking.
It serves as the canonical data bus type for structure data flowing
through the session graph DAG.

Design Concepts
---------------

1. **ND Sparse Storage**

   An ``AtomsNDArray`` holds ``Atoms`` objects in a flat list (``_data``)
   and tracks which cells of the N-dimensional grid are populated via a
   **boolean mask** (``_markers``) of shape ``_shape``. Only the actual
   ``Atoms`` objects are stored — empty cells have no memory overhead.

   .. code-block:: python

       arr = AtomsNDArray([
           [atoms_00, atoms_01, None],        # band 0: 2 frames
           [atoms_10, atoms_11, atoms_12],    # band 1: 3 frames
       ])
       arr.shape   # (2, 3)
       arr.markers # [[True, True, False], [True, True, True]]
       len(arr)    # 5  (non-None Atoms)

2. **Labeled Axes (dims / coords)**

   Each axis can have a human-readable name and optional coordinate labels
   (similar to xarray or pandas Index). This enables label-based selection.

   .. code-block:: python

       arr = AtomsNDArray(
           data,
           dims=("temperature", "time"),
           coords={"temperature": [300, 400, 500]},
       )
       arr.dims            # ("temperature", "time")
       arr.coords          # {"temperature": array([300, 400, 500])}
       arr.sel(temperature=300)  # select by coordinate label

   Default dims are ``("dim_0", "dim_1", ...)`` when not specified.

3. **Selections via Markers**

   The boolean marker mask tracks which cells are "active" for downstream
   processing (selectors, validators, drivers). Key operations:

   .. code-block:: python

       arr.markers                        # get current markers
       arr.markers = new_mask             # set markers (bool or int array)
       arr.reset_markers()                # restore to initial state
       arr.get_marked_structures()        # list[Atoms] of selected cells
       arr.init_markers                   # original markers at construction

4. **Multiple Indexing Modes**

   Integer/slice indexing is unchanged and fully supported alongside
   label-based access:

   .. code-block:: python

       arr[0, :]                  # first row, all columns
       arr[:, -1]                 # last column per row (may return None)
       arr[0, 1, 2]               # single Atoms (3D)
       arr.sel(temperature=300)   # label-based coordinate selection
       arr.loc["300K", :]         # label-based indexing
       arr.take_last(axis=-1)     # last valid (non-None) along axis

5. **Convenience Methods**

   ======================  ====================================================
   Method                  Description
   ======================  ====================================================
   ``.take_last(axis=-1)`` Last non-None element along an axis (avoids padding)
   ``.tolist()``           Reconstruct nested list with ``None`` placeholders
   ``.get_marked_structures(markers=None)``  List of selected Atoms objects
   ``.sel(**kwargs)``      Select by coordinate label (exact or slice)
   ``.loc[key]``           Label-based indexer (pandas-like)
   ``.save_file(target)``  Serialize to HDF5
   ``.from_file(target)``  Deserialize from HDF5 (classmethod)
   ======================  ====================================================

API Reference
-------------

``__init__(data=None, markers=None, *, dims=None, coords=None)``
   Construct from a nested list of ``Atoms`` / ``None``, or copy from
   another ``AtomsNDArray``.

   - ``data``: nested list or ``AtomsNDArray``
   - ``markers``: initial markers (bool array or integer coordinate array)
   - ``dims``: ``tuple[str, ...]`` — axis names, defaults to ``("dim_0", ...)``
   - ``coords``: ``dict[str, array_like]`` — coordinate labels per named axis

``shape`` property → ``tuple[int, ...]``
   Bounding-box shape of the N-dimensional grid.

``ndim`` property → ``int``
   Number of dimensions.

``dims`` property → ``tuple[str, ...]``
   Names of each axis.

``coords`` property → ``dict[str, np.ndarray]``
   Coordinate labels per named axis. Only populated for axes that were
   explicitly assigned coordinates at construction.

``markers`` property → ``np.ndarray`` of bool
   Current selection mask of shape ``shape``.

``init_markers`` → ``np.ndarray`` of bool
   The original marker mask at construction time (for resetting).

``reset_markers()`` → ``None``
   Reset markers to the initial state.

``get_marked_structures(markers=None)`` → ``list[Atoms]``
   Return a flat list of selected Atoms objects.

``sel(**kwargs)`` → ``AtomsNDArray``
   Label-based selection by coordinate value.

   .. code-block:: python

       arr.sel(temperature=300)         # exact match
       arr.sel(temperature=[300, 500])  # multiple values
       arr.sel(time=slice(0, 50))       # slice by label

   Raises ``KeyError`` if the coordinate label is not found.

``loc`` → ``_LocIndexer``
   Label-based indexer supporting ``arr.loc[key]`` syntax.

``take_last(axis=-1)`` → ``AtomsNDArray`` or ``Atoms``
   Return the last non-``None`` element along the given axis, reducing
   ndim by 1. If the result has a single element, returns the ``Atoms``
   object directly.

   .. code-block:: python

       arr.take_last()          # last valid along axis -1
       arr.take_last(axis=1)    # last valid frame per trajectory
       arr[0].take_last()       # single Atoms (first traj, last frame)

``tolist()`` → ``list``
   Reconstruct the nested list representation with ``None`` for empty cells.

``__getitem__(key)`` → ``Atoms`` or ``list[Atoms]``
   Supports integer, slice, and tuple indexing (unchanged from original).

   - Scalar indices return a single ``Atoms`` (or ``None`` for empty cells).
   - Slice indices return nested lists via ``_reshape_data``.

``__len__()`` → ``int``
   Number of non-``None`` Atoms objects in the array.

``__repr__()`` → ``str``
   ``AtomsNDArray(nimages=N, shape=S, dims=D)``

``__eq__(other)`` → ``bool``
   Compare shape, dims, markers, and Atoms data element-wise.

``save_file(target, grp_name="images")`` → ``None``
   Serialize to HDF5. Writes shape, dims, coords, markers, and padded
   atomic arrays. Accepts a file path or open ``h5py.File``.

   .. code-block:: python

       arr.save_file("output.h5")
       with h5py.File("output.h5", "a") as f:
           arr.save_file(f, grp_name="subset")

``from_file(target, grp_name="images")`` → ``AtomsNDArray``
   Deserialize from HDF5. Backward compatible with files that do not
   contain dims/coords (default names are assigned automatically).

Internal Methods
^^^^^^^^^^^^^^^^

``_from_components(shape, data, markers, ind_map, dims, coords)`` → ``AtomsNDArray``
   Fast internal constructor that avoids re-processing already-structured
   data. Used by ``from_file`` and ``__getitem__`` when the result is a
   sub-array.

``_convert_images(grp, images)`` → ``None``
   Write a flat list of ``Atoms`` to HDF5 datasets (padded to max natoms).

``_from_hd5grp(grp)`` → ``list[Atoms]``
   Reconstruct ``Atoms`` objects from HDF5 group.

``_LocIndexer``
   Provides label-based indexing via ``arr.loc[key]``. Handles integer,
   slice, and tuple keys with coordinate label resolution.

Migration Guide
---------------

Changes in marker format (integer → boolean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, ``.markers`` returned a ``(M, ndim)`` integer array of
coordinate tuples. Now it returns a boolean array of shape ``.shape``.

+--------------------------------------+--------------------------------------+
| Old code                             | New code                             |
+======================================+======================================+
| ``len(arr.markers)``                 | ``int(np.count_nonzero(arr.markers))``|
+--------------------------------------+--------------------------------------+
| ``for ind in arr.markers: ...``      | ``for ind in np.argwhere(arr.markers):`` |
+--------------------------------------+--------------------------------------+
| ``arr.markers = [[0,1], [0,2]]``     | ``arr.markers = [[0,1], [0,2]]``     |
|                                      | (still accepted, auto-converted)     |
+--------------------------------------+--------------------------------------+
| ``np.array_equal(arr.markers, m)``   | ``np.array_equal(arr.markers, m)``   |
|                                      | (unchanged if m is also bool)        |
+--------------------------------------+--------------------------------------+

HDF5 backward compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Old HDF5 files (without ``dims`` / ``coords`` datasets) load correctly.
Dimensions default to ``("dim_0", "dim_1", ...)`` and coordinates to
empty dict.

.. code-block:: python

    # Old file — still works
    arr = AtomsNDArray.from_file("old_results.h5")
    arr.dims    # ("dim_0", "dim_1")
    arr.coords  # {}
