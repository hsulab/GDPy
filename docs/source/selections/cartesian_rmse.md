# Cartesian RMSE trajectory selection

Sample each ordered minimization trajectory independently:

```yaml
# select-mintraj.yaml
method: cartesian_rmse
rmse_interval: 0.1       # Angstrom; required, positive
include_first: false    # default: do not force the initial frame
include_last: true      # default: always include the final frame
mic: true               # minimum-image displacement for periodic cells
# group_expression: "tag 1"  # optional atom subset; default is all atoms
```

Starting with the initial frame as reference, select the next frame whose
Cartesian RMSE from the last selected frame reaches `rmse_interval`, then update
the reference. This is displacement from the reference, not accumulated path
length. RMSE is `sqrt(sum(displacement**2)/(3*N))`, over the selected atoms and
Cartesian components, in Angstrom. No translation, rotation, or atom permutation
alignment is performed. Frozen atoms contribute zero displacement when all atoms
are used. Use `group_expression` to measure only a mobile atom subset.

`include_first` and `include_last` force endpoint inclusion. Setting
`include_last: false` still permits the final frame to be selected by the interval
criterion. A single-frame trajectory is included if either endpoint flag is true.
The final frame may be closer than the requested interval. Atom ordering and
composition must remain consistent; periodic minimum-image comparison requires a
fixed cell. This samples structural changes, and does not filter unphysical frames.

## Compute and exploration caches

Create a source file pointing to a completed `gdp compute` or exploration output:

```yaml
# mintraj-source.yaml
method: cached_trajectories
directory: ./_run
```

```shell
gdp -d _select-mintraj-rmse-0p1 select select-mintraj.yaml -s mintraj-source.yaml
```

The source reader recursively discovers ASE `traj.xyz` files and candidate
archives (`cand.tar.zst` and `cand.tgz`), including exploration generation folders.
It merges restart segments by their step numbers and prefers live copies over the
same archived file. It does not load a potential or submit calculations. Each
frame carries `trajectory_id`, `trajectory_source`, and `trajectory_frame`, which
are retained in `selected_frames.xyz`. Frames from different candidates are never
compared with one another. All stored trajectory frames are read; no extra dump
interval is applied. Use a new selection output directory after changing the
interval or source, because the standard selector cache is reused otherwise.

## Workflow extraction

`extract_cache` can now omit `compute` for these ASE caches:

```yaml
steps:
  trajectories:
    __type__: extract_cache
    options:
      cache_wdirs: [./_run]
      save_h5: true
```

Automatic extraction returns an `AtomsNDArray` with axes source, candidate, frame;
ragged trajectories are padded. The selector treats each final-axis trajectory
independently even after an HDF5 cache roundtrip. The HDF5 serializer retains its
existing metadata policy; use the source builder above when exporting trajectory
path identifiers. An explicit `compute` input continues to supply the driver
reader for other calculator output formats, including raw calculation directories.
