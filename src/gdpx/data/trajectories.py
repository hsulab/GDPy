"""Read cached trajectories without submitting jobs or changing source caches."""
import copy
import io
import pathlib
import re

from ase.io import read
from gdpx.utils.archive import open_archive, find_driver_archive


def _natural(path):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(path))]


def _merge_segments(segments):
    frames = []
    for _, current in sorted(segments, key=lambda item: (item[0] == "", item[0])):
        if not current:
            continue
        if frames:
            if not all("step" in a.info for a in frames + current):
                raise ValueError("Restarted trajectories require step metadata.")
            start = current[0].info["step"]
            frames = [a for a in frames if a.info["step"] < start]
        frames.extend(current)
    if frames and all("step" in a.info for a in frames):
        steps = [a.info["step"] for a in frames]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            raise ValueError("Trajectory steps must increase strictly.")
    return frames


def read_cached_trajectories(directory, worker=None):
    """Return candidate trajectories from raw, compute, or exploration caches.

    Automatic reading supports ASE ``traj.xyz`` (including restart segments),
    ``cand.tar.zst`` and ``cand.tgz``. An explicit worker supplies a trajectory
    reader for other providers. Each candidate remains a separate list.
    """
    root = pathlib.Path(directory).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise ValueError("Cache source must be a calculation, compute, or exploration directory.")
    # Keyed by the logical trajectory file: live files override archived copies.
    files = {}
    candidates = {}
    archives = sorted({p.parent for name in ("cand.tar.zst", "cand.tgz") for p in root.rglob(name)}, key=_natural)
    for parent in archives:
        archive = find_driver_archive(parent)
        with open_archive(archive) as tar:
            for member in tar:
                rel = pathlib.PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe archive member {member.name!r}.")
                if member.isdir() and re.fullmatch(r"cand\d+", rel.name):
                    candidates[parent.joinpath(*rel.parts)] = archive
                if member.isfile() and rel.name == "traj.xyz":
                    files[parent.joinpath(*rel.parts)] = tar.extractfile(member).read().decode()
    for path in root.rglob("traj.xyz"):
        files[path] = None
    segments = {}
    for path, content in files.items():
        candidate = path.parent
        segment = ""
        if re.fullmatch(r"\d{4}\.run", candidate.name):
            segment, candidate = candidate.name, candidate.parent
        if worker is None:
            frames = read(path, ":") if content is None else read(io.StringIO(content), ":", format="extxyz")
            segments.setdefault(candidate, []).append((segment, frames))
        else:
            candidates.setdefault(candidate, find_driver_archive(candidate.parent))
    if worker is not None:
        for path in root.rglob("cand*"):
            if path.is_dir() and re.fullmatch(r"cand\d+", path.name):
                candidates[path] = None
        if not candidates:
            candidates[root] = None  # Backward-compatible single calculation directory.
        for path, archive in candidates.items():
            reader = copy.deepcopy(worker.driver)
            reader.directory = path
            frames = reader.read_trajectory(archive_path=archive)
            segments[path] = [("", frames)]
    trajectories = []
    for candidate in sorted(segments, key=_natural):
        frames = _merge_segments(segments[candidate])
        if not frames:
            continue
        for index, frame in enumerate(frames):
            frame.info["trajectory_id"] = str(candidate)
            frame.info["trajectory_source"] = str(root)
            frame.info["trajectory_frame"] = index
        trajectories.append(frames)
    if not trajectories:
        raise ValueError(f"No trajectories found in {root}. Automatic extraction supports ASE traj.xyz; supply a compute worker for other providers.")
    return trajectories
