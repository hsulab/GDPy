"""Validate newly written WAVECAR checkpoints before pruning archived copies."""
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

from ase.io import ParseError

MARKER = ".electronic-checkpoint.json"
EVIDENCE = ("WAVECAR", "OUTCAR", "OSZICAR", "INCAR", "CONTCAR")


def signature(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def tail(path, size=1024 * 1024):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - size))
        return stream.read().decode(errors="replace")


def complete_wavecar(path):
    """Check headers and record extents without reading GBs of coefficients.

    Supports VASP's 45200/45210 and 53300/53310 record formats, both byte orders.
    This detects truncation and invalid layouts, not arbitrary coefficient damage.
    """
    try:
        path = Path(path)
        size = path.stat().st_size
        with path.open("rb") as stream:
            first = stream.read(24)
            if len(first) != 24:
                return False
            for endian in ("<", ">"):
                values = struct.unpack(endian + "3d", first)
                if not all(math.isfinite(x) and x == int(x) for x in values):
                    continue
                record, spins, tag = map(int, values)
                if record < 104 or record % 8 or record > size or spins not in (1, 2):
                    continue
                if tag not in (45200, 45210, 53300, 53310):
                    continue
                stream.seek(record)
                header = stream.read(104)
                if len(header) != 104:
                    continue
                metadata = struct.unpack(endian + "13d", header)
                if not all(math.isfinite(x) for x in metadata):
                    continue
                nk, nb = metadata[:2]
                if nk < 1 or nb < 1 or nk != int(nk) or nb != int(nb) or metadata[2] <= 0:
                    continue
                nk, nb = int(nk), int(nb)
                header_records = ((4 + 3 * nb) * 8 + record - 1) // record
                if tag in (45200, 45210) and header_records != 1:
                    continue
                total_records = 2 + spins * nk * (header_records + nb)
                if size > total_records * record or size <= (total_records - 1) * record:
                    continue
                coefficient_bytes = 16 if tag in (45210, 53310) else 8
                valid = True
                for group in range(spins * nk):
                    offset = (2 + group * (header_records + nb)) * record
                    stream.seek(offset)
                    raw = stream.read(32)
                    if len(raw) != 32:
                        valid = False
                        break
                    info = struct.unpack(endian + "4d", raw)
                    npl = info[0]
                    if not all(math.isfinite(x) for x in info) or npl < 1 or npl != int(npl):
                        valid = False
                        break
                    if npl * coefficient_bytes > record:
                        valid = False
                        break
                    last_payload_end = offset + (header_records + nb - 1) * record + int(npl) * coefficient_bytes
                    if last_payload_end > size:
                        valid = False
                        break
                if valid:
                    return True
        return False
    except (OSError, ValueError, OverflowError, struct.error):
        return False


def final_scf_converged(directory):
    """Require a completed final ionic summary and both electronic tolerances."""
    incar = (directory / "INCAR").read_text()
    clean = "\n".join(re.split(r"[!#]", line, maxsplit=1)[0] for line in incar.splitlines())
    matches = re.findall(r"\bEDIFF\s*=\s*([0-9.eEdD+-]+)", clean, flags=re.I)
    tolerance = float(matches[-1].replace("D", "E").replace("d", "e")) if matches else 1e-4
    if not math.isfinite(tolerance) or tolerance <= 0:
        return False
    lines = [line.strip() for line in tail(directory / "OSZICAR").splitlines() if line.strip()]
    if not lines or not re.match(r"^\d+\s+(?:F|T)\s*=", lines[-1]):
        return False
    for line in reversed(lines[:-1]):
        fields = line.split()
        if fields and fields[0].endswith(":") and len(fields) >= 5 and fields[1].isdigit():
            changes = [float(x.replace("D", "E").replace("d", "e")) for x in fields[3:5]]
            return all(math.isfinite(x) and abs(x) <= tolerance for x in changes)
        if re.match(r"^\d+\s+(?:F|T)\s*=", line):
            break
    return False


def begin_electronic_checkpoint(directory):
    directory = Path(directory)
    (directory / MARKER).unlink(missing_ok=True)
    return {name: signature(directory / name) for name in EVIDENCE}


def certified_checkpoint(directory):
    """A certificate is invalidated when any recorded evidence changes."""
    directory = Path(directory)
    try:
        if (directory / MARKER).is_symlink():
            return False
        saved = json.loads((directory / MARKER).read_text())
        return (
            saved.get("version") == 1
            and set(saved.get("files", {})) == set(EVIDENCE)
            and all(
                saved["files"][name] is not None
                and saved["files"][name] == signature(directory / name)
                for name in EVIDENCE
            )
            and complete_wavecar(directory / "WAVECAR")
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def finalize_electronic_checkpoint(directory, before, keep, log=print):
    """Called only after subprocess success; no certification means no deletion."""
    directory = Path(directory)
    try:
        current = {name: signature(directory / name) for name in EVIDENCE}
        if any(value is None or value[0] == 0 for value in current.values()):
            log("Electronic checkpoint not certified: required output is missing.")
            return []
        if any(current[name] == before[name] for name in ("WAVECAR", "OUTCAR", "OSZICAR")):
            log("Electronic checkpoint not certified: output was not refreshed by this run.")
            return []
        accounting = tail(directory / "OUTCAR")
        if "General timing and accounting" not in accounting or "Elapsed time (sec):" not in accounting:
            log("Electronic checkpoint not certified: no completed VASP accounting output.")
            return []
        from ase.io import read
        geometry = read(directory / "CONTCAR", format="vasp")
        if len(geometry) == 0 or geometry.get_volume() <= 0:
            log("Electronic checkpoint not certified: invalid CONTCAR geometry.")
            return []
        if not final_scf_converged(directory) or not complete_wavecar(directory / "WAVECAR"):
            log("Electronic checkpoint not certified: final SCF or WAVECAR validation failed.")
            return []
        descriptor, temporary = tempfile.mkstemp(prefix=".electronic-checkpoint-", dir=directory)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump({"version": 1, "files": current}, stream)
            os.replace(temporary, directory / MARKER)
        finally:
            temporary.unlink(missing_ok=True)
        log("Certified completed electronic WAVECAR checkpoint.")
        if keep is None:
            return []
        archives = sorted(
            path for path in directory.glob("[0-9][0-9][0-9][0-9].run")
            if path.is_dir() and not path.is_symlink()
        )
        usable = [path for path in archives if certified_checkpoint(path)]
        if len(usable) < keep:
            log(f"Retaining all older electronic files: only {len(usable)} certified archives, need {keep}.")
            return []
        oldest_retained = usable[-keep]
        removed = []
        for archive in archives:
            if archive >= oldest_retained:
                continue
            for name in ("WAVECAR", "CHGCAR"):
                path = archive / name
                if path.is_symlink():
                    continue  # Never traverse external or shared electronic files.
                if path.is_file():
                    try:
                        path.unlink()
                        removed.append(str(path))
                    except OSError as exc:
                        log(f"Could not prune {path}: {exc}")
        log(f"Pruned {len(removed)} old electronic files; kept {keep} certified archives plus active output.")
        return removed
    except (OSError, ValueError, TypeError, KeyError, IndexError, AssertionError, ParseError, StopIteration, RuntimeError) as exc:
        log(f"Skipping electronic checkpoint retention: {exc}")
        return []
