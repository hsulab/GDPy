"""Plot two completed benchmark runs without comparing absolute model energies."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Contains xreac/ and mattersim/ run directories")
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for provider, label, color in (("xreac", "ReaxFF / xreac", "#157a6e"),
                                   ("mattersim", "MatterSim 1M", "#ad4b27")):
        root = args.directory / provider
        # Historical benchmark outputs used reax for the Python implementation.
        if provider == "xreac" and not root.exists():
            root = args.directory / "reax"
        if not (root / "summary.json").exists():
            parser.error(f"{provider}: no summary; the time limit may have expired during startup")
        summary = json.loads((root / "summary.json").read_text())
        if not summary["stages"].get("nve", {}).get("steps", 0):
            parser.error(f"{provider}: no completed NVE steps to plot")
        nvt = np.atleast_1d(np.genfromtxt(root / "nvt.csv", delimiter=",", names=True))
        nve = np.atleast_1d(np.genfromtxt(root / "nve.csv", delimiter=",", names=True))
        offset = summary["stages"]["nvt"]["duration_fs"]
        time = np.concatenate((nvt["time_fs"], nve["time_fs"] + offset)) / 1000
        temperature = np.concatenate((nvt["temperature_K"], nve["temperature_K"]))
        axes[0].plot(time, temperature, color=color, label=label, lw=1)
        axes[0].axvline(offset / 1000, color=color, ls="--", lw=1)
        axes[1].plot(nve["time_fs"] / 1000,
                     (nve["total_eV"] - nve["total_eV"][0]) * 1000 / summary["atoms"],
                     color=color, label=label, lw=1)
        timing = summary["stages"]["nve"]["ms_per_step"]
        axes[2].bar(label, timing, color=color, width=0.55)
        axes[2].text(label, timing, f"{timing:.1f}", ha="center", va="bottom")
    axes[0].axhline(summary["config"]["temperature"], color="0.7", ls=":", lw=1)
    axes[0].set(xlabel="Simulation time (ps)", ylabel="Temperature (K)", title="Temperature (dashed: NVT → NVE)")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="NVE time (ps)", ylabel="Δ total energy (meV/atom)", title="Energy conservation")
    axes[2].set(ylabel="Wall time / NVE step (ms)", title="Sustained MD cost")
    axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.15)
    fig.suptitle(f"{summary['waters']} waters · periodic box · Δt = {summary['config']['timestep_fs']} fs · CPU")
    fig.savefig(args.directory / "comparison.png", dpi=180)


if __name__ == "__main__":
    main()
