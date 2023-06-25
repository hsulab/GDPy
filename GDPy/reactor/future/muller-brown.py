#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
See https://demonstrations.wolfram.com/TrajectoriesOnTheMullerBrownPotentialEnergySurface/#more
K. Müller and L. D. Brown, 
"Location of Saddle Points and Minimum Energy Paths by a Constrained Simplex Optimization Procedure," 
Theoretical Chemistry Accounts, 53, 1979 pp. 75–93.

Wolfe-Quapp PES
"""

from termios import VEOL
import torch
import numpy as np

def pes_np(x, y):
    # parameters
    A =  [-200, -100, -170, 15]
    a =  [-1, -1, -6.5, 0.7]
    b =  [0, 0, 11, 0.6]
    c =  [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    energy = 0.0
    for k in range(4):
        energy += (
            A[k] * np.exp(a[k]*(x-x0[k])**2 + 
            b[k]*(x-x0[k])*(y-y0[k]) + 
            c[k]*(y-y0[k])**2)
        )

    return energy

def MullerBrownPES(pos):
    # unpack position and require gradients
    pos.requires_grad = True

    # parameters
    A =  torch.tensor([-200, -100, -170, 15])
    a =  torch.tensor([-1, -1, -6.5, 0.7])
    b =  torch.tensor([0, 0, 11, 0.6])
    c =  torch.tensor([-10, -10, -6.5, 0.7])
    x0 = torch.tensor([1, 0, -0.5, -1])
    y0 = torch.tensor([0, 0.5, 1.5, 1])

    x = pos[:, 0].reshape(-1,1)
    y = pos[:, 1].reshape(-1,1)

    dx = x - x0
    dy = y - y0

    energy = torch.sum(
        A * torch.exp(a*dx**2 + b*dx*dy + c*dy**2), 
        dim=-1
    )
    energy_list = [e for e in energy]
    forces = -torch.autograd.grad(energy_list, pos, retain_graph=False)[0]

    return energy, forces

if __name__ == '__main__':
    #pos = torch.tensor([0.5, 0.5], requires_grad=True)
    pos = torch.tensor(
        [
            [0.5, 0.5],
            [1.5, 1.5]
        ], 
        requires_grad=False
    )
    print(pos)
    energy, forces = MullerBrownPES(pos)
    print(energy)
    print(forces)

    # plot data
    import matplotlib as mpl
    mpl.use('Agg')  # silent mode
    from matplotlib import pyplot as plt
    plt.style.use('presentation')

    import matplotlib.cm as cm
    delta = 0.025
    x = np.arange(-1.6, 1.6, delta)
    y = np.arange(-0.8, 2.4, delta)
    X, Y = np.meshgrid(x, y)
    print(X.shape)
    print(Y.shape)

    positions = np.array([[xs, ys] for xs, ys in zip(X.flatten(),Y.flatten())])
    positions = torch.from_numpy(positions)
    Z, _ = MullerBrownPES(positions)
    Z = Z.detach().numpy()
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(16,12), constrained_layout=False)

    plt.suptitle("Muller-Brown PES")

    levels = np.arange(-160, 240, 40).tolist()
    levels.extend([320,400])
    cset1 = ax.contourf(X, Y, Z, levels)
    cset2 = plt.contour(X, Y, Z, cset1.levels, colors='k')

    ax.clabel(cset2, inline=True, fmt="%d")

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cset1)
    cbar.ax.set_ylabel("Potential Energy")
    # Add the contour line levels to the colorbar
    # cbar.add_lines(cset1)

    plt.tight_layout()
    plt.savefig("pes.png")