#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate various defects using
pymatgen.analysis.defects.generators module
and
https://github.com/kaist-amsg/LS-CGCNN-ens/blob/master/FindSite.py
"""

from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
import copy

from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.defects.generators import InterstitialGenerator, VoronoiInterstitialGenerator

def alpha_shape_3D(pos, alpha,roundtol = 9):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
        roundtol - number of decimal for rounding 
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra_vertices = Delaunay(pos).vertices
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra_vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    # Remove bad tetrahedrals. These the ones where volume is zero.
    bad = a==0
    num = Dx**2+Dy**2+Dz**2-4*a*c
    bad[num<0] = True
    bad = np.where(bad)[0]
    tetra_vertices = np.delete(tetra_vertices,bad,axis=0)
    num = np.delete(num,bad,axis=0)
    a = np.delete(a,bad,axis=0)
    # get radius
    r = np.sqrt(num)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra_vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    if Triangles.size==0:
        return [], [], []
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles

atoms = read('/users/40247882/scratch2/validations/surfaces/surf-111.xyz') # (2x2) Pt(111) surface
#print(atoms)

structure = AseAtomsAdaptor.get_structure(atoms)
#print(structure)

vig = InterstitialGenerator(structure, 'O')
print(vig.unique_defect_seq)
exit()
structures = [v.generate_defect_structure() for v in vig]
frames = [AseAtomsAdaptor.get_atoms(s) for s in structures]

write('miaow.xyz', frames)


if __name__ == '__main__':
    pass