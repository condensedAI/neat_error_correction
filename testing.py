from __future__ import division
import numpy as np
import math
import copy
import time
import sys
import os
import imp

import toric_lattice
import perfect_matching

home = os.environ['HOME']

def run2D(size=4,p=0.1):

   L=toric_lattice.PlanarLattice(size)

   L.applyRandomErrors(p,0)
   L.measureStars()
   L.measurePlaquettes()

   L.findAnyons()

   matchingX=perfect_matching.match_toric_2D(size,L.positions_anyons_P)
   matchingZ=perfect_matching.match_toric_2D(size,L.positions_anyons_S)

   L.apply_matching("X",matchingX)
   L.apply_matching("Z",matchingZ)

   return L.measure_logical()

curve = []
pXs = np.arange(0.01, 0.15, 0.01)
for p in pXs:
    print(p)
    count = 200
    for i in range(200):
        result = run2D(5,p)
        if (result[0][0] == -1) or (result[0][1] == -1):
            count -= 1
    curve.append( count/200 )
    print(count/200)

np.savetxt("mwpm-snapshot-d5-200.txt", np.column_stack([pXs,curve]))
