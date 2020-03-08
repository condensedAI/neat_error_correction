import random
import math
import copy
import itertools
import numpy as np


""" 2D version of the Toric Lattice

Input Parameters:
-----------------
size --> dimension of the lattice

Data Array:
----------

STAR -- Q -- STAR -- Q -- STAR -- Q --
|            |            |
|            |            |
Q    PLAQ    Q    PLAQ    Q    PLAQ
|            |            |
|            |            |
PLAQ -- Q -- STAR -- Q -- STAR -- Q --
|            |            |
|            |            |
Q    PLAQ    Q    PLAQ    Q    PLAQ
|            |            |
|            |            |
PLAQ -- Q -- STAR -- Q -- STAR -- Q --
|            |            |
|            |            |

PLAQ and STAR positions store the value of the most recent
stabilizer measurement. They can take a value of 0 or 1.

Q(ubit) positions store the state of a qubit, a list: [x,z] x,z in {1,-1}

"""

class PlanarLattice:
    """2D version of the toric lattice"""

    def __init__(self,size):
        self.size=size

        # Number of qubits
        self.N_Q=2*size*size
        # Number of plaquettes
        self.N_P=size*size

        # Define basic qubit and stabilizer positions
        self.positions_Q=[(x,y) for x in range(0,2*size) for y in range(((x+1)%2),2*size,2) ]
        self.positions_S=[(x,y) for x in range(0,2*size,2) for y in range(0,2*size,2)]
        self.positions_P=[(x,y) for x in range(1,2*size,2) for y in range(1,2*size,2)]

        # Defining list of positions for two interspersed round of plaquette measurement
        # This is to simulate true measurements that determine the anyons
        self.positions_P1=[(x,y) for x in range(1,2*size,2) for y in range(x%4,2*size,4)]
        self.positions_P2=[(x,y) for x in range(1,2*size,2) for y in range((x+2)%4,2*size,4)]
        self.positions_S1=[(x,y)  for x in range(0,2*size,2) for y in range(x%4,2*size,4)]
        self.positions_S2=[(x,y)  for x in range(0,2*size,2) for y in range((x+2)%4,2*size,4)]

        # Locations of the syndrome
        self.positions_anyons_P=None
        self.positions_anyons_S=None

        ## Initialise array
        self.array=[[0 for x in range(2*self.size)] for _ in range(2*self.size)]

        # Set all the qubits to +1 in Z and +1 in X
        for i in range(self.N_Q):
            self.array[self.positions_Q[i][0]][self.positions_Q[i][1]]=[1,1]

        # Set all the plaquettes and stars to +1 (alternatively we could do a
        # round of perfect measurements)
        for i in range(self.N_P):
            self.array[self.positions_P[i][0]][self.positions_P[i][1]]=1
            self.array[self.positions_S[i][0]][self.positions_S[i][1]]=1

    def applyRandomErrors(self,pX,pZ):
        """ Applies random X and Z errors with the given probabilites to all qubits.

        The qubit values in self.array are updated.
        """
        for q0,q1 in self.positions_Q:
            rand1=np.random.rand()
            #rand2=np.random.rand()

            if rand1<pX:
                self.array[q0][q1][0]*=-1
            #if rand2<pZ:
            #    self.array[q0][q1][1]*=-1

    def measurePlaquettes(self):
        """ Calculates the value of each plaquette stabilizer
        """

        m=2*self.size

        # Reset the list of measured plaquettes
        self.plaq = []
        for p0,p1 in self.positions_P:

            # Get list of all the qubits that surroind this plaquette
            stabQubits=((p0,(p1-1)%m),(p0,(p1+1)%m),((p0-1)%m,p1),((p0+1)%m,p1))

            # The value of this plaquette is just the product of the z values
            stab=1
            for s0,s1 in stabQubits:
                stab*=self.array[s0][s1][0]

            # Update the list of plaquette values
            self.plaq.append(stab)

            # Update the array
            self.array[p0][p1]=stab


    def measureStars(self):
        """ calculates the value of each star stabilizer
        """

        m=2*self.size

        # Reset the list of measured stars
        self.star = []
        for p0,p1 in self.positions_S:
            stabQubits=((p0,(p1-1)%m),(p0,(p1+1)%m),((p0-1)%m,p1),((p0+1)%m,p1))

            # The value of this star is just the product of the x values
            stab=1
            for s0,s1 in stabQubits:
                stab*=self.array[s0][s1][1]

            self.star.append(stab)
            self.array[p0][p1]=stab

    def measure_logical(self):
        """ measures the logical state of the array

        Assumes:
        -------
        That the array is in the code space. That is, if all stabilizers
        were to be measured they should all return +1.

        Returns:
        -------
        A list of the form [[x1,z1],[x2,z2]] where all values in {1,-1}
        giving the logical state of the x and z components of the two
        encoded qubits.

        """

        logical_x=[1,1]
        logical_z=[1,1]
        positions_z1=[[1,x] for x in range(0,2*self.size,2)]
        positions_z2=[[y,1] for y in range(0,2*self.size,2)]
        positions_x1=[[0,x] for x in range(1,2*self.size,2)]
        positions_x2=[[y,0] for y in range(1,2*self.size,2)]

        for pos in positions_z1:
            logical_z[0]*=self.array[pos[0]][pos[1]][1]
        for pos in positions_z2:
            logical_z[1]*=self.array[pos[0]][pos[1]][1]
        for pos in positions_x1:
            logical_x[0]*=self.array[pos[0]][pos[1]][0]
        for pos in positions_x2:
            logical_x[1]*=self.array[pos[0]][pos[1]][0]

        return [logical_x,logical_z]

    def findAnyons(self):
        """ Locates all the '-1' stabilizer outcomes in the 2D array

        Returns:
        -------
        No return value. The list of anyon positions is stored in the
        variable self.positions_anyons_P(S).

        """
        anyon_positions_S=[]
        anyon_positions_P=[]

        for i in range(self.N_P):
            if self.star[i]==-1:
                anyon_positions_S+=[self.positions_S[i]]
            if self.plaq[i]==-1:
                anyon_positions_P+=[self.positions_P[i]]

        self.positions_anyons_P=anyon_positions_P
        self.positions_anyons_S=anyon_positions_S

    def apply_matching(self,error_type,matching):
        """ For correction of a 2D array. Pauli X or Z flips applied to return state to the codespace according to the given matching.

        Params:
        ------
        error_type --> which channel should the matching be applied to, X or Z
        matching --> list of pairs of anyon positions
        """

        if error_type in ["X","x",0]: channel=0
        elif error_type in ["Z","z",1]: channel =1
        else:
            raise ValueError('valid error types are "X" or "Z"')

        flips=[]
        for pair in matching:

            [p0,p1]=pair[0]
            [q0,q1]=pair[1]


            m=2*self.size

            d0=(q0-p0)%m
            d1=(q1-p1)%m


            if d0 < m-d0:
                end0=q0
                for x in range(1,d0,2):
                    flips+=[[(p0+x)%m,p1]]
            else:
                end0=p0
                for x in range(1,m-d0,2):
                    flips+=[[(q0+x)%m,q1]]

            if d1 < m-d1:
                for y in range(1,d1,2):
                    flips+=[[end0,(p1+y)%m]]
            else:
                for y in range(1,m-d1,2):
                    flips+=[[end0,(q1+y)%m]]


        for flip in flips:
            self.array[flip[0]][flip[1]][channel]*=-1


    def apply_flip_array(self,channel,flip_array):

        if channel in ["X","x",0]: c=0
        elif channel in ["Z","z",1]: c=1
        else:
            raise ValueError('channel must be X or Z')

        for (x0,x1) in self.positions_Q:
            self.array[x0][x1][c]*=flip_array[x0][x1]
