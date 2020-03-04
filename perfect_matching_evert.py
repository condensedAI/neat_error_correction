import random
import os
import csv
import subprocess
import time
import copy

import blossom5.pyMatch as pm

## Functions to perform minimum weight matching on the toric and planar topological codes
## There are 4 variants of this here to carry out the matching in the 2D and 3D (imperfect
## measurement) cases, and for the toric and planar codes. Each takes a list of anyon_positions,
## constructs the corresponding graph problem, and interfaces with the Blossom V algorithm (Kologomorov)
## to perform minimum weight matching.

def match_toric_2D(lattice_size,anyon_positions):

    """ Uses perfect matching to return a matching to fix the 2D TORIC code from the given positions of '-1' stabilizer outcomes in the code.

    Assumptions:
    -----------
    Perfect measurement, meaning there must be an even number of anyons.

    Parameters:
    ----------
    lattice_size -- size of the code.
    anyon_positions -- List of all '-1' stabilizer outcomes in the code. E.g. [[x0,y0],[x1,y1],..].

    Returns:
    -------
    The perfect matching, a list of paired anyon positions.

    """

    nodes_list=anyon_positions
    n_nodes=len(nodes_list)

    if n_nodes==0:
        return []

    m=2*lattice_size

    graphArray=[]

    ##fully connect the nodes within the 2D layer
    ## node numbering starts at 0

    for i in range(n_nodes-1):
        (p0,p1)=nodes_list[i]

        for j in range(n_nodes-i-1):
            (q0,q1)=nodes_list[j+i+1]

            w0=(p0-q0)%m
            w1=(p1-q1)%m
            weight=min([w0,m-w0])+min([w1,m-w1])

            graphArray+=[[i,j+i+1,weight]]

    n_edges=len(graphArray)

    ## PERFORM MATCHING
    ## Use the blossom5 perfect matching algorithm to return a matching

    matching=pm.getMatching(n_nodes,graphArray)


    ## REFORMAT MATCHING
    matching_pairs=[[i,matching[i]] for i in range(n_nodes) if matching[i]>i]
    points=[] if len(matching_pairs)==0 else [[nodes_list[i] for i in x] for x in matching_pairs]

    return points
