# -*- coding: utf-8 -*-
"""
Created on 29/8/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
    
Description:
    Example case based on the pypsa example "Meshed AC-DC" from:
    https://pypsa.readthedocs.io/en/latest/examples/ac-dc-lopf.html
"""
# Add parent folder to directory to load PyMGA package
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D
import numpy as np
import yaml


#Setting __name__ to properly handle multiprocessing
if __name__ == '__main__':
    
    # Create or load network
    network = 'example_ac-dc-meshed.nc'
    
    
    # Load options from configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
        
    # Set MAA variables to explore
    variables = {'x1': ['Generator', # Component type
                       ['wind'],     # Carrier(s)
                       'p_nom',],    # Component variable to explore
                 'x2': ['Generator',
                       ['gas'],
                       'p_nom',]
                    } 
    

    #### PyMGA ####
    # PyMGA: Build case from PyPSA network
    case = PyMGA.cases.PyPSA_to_case(config, 
                                     network,
                                     variables = variables,
                                     mga_slack = 0.1,
                                     n_snapshots = 10)
    
    # PyMGA: Choose MAA method
    method = PyMGA.methods.MAA(case)
    
    # PyMGA: Solve optimal system
    opt_sol, obj, n_solved = method.find_optimum()
    
    # PyMGA: Search near-optimal space using chosen method
    verticies, directions, _, _ = method.search_directions(14, n_workers = 16)

    # PyMGA: Sample the identified near-optimal space
    MAA_samples = PyMGA.sampler.har_sample(100_000, x0 = np.zeros(len(variables.keys())), 
                                           directions = directions, 
                                           verticies = verticies)


    #### Processing results ####
    # Plot near-optimal space of Data and P2X
    all_variables    = list(variables.keys())
    chosen_variables = ['x1', 'x2']
    near_optimal_space_2D(all_variables, chosen_variables,
                          verticies, MAA_samples,
                          plot_MAA_points = True,)
