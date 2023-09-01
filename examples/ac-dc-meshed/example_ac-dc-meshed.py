# -*- coding: utf-8 -*-
"""
Created on 29/8/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
    
Description:
    Example case based on the North Sea Energy Island. Consists of an island 
    with wind, P2X and storage capacity, connected to several countries.
    This example include custom constraints being defined using extra_func.
"""

import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D
import numpy as np
import yaml
# from pypsa_netview.draw import draw_network


if __name__ == '__main__':
    
    # Create or load network
    network = 'example_ac-dc-meshed.nc'
    
    
    # Load options from configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
        
    # Set MAA variables to explore
    variables = {'x1': ['Generator',
                       ['wind'],
                       'p_nom',],
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
    method = PyMGA.methods.bMAA(case)
    
    # PyMGA: Solve optimal system
    opt_sol, obj, n_solved = method.find_optimum()
    
    # Draw optimal system (optional)
    # draw_network(n_solved, show_capacities = True)
    
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
