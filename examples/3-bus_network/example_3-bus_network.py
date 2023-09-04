# -*- coding: utf-8 -*-
"""
Created on 29/8/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
    
Description:
    Exmple use of PyMGA to explore a network with 3 buses.
"""

import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D, near_optimal_space_matrix
import numpy as np
import yaml


#Setting __name__ to properly handle multiprocessing
if __name__ == '__main__':
    
    # Create or load network
    network = 'example_3-bus_network.nc'
    
    
    # Load options from configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
        
    # Set MAA variables to explore
    variables = {'x1': ['Generator', # Component type
                       ['wind'],     # Carrier(s)
                       'p_nom',],    # Component variable to explore
                 'x2': ['Generator',
                       ['coal'],
                       'p_nom',],
                 'x3': ['Generator',
                       ['solar'],
                       'p_nom',],
                 'x4': ['Store',
                       ['battery'],
                       'e_nom',],
                    } 
    


    #### PyMGA ####
    # PyMGA: Build case from PyPSA network
    case = PyMGA.cases.PyPSA_to_case(config, 
                                     network,
                                     variables = variables,
                                     mga_slack = 0.1,
                                     n_snapshots = 8760)
    
    # PyMGA: Choose MAA method
    method = PyMGA.methods.bMAA(case)
    
    # PyMGA: Solve optimal system
    opt_sol, obj, n_solved = method.find_optimum()

    # PyMGA: Search near-optimal space using chosen method
    verticies, directions, _, _ = method.search_directions(200, n_workers = 16)

    # PyMGA: Sample the identified near-optimal space
    MAA_samples = PyMGA.sampler.har_sample(1000_000, x0 = np.zeros(len(variables.keys())), 
                                            directions = directions, 
                                            verticies = verticies)


    #### Processing results ####
#
    # # Plot near-optimal space of Data and P2X
    all_variables    = ['Wind', 'Coal', 'PV', 'Battery'] #list(variables.keys())
    chosen_variables = ['Wind', 'Coal']
    # near_optimal_space_2D(all_variables, chosen_variables,
    #                       verticies, MAA_samples,
    #                       plot_MAA_points = True,)
    
    
    # Matrix plot of 2D "sides" of polytope, with histograms and correlations
    near_optimal_space_matrix(all_variables, verticies, MAA_samples,
                              opt_solution = opt_sol,
                              title = 'Near-optimal space')
