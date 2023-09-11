# -*- coding: utf-8 -*-
"""
Created on 29/8/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
    
Description:
    Exmple use of PyMGA to explore a network with 3 buses.
"""

import os
import sys

# Add parent folder to directory to load PyMGA package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

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
    verticies, directions, _, _ = method.search_directions(14, n_workers = 16)
    
    
#%%
    # PyMGA: Sample the identified near-optimal space
    # Hit-and-run sampler, valid for all dimensions
    har_samples = PyMGA.sampler.har_sample(1000_000, x0 = np.zeros(len(variables.keys())), 
                                            directions = directions, 
                                            verticies = verticies)
    # Bayesian bootstrap sampler, good up to aorund 8 dimensions
    bayesian_samples = PyMGA.sampler.bayesian_sample(1_000_000, verticies) 


    #### Processing results ####

    # # Plot near-optimal space of Data and P2X
    all_variables    = ['Wind', 'Coal', 
                        'PV', 
                        'Battery'
                        ] #list(variables.keys())
    chosen_variables = ['Wind', 'Coal']

    
    # Matrix plot of 2D "sides" of polytope, with histograms and correlations
    # Plot Hit-and-Run samples
    near_optimal_space_matrix(all_variables, verticies, har_samples,
                              opt_solution = opt_sol,
                              title = 'Near-optimal space')
    
    # Plot Bayesian Bootstrap samples
    near_optimal_space_matrix(all_variables, verticies, bayesian_samples,
                              opt_solution = opt_sol,
                              title = 'Near-optimal space')
    
    #%%

import numpy as np

# Define your convex polytope by specifying its vertices and associated directions.
# vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
# directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

vertices = verticies

# Define the number of samples you want to generate.
num_samples = 1000

# Initialize an empty list to store the generated samples.
samples = []

# Perform hit-and-run sampling
for _ in range(num_samples):
    # Choose a random vertex as the starting point
    start_vertex = np.random.choice(len(vertices))
    current_point = vertices[start_vertex]

    while True:
        # Select a random direction
        direction_index = np.random.choice(len(directions))
        direction = directions[direction_index]

        # Calculate the distance to the nearest boundary
        lambdas = []
        for i in range(len(vertices)):
            vertex = vertices[i]
            edge = vertices[(i + 1) % len(vertices)] - vertex

            # Calculate the intersection point
            numerator = np.dot((vertex - current_point), direction)
            denominator = np.dot(edge, direction)

            if denominator == 0:
                lambda_i = float('inf')
            else:
                lambda_i = numerator / denominator

            if lambda_i > 0:
                lambdas.append(lambda_i)

        if not lambdas:
            break

        # Move along the selected direction by the minimum lambda
        min_lambda = min(lambdas)
        current_point += min_lambda * direction

    # Append the current sample point to the list
    samples.append(current_point)
    print('Sample obtained')

# Convert the list of samples to a numpy array
samples = np.array(samples)

# The 'samples' numpy array now contains randomly generated points within the convex polytope in N-dimensional space.


#%%
near_optimal_space_matrix(all_variables, verticies, samples,
                          opt_solution = opt_sol,
                          title = 'Near-optimal space')
