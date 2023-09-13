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

import os
import sys

# Add parent folder to directory to load PyMGA package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D, near_optimal_space_matrix
import numpy as np
import yaml
import pandas as pd
import pypsa


if __name__ == '__main__':
    
    # Create or load network
    network = 'example_island_network.nc'
    
    # Define total island area
    total_area = 0.5*120_000 #[m^2]
    
    # Define area uses
    area_use = pd.Series( data = {'storage':  01.0,  #[m^2/MWh] Capacity
                                  'hydrogen': 02.1,  #[m^2/MW] capacity
                                  'data':     27.3,  #[m^2/MW] IT output
                                  })
    
    
    # Load options from configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
        
    # Set MAA variables to explore
    variables = {'x1': ['Generator',
                        ['P2X'],
                        'p_nom',],
                'x2': ['Generator',
                        ['Data'],
                        'p_nom',],
                'x3': ['Store',
                        ['Storage'],
                        'e_nom',]
                        } 
    
    
    # Define constraints to be passed to extra_functionalities in n.lopf()
    def extra_func(n, snapshots, mga_options):
        
        ### Define custom constraints
        def link_constraint(n):
            from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
            '''
            This function sets an upper limit for the sum of link capacities.
            '''
            
            # Chosen Link names
            link_names = ['Island_to_Denmark', 'Island_to_Norway', 'Island_to_Germany',
                          'Island_to_Netherlands', 'Island_to_Belgium',
                          'Island_to_United Kingdom']               
            
            # Get all link variables, and filter for only chosen links
            vars_links   = get_var(n, 'Link', 'p_nom')
            vars_links   = vars_links[link_names]
            
            # Set up left and right side of constraint
            rhs          = 3_000 # [MW] Maximum total link capacity
            lhs          = join_exprs(linexpr((1, vars_links))) #Sum of all link capacities
            
            # Define constraint and name it 'Sum constraint'
            define_constraints(n, lhs, '<=', rhs, 'Link', 'Sum constraint')
          
        
        def marry_links(n):
            from pypsa.linopt import get_var, linexpr, define_constraints
            '''
            Each country has a link to and from the island. This constraint
            ensures that these links behave as bidirectional links by 
            constraining the to always have the same capacity. 
            This is done for each country.
            '''
            
            # Get all link varuables
            vars_links   = get_var(n, 'Link', 'p_nom')
            
            # List of countries to which the island is connected.
            connected_countries =  [
                                    "Denmark",         
                                    "Norway",          
                                    "Germany",         
                                    "Netherlands",     
                                    "Belgium",         
                                    "United Kingdom"
                                    ]
            
            # loop through countries and define constraints
            for country in connected_countries:
                
                # Define left side as the capacity of the link from the island
                # minus the capacity of the link to the island.
                lhs = linexpr((1, vars_links['Island_to_' + country]),
                              (-1, vars_links[country + '_to_Island']))
                
                # Set up right side to be 0. forcing lnks to be equal
                rhs = 0
                
                # Set up constraint
                define_constraints(n, lhs, '=', rhs, 'Link', country + '_link_capacity_constraint')
        
        
        def area_constraint(n):
            from pypsa.linopt import get_var, linexpr, define_constraints
            '''
            This function constrains the area available to the technologies
            on the island. This is done by multiplying the area use [m^2/MW] 
            with the capacity for each technology taking up space on the island.
            '''
            
            # Get variables to include in constraint
            vars_gen   = get_var(n, 'Generator', 'p_nom')
            vars_store = get_var(n, 'Store', 'e_nom')
            
            # Apply area use on variable and create linear expression 
            lhs = linexpr(
                           (area_use['hydrogen'], vars_gen["P2X"]), 
                           (area_use['data'],     vars_gen["Data"]), 
                           (area_use['storage'],  vars_store['Storage'])
                          )
            
            # Define right side as total area
            rhs = total_area #[m^2]
            
            # Define constraint
            define_constraints(n, lhs, '<=', rhs, 'Island', 'Area_Use')
        
        # Define partial MGA constraints
        def local_mga_constraint(n, snapshots, mga_options):
            if mga_options is not None and mga_options['mga_slack'] is not None:
                from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
                
                # Get MGA slack
                epsilon = mga_options['mga_slack']
                
                # Get optimum system data for local 
                data_local = pd.read_pickle('n_opt_data_local.pkl')
            
                ## Create partial mga constraint for local demand
                # Loop through optimal system to find cost of local variables
                local_cost = 0
                for variable in data_local.columns:
                    
                    local_cost += (data_local[variable]['p_nom_opt'] * data_local[variable]['capital_cost']
                                    + data_local[variable]['p_sum'] * data_local[variable]['marginal_cost'])
                    
                # Get variables
                vars_gen     = get_var(n, 'Generator', 'p_nom')
                vars_gen_t   = get_var(n, 'Generator', 'p')
                vars_store   = get_var(n, 'Store', 'e_nom')
                vars_store_t = get_var(n, 'Store', 'e')
                
                # Right-hand-side: Upper limit of partial MGA constraint for local demand
                rhs = local_cost + abs(local_cost) * epsilon
                
                # Left-hand-side: Variables to limit
                lhs = ''
                for variable in data_local.columns:
                    if variable == 'Storage':
                        lhs += (linexpr( (data_local[variable]['capital_cost'], vars_store[variable]) ) 
                                + linexpr( (data_local[variable]['marginal_cost'], vars_store_t[variable]) ).sum())
                    else:
                        lhs += (linexpr( (data_local[variable]['capital_cost'], vars_gen[variable]) ) 
                                + linexpr( (data_local[variable]['marginal_cost'], vars_gen_t[variable]) ).sum())
                
                # lhs = (   linexpr( (data_local['Data']['capital_cost'],    vars_gen['Data']) ) 
                #           + linexpr( (data_local['Data']['marginal_cost'], vars_gen_t['Data']) ).sum()
                #         + linexpr( (data_local['P2X']['capital_cost'],     vars_gen['P2X']) ) 
                #           + linexpr( (data_local['P2X']['marginal_cost'], vars_gen_t['P2X']) ).sum()
                #         + linexpr( (data_local['Storage']['capital_cost'], vars_store['Storage']) ) 
                #           + linexpr( (data_local['Storage']['marginal_cost'], vars_store_t['Storage']) ).sum()
                #         )
                
                # Define partial mga constraint
                define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "partial_mga_constraint_local")
                
        def link_mga_constraint(n, snapshots, mga_options):
            if mga_options is not None and mga_options['mga_slack'] is not None:
                from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
                
                # Get MGA slack
                epsilon = mga_options['mga_slack']
                
                # # Get optimum system data
                data_links = pd.read_pickle('n_opt_data_link.pkl')
            
                ## Create partial mga constraint for local demand
                # Loop through optimal system to find cost of links.
                link_cost = 0
                for variable in data_links.columns:
                    link_cost += (data_links[variable]['p_nom_opt'] * data_links[variable]['capital_cost'])
                    
                # Get variables
                vars_links = get_var(n, 'Link', 'p_nom')
                
                # Right-hand-side: Upper limit of partial MGA constraint for links
                rhs = link_cost * (1 + epsilon)
                
                # Left-hand-side: Variables to limit
                lhs = ''
                for link in data_links.columns:
                    lhs += linexpr( (data_links[link]['capital_cost'], vars_links[link]) ) 
                
                # Define partial mga constraint
                define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "partial_mga_constraint_links")
        
        ### Call custom constraints 
        link_constraint(n)
        marry_links(n)
        area_constraint(n)
        
        ### Call partial mga constraints
        local_mga_constraint(n, snapshots, mga_options)
        link_mga_constraint(n, snapshots, mga_options)

    #### PyMGA ####
    # PyMGA: Build case from PyPSA network
    case = PyMGA.cases.PyPSA_to_case(config, 
                                      network,
                                      extra_func = extra_func,
                                      variables = variables,
                                      mga_slack = 0.1,
                                      n_snapshots = 8760)
    
    # PyMGA: Choose MAA method
    method = PyMGA.methods.MAA(case)
    
    # PyMGA: Solve optimal system
    opt_sol, obj, n_opt = method.find_optimum()
    
    # Export local data for partial mga constraint
    n_opt_data_local = pd.DataFrame( {
        'Data':[n_opt.generators.p_nom_opt['Data'], n_opt.generators_t.p['Data'].sum(),
                n_opt.generators.capital_cost['Data'], n_opt.generators.marginal_cost['Data']],
        'P2X':[n_opt.generators.p_nom_opt['P2X'], n_opt.generators_t.p['P2X'].sum(),
               n_opt.generators.capital_cost['P2X'], n_opt.generators.marginal_cost['P2X']],
        'Storage':[n_opt.stores.e_nom_opt['Storage'], n_opt.stores_t.e['Storage'].sum(),
                   n_opt.stores.capital_cost['Storage'], n_opt.stores.marginal_cost['Storage']],
        },
        index = ['p_nom_opt', 'p_sum', 'capital_cost', 'marginal_cost'])
    
    n_opt_data_local.to_pickle('n_opt_data_local.pkl')
    
    # Export link data for partial mga constraint
    n_opt_data_link = pd.DataFrame()
    link_names = n_opt.links[n_opt.links.index.str.startswith('Island')].index
    for link in link_names:
        row_data = pd.DataFrame({ link:[n_opt.links.p_nom_opt[link], n_opt.links.capital_cost[link]]}, index = ['p_nom_opt', 'capital_cost'])
        n_opt_data_link = pd.concat([n_opt_data_link, row_data], axis = 1)
        
    n_opt_data_link.to_pickle('n_opt_data_link.pkl')
    
    
    # PyMGA: Search near-optimal space using chosen method
    verticies, directions, _, _ = method.search_directions(14, n_workers = 16)

    # # PyMGA: Sample the identified near-optimal space
    har_samples = PyMGA.sampler.har_sample(1_000_000, x0 = np.zeros(len(variables.keys())), 
                                            directions = directions, 
                                            verticies = verticies)
    
    bayesian_samples = PyMGA.sampler.bayesian_sample(1_000_000, verticies)


    # #### Process results ####
    # Plot near-optimal space of Data (x1) and P2X (x2)
    all_variables    = list(variables.keys())
    # chosen_variables = ['x1', 'x2']
    # near_optimal_space_2D(all_variables, chosen_variables,
    #                       verticies, MAA_samples,
    #                       plot_MAA_points = True,
    #                       bins = 50)
    
    # # Matrix plot of 2D "sides" of polytope, with histograms and correlations
    near_optimal_space_matrix(all_variables, verticies, har_samples,
                              xlabel = 'Unit []', ylabel = 'Unit []',
                              opt_solution = opt_sol)
    
    near_optimal_space_matrix(all_variables, verticies, bayesian_samples,
                              xlabel = 'Unit []', ylabel = 'Unit []',
                              opt_solution = opt_sol)
