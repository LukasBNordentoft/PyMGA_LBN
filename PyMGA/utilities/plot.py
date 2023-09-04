# -*- coding: utf-8 -*-
"""
Created on 31/08/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
"""

def set_options():
    import matplotlib.pyplot as plt
    import matplotlib
    color_bg      = "0.99"          #Choose background color
    color_gridaxe = "0.85"          #Choose grid and spine color
    rc = {"axes.edgecolor":color_gridaxe} 
    plt.style.use(('ggplot', rc))           #Set style with extra spines
    plt.rcParams['figure.dpi'] = 300        #Set resolution
    plt.rcParams['figure.figsize'] = [10, 5]
    matplotlib.rc('font', size=15)
    matplotlib.rc('axes', titlesize=20)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']     #Change font to Computer Modern Sans Serif
    plt.rcParams['axes.unicode_minus'] = False          #Re-enable minus signs on axes))
    plt.rcParams['axes.facecolor']= color_bg             #Set plot background color
    plt.rcParams.update({"axes.grid" : True, "grid.color": color_gridaxe}) #Set grid color
    plt.rcParams['axes.grid'] = True
    # plt.fontname = "Computer Modern Serif"
    
def near_optimal_space_2D(all_variables, chosen_variables,
                          verticies, samples,
                          bins = 50, ax = None,
                          linewidth = 2, linecolor = 'gray',
                          xlim = [None, None], ylim = [None, None],
                          plot_MAA_points = False, filename = None, show_text = True,
                          textcolor = 'black',
                          title = 'Near-optimal space',
                          ):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import ConvexHull
    import numpy as np
    import pandas as pd
    '''
    Plots 2D slice of a near-optimal space based on two chosen variables.
    
    Required packages: 
        matplotlib
        scipy
        numpy
        pandas
    
    Args:
        all_variables (list): List of variables explored
        chosen_variables(list): List of the two variables to plot
        verticies (array): Boundary points as n-dimensional numpy array of floats.
        samples (array): Samples as n-dimensional array of floats.
        bins (int): Number of bins to use for pcolormesh
    
    '''
    
    # Set plotting options
    set_options()
    
    # Dataframe with all verticies
    verticies_df = pd.DataFrame(verticies,
                                columns = all_variables)
    
    # Get verticies from only the chosen variables
    variable_verticies = verticies_df[chosen_variables]
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (10,8))
    
    if show_text:
        ax.set_xlabel(chosen_variables[0], fontsize = 24)
        ax.set_ylabel(chosen_variables[1], fontsize = 24)
        ax.set_title(title, color = textcolor)
    
    # Set x and y to be verticies for the first two variables
    x, y = variable_verticies[chosen_variables[0]], variable_verticies[chosen_variables[1]]
    
    samples_df = pd.DataFrame(samples,
                              columns = all_variables)
    
    # Set x and y as samples for this dimension
    x_samples = samples_df[chosen_variables[0]]
    y_samples = samples_df[chosen_variables[1]]
    
    # --------  Create 2D histogram --------------------
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                          bins = bins)
    
    # Create grid for pcolormesh
    x_grid, y_grid = np.meshgrid(xedges, yedges)
    
    # Create pcolormesh plot with square bins
    ax.pcolormesh(x_grid, y_grid, hist.T, cmap = 'Blues', 
                  zorder = 0)
    
    # Create patch to serve as hexbin label
    hb = mpatches.Patch(color = 'tab:blue')
    
    ax.grid('on')
    
    # --------  Plot hull --------------------
    hull = ConvexHull(variable_verticies.values)
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = ax.plot(variable_verticies.values[simplex, 0], variable_verticies.values[simplex, 1], 'k-', 
                color = linecolor, label = 'faces',
                linewidth = linewidth, zorder = 0)
    
    # list of legend handles and labels
    l_list, l_labels   = [l0, hb], ['Convex hull', 'Sample density']
    
    if plot_MAA_points:
        # Plot vertices from solutions
        l1, = ax.plot(x, y,
                  'o', label = "Near-optimal",
                  color = 'lightcoral', zorder = 2)
        l_list.append(l1)
        l_labels.append('MAA points')
        
    if show_text:
        ax.legend(l_list, l_labels, 
                  loc = 'center', ncol = len(l_list),
                  bbox_to_anchor = (0.5, -0.15), fancybox=False, shadow=False,)
    
    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
        
    plt.show()
        
    return ax 

def near_optimal_space_matrix(variables, verticies, 
                              samples,
                              bins = 50, ncols = 3,
                              title = 'Near-optimal space matrix', cmap = 'Blues',
                              xlim = [None, None], ylim = [None, None],
                              xlabel = None, ylabel = None,
                              opt_solution = None,
                              tech_titles = None, 
                              plot_MAA_points = False,
                              filename = None, show_cheb_radius = False,
                              show_minmax = False,
                              ):
    # Take a multi-dimensional MAA polyhedron, and plot each "side" in 2D.
    # Plot the polyhedron shape, samples within and correlations.
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    import numpy as np
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    
    pad = 5
    
    # colors = get_color_codes()
    
    var_titles = variables

    
    # -------- create correlation matrix --------------------------
    # Create dataframe from samples
    samples_df = pd.DataFrame(samples, columns = variables)
    
    # Calculate correlation and normalize
    samples_corr = samples_df.corr()
    
    # Calculate normalized correlation, used to color heatmap.
    samples_temp = samples_corr + abs(samples_corr.min().min())
    samples_norm = samples_temp / samples_temp.max().max()
    
    # -------- Set up plot ----------------------------------------
    set_options()
    
    text_lift = 1.075
    
    # define the endpoints of the colormap
    red    = (1.0, 0.7, 0.6)  # light red
    yellow = (1.0, 1.0, 0.8)  # light yellow
    green  = (0.6, 1.0, 0.6)  # light green
    
    # define the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [red, yellow, green])
    
    # Initialize and adjust figure
    plt.figure()
    fig, axs = plt.subplots(len(variables), len(variables), 
                            figsize = (20/3 * len(variables),
                                       5 * len(variables)))
    
    title_y = 0.94 if len(variables) == 4 else 0.96
    
    fig.suptitle(title, fontsize = 28, y = title_y)
    
    # figspace = 0.5 if len(techs) == 4 else 0.35
    fig.subplots_adjust(wspace = 0.25, hspace = 0.35)
    
    # Set top titles
    for ax, col in zip(axs[0], var_titles):
        ax.set_title(col + '\n')
    
    # Set side titles
    for ax, row in zip(axs[:,0], var_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords = ax.yaxis.label, textcoords='offset points',
                    size = 24, ha = 'right', va = 'center',
                    rotation = 90)
    
    # -------- Plotting -------------------------------
    
    # Upper triangle of subplots
    for i in range(0, len(variables)):
        for j in range(0, i):
            
            corr = samples_norm[variables[i]][variables[j]] # Is only used for coloring
            num  = samples_corr[variables[i]][variables[j]] # Is shown
            
            ax = axs[j][i]
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Write correlation
            corr_text = str(round(num,2))
            ax.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=20)
            
            ax.text(0.5, text_lift, 'Correlation', ha='center', va='top',
                    transform=ax.transAxes, fontsize = 16, color = 'gray')
            
            # Change bg color according to correlation
            ax.patch.set_facecolor(cmap(corr))
    
    
    # Diagonal plotting
    for j in range(0, len(variables)):

        ax = axs[j][j]
        
        sns.histplot(samples_df[variables[j]].values,
                     # color = colors[variables[j]],
                     bins = bins,
                     line_kws = {'linewidth':3},
                     element = 'bars',
                     stat = 'probability',
                     kde = True,
                     ax = ax, label = '_nolegend_',)
        
        if not xlabel == None or ylabel == None:
            ax.set_xlabel(xlabel, color = 'gray', size = 16)
            ax.set_ylabel(ylabel, color = 'gray', size = 16)
        else:
            ax.set_ylabel('Proportion', color = 'gray', size = 16)
            ax.set_xlabel('Capcity [GW]', color = 'gray', size = 16)
        
        ax.text(0.5, text_lift, 'Histogram', ha='center', va='top', 
                transform=ax.transAxes, fontsize = 16, color = 'gray')
        
        if not opt_solution == None:
            ax.axvline(x = opt_solution[j], 
                       color = 'gold', linestyle = '--',
                       linewidth = 4, gapcolor = 'darkorange',)
            
    
    
    # lower traingle of subplots
    for j in range(0, len(variables)):
        for i in range(0, j):
            
            ax = axs[j][i]
            
            if not xlabel == None or ylabel == None:
                ax.set_xlabel(xlabel, color = 'gray', size = 16)
                ax.set_ylabel(ylabel, color = 'gray', size = 16)
            else:
                ax.set_xlabel('Capacity [GW]', color = 'gray', size = 16)
                ax.set_ylabel('Capacity [GW]', color = 'gray', size = 16)
            
            ax.text(0.5, text_lift, 'Near-optimal space', ha='center', va='top',
                    transform=ax.transAxes, fontsize=16, color = 'gray')
            
            
            # MAA solutions
            x, y = verticies[:,i],   verticies[:,j]
            
            # Set x and y as samples for this dimension
            x_samples = samples[:,i]
            y_samples = samples[:,j]
            
            # --------  Create 2D histogram --------------------
            hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                                  bins = bins)
    
            # Create grid for pcolormesh
            X, Y = np.meshgrid(xedges, yedges)
            
            # Create pcolormesh plot with square bins
            ax.pcolormesh(X, Y, hist.T, cmap = 'Blues', zorder = 0)
            
            # Create patch to serve as hexbin label
            hb = mpatches.Patch(color = 'tab:blue')
            
            ax.grid('on')
            
            # --------  Plot hull --------------------
            hull = ConvexHull(verticies[:,[i,j]])
            
            # plot simplexes
            for simplex in hull.simplices:
                l0, = ax.plot(verticies[simplex, i], verticies[simplex, j], '-', 
                        color = 'silver', label = 'faces', zorder = 0)
            
            # list of legend handles and labels
            l_list, l_labels   = [l0, hb], ['Polyhedron face', 'Sample density']
            
            if plot_MAA_points:
                # Plot vertices from solutions
                l1, = ax.plot(x, y,
                          'o', label = "Near-optimal",
                          color = 'lightcoral', zorder = 2)
                l_list.append(l1)
                l_labels.append('MAA points')
                
                        
            # optimal solutions
            if not opt_solution == None:
                x_opt, y_opt = opt_solution[i],   opt_solution[j]
                
                # Plot optimal solutions
                ax.scatter(x_opt, y_opt,
                            marker = '*', 
                            s = 1000, zorder = 4,
                            linewidth = 2, alpha = 0.85,
                            facecolor = 'gold', edgecolor = 'darkorange',)
                
                l2 = Line2D([0], [0], marker = '*', color = 'gold',
                            markeredgecolor = 'darkorange', markeredgewidth = 2,
                            markersize = 25, label = 'Optimal Solutions',
                            linestyle = '',)
                l2_2 = Line2D([0], [0], linestyle = '--', color = 'gold',
                              gapcolor = 'darkorange', linewidth = 4,)
                
                l_list.append(l2)
                l_labels.append('Optimal solution')
                l_list.append(l2_2)
                l_labels.append('Optimal line')
                
            # Set limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
    
    # Place legend below subplots
    ax = axs[len(variables)-1, int(np.median([1,2,3]))-1] # Get center axis
    
    legend_right = 1.1 if len(variables) == 4 else 0.5
    
    ax.legend(l_list,
              l_labels, 
              loc = 'center', ncol = ncols,
              bbox_to_anchor=(legend_right, -0.12*len(variables)), fancybox=False, shadow=False,)
    
    # fig.suptitle(title, fontsize = 24, y = 0.96)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
        
    plt.show()
        
    return axs
