#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Add path to graphviz executables

import os
os.environ["PATH"] += os.pathsep + '/Users/danielrinder/Miniconda3/bin/'

#%% Import libraries

from gplearn_PDE.genetic import SymbolicRegressor
from gplearn_PDE.functions import _function_map, _Function, sig1 as sigmoid, make_function
from gplearn_PDE._program import _Program

from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz

from scipy.io import loadmat

#%% Define derivative function

def diff(u,a):
    
    if (a[0,1] - a[0,0]) > (a[1,0] - a[0,0]):
        a = a[0,:]
        return np.gradient(u,a,a, edge_order=2)[1]
    else:
        a = a[:,0]
        return np.gradient(u,a,a, edge_order=2)[0]


#%% Generate ground truth data for the PDE: u_{t} = u_{x}

x = np.linspace(start=0, stop=1, num=100)
t = np.linspace(start=0, stop=1, num=100)

x_mesh,t_mesh = np.meshgrid(x,t)

u = np.exp(t_mesh)*np.exp(x_mesh) # anaytical solution

X = np.concatenate((x[:,np.newaxis],t[:,np.newaxis],u), axis=1)
u_t = diff(u,t_mesh)

#%% Fit symbolic regression model

rng = check_random_state(0)

est_gp = SymbolicRegressor(population_size = 5000, 
                           function_set = ['add','sub','mul','div','Diff','Diff2'], 
                           const_range=(-1.,1.),
                           generations = 20, stopping_criteria = 0.01,
                           p_crossover = 0.7, p_subtree_mutation = 0.2,
                           p_hoist_mutation = 0.05, p_point_mutation = 0,
                           max_samples = 0.9, verbose = 1,
                           parsimony_coefficient = 0.01, random_state = 0)

est_gp.fit(X, u_t)
print(est_gp._program)

#%% Visualize the final solution

dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.render('images/ex1_child', format='png', cleanup=True)
graph.render(view=True)

#%% Generate ground truth data for the PDE: u_{t} = u_{x} + u_{xx}

x = np.linspace(start=0, stop=1, num=100)
t = np.linspace(start=0, stop=1, num=100)

x_mesh,t_mesh = np.meshgrid(x,t)

u = np.exp(6*t_mesh)*(np.exp(-3*x_mesh)+np.exp(2*x_mesh)) # anaytical solution 

X = np.concatenate((x[:,np.newaxis],t[:,np.newaxis],u), axis=1)
u_t = diff(u,t_mesh)

#%% Fit symbolic regression model

rng = check_random_state(0)

est_gp = SymbolicRegressor(population_size = 5000, 
                           function_set = ['add','sub','mul','div','Diff','Diff2'], 
                           const_range=(-1.,1.),
                           generations = 20, stopping_criteria = 0.01,
                           p_crossover = 0.7, p_subtree_mutation = 0.2,
                           p_hoist_mutation = 0.05, p_point_mutation = 0,
                           max_samples = 0.9, verbose = 1,
                           parsimony_coefficient = 0.1, random_state = 0)

est_gp.fit(X, u_t)
print(est_gp._program)

#%% Visualize the final solution

dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.render('images/ex1_child', format='png', cleanup=True)
graph.render(view=True)

#%% Generate ground truth data for the PDE: u_{t} = u_{xx}

x = np.linspace(start=0, stop=1, num=100)
t = np.linspace(start=0, stop=1, num=100)

x_mesh,t_mesh = np.meshgrid(x,t)

u = np.exp(4*t_mesh)*(np.exp(2*x_mesh)+np.exp(-2*x_mesh)) # anaytical solution 

X = np.concatenate((x[:,np.newaxis],t[:,np.newaxis],u), axis=1)
u_t = diff(u,t_mesh)

#%% Fit symbolic regression model

rng = check_random_state(0)

est_gp = SymbolicRegressor(population_size = 5000, 
                           function_set = ['add','sub','mul','div','Diff','Diff2'], 
                           const_range=(-1.,1.),
                           generations = 10, stopping_criteria = 0.01,
                           p_crossover = 0.7, p_subtree_mutation = 0.2,
                           p_hoist_mutation = 0.05, p_point_mutation = 0,
                           max_samples = 0.9, verbose = 1,
                           parsimony_coefficient = 0.1, random_state = 0)

est_gp.fit(X, u_t)
print(est_gp._program)

#%% Visualize the final solution

dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.render('images/ex1_child', format='png', cleanup=True)
graph.render(view=True)

#%% burgers' equation

burgers = loadmat("./burgers.mat")

x = burgers['x']
t = burgers['t']

length = np.min([x.size, t.size])

x = x[0,0:length]
t = t[0:length,0]

x_mesh,t_mesh = np.meshgrid(x,t)

u = burgers['usol'][0:length,0:length]
u = np.real(u)
u = np.flip(u, axis=0)

ax = plt.figure().add_subplot(projection='3d')
surf = ax.plot_surface(x_mesh, t_mesh, u, rstride=1, cstride=1, color='green', alpha=0.5)
plt.show()

del ax, surf

#%%

function_set=('add', 'sub', 'mul', 'div','Diff', 'Diff2')

_function_set = []
for function in function_set:
    if isinstance(function, str):
        if function not in _function_map:
            raise ValueError('invalid function name %s found in '
                              '`function_set`.' % function)
        _function_set.append(_function_map[function])
    elif isinstance(function, _Function):
        _function_set.append(function)
    else:
        raise ValueError('invalid type %s found in `function_set`.'
                          % type(function))
function_set = _function_set

arities = {}
for function in _function_set:
    arity = function.arity
    arities[arity] = arities.get(arity, [])
    arities[arity].append(function)
    
init_depth=(2, 6)
init_method='half and half'
transformer=None
metric='mean absolute error'
parsimony_coefficient=0.001
p_point_replace=0.05
feature_names=None  
const_range = (-1.,1.) 

# x = np.linspace(start=0, stop=1, num=100)
# t = np.linspace(start=0, stop=10, num=100)

# x_mesh,t_mesh = np.meshgrid(x,t)

# u = x_mesh + t_mesh

# X = np.concatenate((x[:,np.newaxis],t[:,np.newaxis],u), axis=1)

n_samples = X.shape[0]
n_features = 3


program = _Program(function_set=function_set,
                    arities=arities,
                    init_depth=init_depth,
                    init_method=init_method,
                    n_features=n_features,
                    metric=metric,
                    transformer=transformer,
                    const_range=const_range,
                    p_point_replace=p_point_replace,
                    parsimony_coefficient=parsimony_coefficient,
                    feature_names=feature_names,
                    random_state=check_random_state(1),
                    program=None)

y_hat = program.validate_program()



#%% Visualize

dot_data = program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.render('images/ex1_child', format='png', cleanup=True)
graph.render(view=True)


