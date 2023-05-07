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

#%% PDE-FIND example: Burgers' equation

from scipy.io import loadmat
from PDE_FIND.PDE_FIND import build_linear_system, TrainSTRidge, print_pde

data = loadmat("./PDE_FIND/datasets/burgers.mat")

x = data['x']
t = data['t']

u = data['usol']
u = np.real(u)

x, t = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, t, u.T, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='coolwarm')
plt.title('u(x,t)', fontsize = 16)
plt.xlabel('x', fontsize = 16)
plt.ylabel('t', fontsize = 16)

u_t, theta, candidates = build_linear_system(u, dt = t[1,0]-t[0,0], dx = x[0,1]-x[0,0], D=3, P=3, time_diff = 'FD', space_diff = 'FD')
candidates[0] = '1'

w = TrainSTRidge(theta, u_t, lam=10**-5, d_tol=0.1)
print("PDE derived using STRidge")
print_pde(w, candidates)

#%% PDE-FIND example 2: Kuramoto-Sivashinksy equation

data = loadmat("./PDE_FIND/Datasets/kuramoto_sivishinky.mat")

x = data['x']
t = data['tt']

u = data['uu']

x, t = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, t, u.T, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='coolwarm')
plt.title('u(x,t)', fontsize = 16)
plt.xlabel('x', fontsize = 16)
plt.ylabel('t', fontsize = 16)

u_t, theta, candidates = build_linear_system(u, dt = t[1,0]-t[0,0], dx = x[0,1]-x[0,0], D=5, P=5, time_diff = 'FD', space_diff = 'FD')
candidates[0] = '1'

w = TrainSTRidge(theta, u_t, lam=10**-5, d_tol=5)
print("PDE derived using STRidge")
print_pde(w, candidates)





