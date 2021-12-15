from sko.PSO import PSO
from typing import List
import numpy as np


def max_function(x):
    x1, x2 = x

    return -(2**(-(x1**2 + x2**2)))

#Demo för maximeringsproblem
pso = PSO(func=max_function, dim=2, pop=40, max_iter=150, lb=[-3, -3], ub=[3, 3], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('Best coords max_problem:: ', pso.gbest_x, 'Best value max_problem:', -pso.gbest_y)


def min_function(x):
    '''
    '''

    x1, x2 = x

    return x1**2 -2*x1 +2 + x2**2

pso = PSO(func=min_function, dim=2, pop=40, max_iter=150, lb=[-3, -3], ub=[3, 3], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('Best coords min problem: ', pso.gbest_x, 'Best value min problem:', pso.gbest_y)


def pso(function, n_particles:int, dimensions: int, lb: List, ub: List, min = True, max_iter:int = 100, w: float = 0.8, c1: float = 0.5, c2: float = 0.5):

    '''
    :param: function - n-dimensional objective function to minimize or maximize
    :param: n_particles - number of particles in the particle swarm
    :param: dimensions - dimensionality of objective function
    :param: lb - list of lower boundaries for each dimension
    :param: ub - list of upper boundaries for each dimension
    :param: min - minimization or maximization criteria, defaults to minimize
    :param: max_iter - maximum number of iterations
    :param: w - inertia coefficient for all particles \in [0,1]
    :param: c1 - cognitive coefficient for all particles \in [0,1]
    :param: c2 - swarm coefficient for all particles  \in [0,1]

    Returns value and coordinates for best optimum found
    '''
    #Optimera inte något som ej optimeras bör.
    assert len(lb) == len(ub) == dimensions, 'Non-matching dimensions.'

    #Sätt upp startpositioner och hastigheter för partiklar
    start_positions  = [np.asarray([np.random.uniform(low, high) for low, high in zip(lb, ub)]) for _ in range(n_particles)]
    start_velocities = [np.asarray([np.random.uniform(low, high) for low, high in zip(lb, ub)]) for _ in range(n_particles)]
    #Skapa lista med partiklars position, hastighet, bästa egna position och bästa value
    particles = [{"Position": pos, "Velocity": vel, "p_best": pos, "val_best": function(pos)} for pos, vel in zip(start_positions, start_velocities)]

    #Hitta globalt optima - värde och position
    global_best_pos = None
    if min:
        global_best_val = np.inf
        for particle in particles:
            if particle["val_best"] < global_best_val:
                global_best_val = particle["val_best"]
                global_best_pos = particle["p_best"]
    else:
        global_best_val = -np.inf
        for particle in particles:
            if particle["val_best"] > global_best_val:
                global_best_val = particle["val_best"]
                global_best_pos = particle["p_best"]

    for _ in range(max_iter):
        for particle in particles:
            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)
            particle["Velocity"] *= w
            particle["Velocity"] += c1*r1*(particle["p_best"] - particle["Position"]) + c2*r2*(global_best_pos - particle["Position"])
            particle["Position"] += particle["Velocity"]

            #Uppdatera partikelns egna optimum
            if function(particle["Position"]) < particle["val_best"]:
                particle["val_best"] = function(particle["Position"])
                particle["p_best"] = particle["Position"]
            #Uppdatera globalt optimum 
            if particle["val_best"] < global_best_val:
                global_best_val = particle["val_best"]
                global_best_pos = particle["p_best"]
    return global_best_pos, global_best_val
    

gbest_pos, gbest_val = pso(function = min_function, n_particles = 20, dimensions = 2, lb =  [-5,-5], ub = [5, 5],\
     w = 0.3, max_iter = 150, c1 = 0.1, c2 = 0.2)

print(f"Global best position:{gbest_pos}")
print(f"Global best value: {gbest_val}")