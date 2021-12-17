from random import randint
from typing import List
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def fitness_function(x: np.array, x_vals: np.array, y_vals:np.array):
 

    a, b, c, d = x

    pred = a*x_vals**3 +b*x_vals**2 + c*x_vals + d

    #Synonymt med (y[0]-pred[0])**2 + (y[1] - pred[1])**2 +...
    return np.linalg.norm(y_vals - pred)**2


def cross_over(population: List[np.array]) -> List[np.array]:
    '''
    Returns single point cross over offspring from population.

    :param: population - list of np.arrays representing individuals

    Returns: Crossovers, offspring from mated individuals
    '''
    random.shuffle(population)

    idx = -1
    crossovers = []
    max_cross_len = 3
    #print(population)
    for i in range(len(population)//2):
        cross_over_length = random.randint(0,max_cross_len)
        cross_over_point = random.randint(0, max_cross_len - cross_over_length)
        idx += 1
        ind1 = population[idx]
        idx += 1
        ind2 = population[idx]
        ind3 = np.concatenate((ind1[0:cross_over_point], ind2[cross_over_point:]))
        ind4 = np.concatenate((ind2[0:cross_over_point], ind1[cross_over_point:]))
        crossovers.append(ind3)
        crossovers.append(ind4)
    return crossovers



def mutate(population: List[List],generation: int, domain: List, mutation_reset: None, mutation_rate: float = 0.15, decay_scale: float = 0.1):
    '''
    Mutate individuals for local search step in GA

    :param: popuation - list of individuals
    :param: generation - iteration number, used for mutation rate decay
    :param: mutation_reset - determines mutation rate decay, no decay implemented if None
    :param: mutation_rate - base rate of bitwise flip mutation for individuals
    :param: domain - target range for mutation

    Returns: List of bitwise flipped, mutated individuals

    '''
    if mutation_reset:
        factor = generation % mutation_reset
        mutation_rate *= 2**(-0.1*factor)

    for i, individual in enumerate(population):
        for k in range(len(individual)):
            p = random.random()
            if p < mutation_rate:
                step_length = 2**(-0.1*factor)*np.random.normal()
                individual[k] = individual[k] + step_length
        population[i] = individual


            

    return population

def selection(population: List[np.array],x_vals: np.array, y_vals: np.array, fitness_func = fitness_function, elitism: float = 0.1):
    '''
    Performs elitism and roulette wheel selection on individuals sorted fitness function.
    Selects half of the total number of individuals.
    TODO - Implementation of diversity selection/injection

    :param: items - List of possible value/weight tuples
    :param: population - List of individuals
    :param: max_weight - maximum allowed weight for fitness function
    :param: fitness_func - fitness function used for Knapsack Problem
    :param: elitism - Fraction of best individuals to guarantee survival for next generation

    Returns: Half of the number of original individuals based on roulette wheel selection and elitism.
    '''

    elitist_choices = int(elitism*len(population))
    selected_pop = population[:elitist_choices]
    remaining_pop = population[elitist_choices:]
    assert len(selected_pop) + len(remaining_pop) == len(population)

    fitness_values = np.asarray([1/fitness_func(individual,x_vals, y_vals) for individual in remaining_pop])
    normalize_factor = sum(fitness_values)
    probability_scores = fitness_values/normalize_factor

    #Vi har redan valt elitism*len(population), vill halvera totala populationen
    nr_choices = int((0.5*len(population)) - elitist_choices)
    roulette_indices = np.random.choice(len(remaining_pop), nr_choices, replace=False, p=probability_scores)
    roulette_population = [remaining_pop[i] for i in roulette_indices]

    selected_pop.extend(roulette_population)

    return selected_pop





def initialize_population(lb: np.array, ub: np.array, population_size: int =200):
    '''
    Initializes a valid population (feasible solutions) based on random search.
    
    :param: population_size - number of individuals to return
    :param: lb - lower boundaries for feasible solution range
    :param: ub - upper boundaries for feasible solution range

    Returns: Population (list of feasible solutions (individuals))

    '''
    population = [np.asarray([np.random.uniform(low, high) for low, high in zip(lb, ub)]) for _ in range(population_size)]

    
    return population


def func_approx_genetic(lb: List, ub: List, x_vals: np.array, y_vals: np.array, mutation_rate: float = 0.1, 
                            fitness_func = fitness_function, nr_generations: int = 600,
                             population_size: int = 200, elitism: float = 0.02,
                              cross_over_rate: float = 0.85):
    '''
    Performs genetic algorithm approach for polynomial approximation.
    '''
    #Initializa så att vi har slumpade, tillåtna individer
    population = initialize_population(lb = lb, ub = ub, population_size = population_size)
    population = sorted(population, key = lambda x: 1/fitness_func(x,x_vals, y_vals), reverse= True)

    for gen in range(nr_generations):
        population = selection(population = population,x_vals = x_vals, y_vals = y_vals, fitness_func=fitness_function, elitism=elitism)
        population.extend(cross_over(population))
        population[len(population)//2:] = mutate(population[len(population)//2:], generation=gen, domain =(lb, ub), mutation_reset= 5)
        population = sorted(population, key = lambda x: 1/fitness_func(x,x_vals, y_vals), reverse= True)

    return population[0]



df = pd.read_csv("polynomial_data.csv")

x_vals = np.asarray(df['X'])
y_vals = np.asarray(df['Y'])


a, b, c, d = func_approx_genetic(lb = [-10]*4, ub = [10]*4,x_vals = x_vals, y_vals = y_vals, population_size=80, nr_generations=5000)

pred_vals = np.asarray([[a*x**3 + b*x**2 + c*x + d] for x in x_vals])

print(f"Coefficients: {a, b, c, d}")
print(f"MSE: {mean_squared_error(y_vals, pred_vals)}")
print(f"MAE: {mean_absolute_error(y_vals, pred_vals)}")
plt.plot(x_vals, y_vals, label = "True values")
plt.plot(x_vals, pred_vals, label = "Predicted values")
plt.title("MSE: " + str(mean_squared_error(y_vals, pred_vals)) + " MAE: " + str(mean_absolute_error(y_vals, pred_vals)))

plt.legend()

plt.show()