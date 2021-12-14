from random import randint
from typing import List
import numpy as np
import random


def fitness_function(x: np.array, lb: List = None, ub: List = None):
 
    '''
    '''
    min_x = -10
    max_x = 10
    min_y = -10
    max_y = 10

    x1, x2 = x
    boolean = True
    #boolean = True
    #if int(x1) < min_x | int(x1) > max_x | int(x2) < min_y | int(x2) > max_y:
    #    boolean = False

    return x1**2 -2*x1 +2 + x2**2 if boolean else 10**3


def cross_over(population: List[List],cross_over_rate: float = 0.85 ) -> List[List]:
    '''
    Returns single point cross over offspring from population.

    :param: population - list of lists of individuals
    :param: cross_over_rate - not implemented yet

    Returns: Crossovers, offspring from mated individuals
    '''
    random.shuffle(population)

    idx = -1
    crossovers = []
    max_cross_len = len(population[0])//2
    #print(population)
    for i in range(len(population)//2):
        cross_over_length = random.randint(0,max_cross_len)
        cross_over_point = random.randint(0, max_cross_len - cross_over_length)
        idx += 1
        ind1 = population[idx]
        idx += 1
        ind2 = population[idx]
        ind3 = [ind1[0], ind2[0]]
        ind4 = [ind2[0], ind1[0]]
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

def selection(population: List[np.array], fitness_func = fitness_function, elitism: float = 0.1):
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

    fitness_values = np.asarray([1/fitness_func(individual) for individual in remaining_pop])
    normalize_factor = sum(fitness_values)
    probability_scores = fitness_values/normalize_factor

    #Vi har redan valt elitism*len(population), vill halvera totala populationen
    nr_choices = int((0.5*len(population)) - elitist_choices)
    roulette_indices = np.random.choice(len(remaining_pop), nr_choices, replace=False, p=probability_scores)
    roulette_population = [remaining_pop[i] for i in roulette_indices]

    selected_pop.extend(roulette_population)

    return selected_pop





def initialize_population(lb: List, ub: List, population_size: int =200):
    '''
    Initializes a valid population (feasible solutions) based on random search.
    
    :param: population_size - number of individuals to return
    :param: lb - lower boundaries for feasible solution range
    :param: ub - upper boundaries for feasible solution range

    Returns: Population (list of feasible solutions (individuals))

    '''
    population = [[np.random.uniform(low, high) for low, high in zip(lb, ub)] for _ in range(population_size)]

    
    return population


def minimize_genetic(lb: List, ub: List, mutation_rate: float = 0.1, 
                            fitness_func = fitness_function, nr_generations: int = 600,
                             population_size: int = 200, elitism: float = 0.02,
                              cross_over_rate: float = 0.85):
    '''
    Performs genetic algorithm approach for Knapsack Problem.
    '''
    #Initializa så att vi har slumpade, tillåtna individer
    population = initialize_population(lb = lb, ub = ub, population_size = population_size)
    population = sorted(population, key = lambda x: 1/fitness_func(x), reverse= True)


    for gen in range(nr_generations):
        population = selection(population = population,fitness_func=fitness_function, elitism=elitism)
        population.extend(cross_over(population))
        population[len(population)//2:] = mutate(population[len(population)//2:], generation=gen, domain =(lb, ub), mutation_reset= 5)
        population = sorted(population, key = lambda x: 1/fitness_func(x), reverse= True)

    return population[0]

best_individual = minimize_genetic(lb = [-5, -5], ub = [5, 5], population_size=80, nr_generations=1000)

print(f"Best individual coordz: {best_individual}")
print(f"Analytical best coordz: {[1,0]}")
print(f"Best individual fitness score:{fitness_function(best_individual)}")
print(f"Analytical best score: {1}")
