from random import randint
from typing import List
import numpy as np
import random



def fitness_function(items: List, individual: List, max_weight: int, verbose: bool = False):
    '''
    Evaluates individual for Knapsack problem, returns fitness score for individual.

    :param: items - list of tuples containing value/weight pairs
    :param: individual - binary list of same length as items, marking the chosen items
    :param: max_weight - maximum allowed weight

    Returns: Score equal to sum of values if Knapsack is not busted, else 0.
    '''
    weight = 0
    value = 0
    for item, mask in zip(items, individual):
        value += item[0]*mask
        weight += item[1]*mask
    if verbose:
        print(f"Weight: {weight}")
        print(f"Value: {value}")
    return value if weight <= max_weight else 0


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
    for i in range(len(population)//2):
        cross_over_length = random.randint(0,max_cross_len -1)
        cross_over_point = random.randint(0, max_cross_len - cross_over_length)
        idx += 1
        ind1 = population[idx]
        idx += 1
        ind2 = population[idx]

        ind3 = ind1[0:cross_over_point] + ind2[cross_over_point:cross_over_point+cross_over_length] + ind1[cross_over_point+cross_over_length:]
        ind4 = ind2[0:cross_over_point] + ind1[cross_over_point:cross_over_point+cross_over_length] + ind2[cross_over_point+cross_over_length:]
        crossovers.append(ind3)
        crossovers.append(ind4)
    return crossovers



def mutate(population: List[List],generation: int,mutation_reset: None, mutation_rate: float = 0.15,decay_scale: float = 0.1, domain: tuple = (1, 0)):
    '''
    Mutate individuals for local search step in GA

    :param: popuation - list of individuals
    :param: generation - iteration number, used for mutation rate decay
    :param: mutation_reset - determines mutation rate decay, no decay implemented if None
    :param: mutation_rate - base rate of bitwise flip mutation for individuals
    :param: decay_scale - scale of decay for mutation rate
    :param: domain - target range for mutation

    Returns: List of bitwise flipped, mutated individuals

    '''
    if mutation_reset:
        cycle_factor = generation % mutation_reset
        mutation_rate *= 2**(-decay_scale*cycle_factor)

    for i, individual in enumerate(population):
        for k in range(len(individual)):
            p = random.random()
            if p < mutation_rate:
                individual[k] = domain[individual[k]]
        population[i] = individual

            

    return population

def selection(items: List, population: List[List], max_weight: int, fitness_func = fitness_function, elitism: float = 0.1):
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

    fitness_values = np.asarray([fitness_func(items, individual, max_weight) for individual in remaining_pop])
    normalize_factor = sum(fitness_values)
    probability_scores = fitness_values/normalize_factor

    #Vi har redan valt elitism*len(population), vill halvera totala populationen
    nr_choices = int((0.5*len(population)) - elitist_choices)
    roulette_indices = np.random.choice(len(remaining_pop), nr_choices, replace=False, p=probability_scores)
    roulette_population = [remaining_pop[i] for i in roulette_indices]

    selected_pop.extend(roulette_population)

    return selected_pop





def initialize_population(items: List, max_weight: int, population_size: int =200):
    '''
    Initializes a valid population (feasible solutions) based on random search.
    
    :param: items - List of value/weight tuples
    :param: max_weight - max weight criteria for Knapsack Problem
    :param: population_size - number of individuals to return

    Returns: Population (list of feasible solutions (individuals))

    '''
    population = []
    for i in range(population_size):
        weight = 0
        individual = [0]*len(items)
        choices = np.arange(len(items))
        random.shuffle(choices)
        for idx in choices:
            #Vikt för item idx + nuvarande vikt.
            if items[idx][1] + weight <= max_weight:
                weight += items[idx][1]
                individual[idx] = 1
        population.append(individual)
    
    return population