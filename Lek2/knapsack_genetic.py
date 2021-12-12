from knapsack_utils import *




def knapsack_greedy(items: List, max_weight: int) -> List:
    '''
    Sorted knapsack based on fraction value/weight, descending order.
    '''
    items = sorted(items, key = lambda x: x[0]/x[1], reverse = True)
    current_weight = 0
    current_value = 0
    return_items = []
    for value, weight in items:
        if current_weight + weight <= max_weight:
            current_weight += weight
            current_value += value
            return_items.append((value, weight, value/weight))
    print(f"Value: {current_value}")
    print(f"Weight: {current_weight}")
    return return_items


def knapsack_genetic(items: List, max_weight: int, mutation_rate: float = 0.1, 
                            fitness_func = fitness_function, nr_generations: int = 600,
                             population_size: int = 200, elitism: float = 0.02,
                              cross_over_rate: float = 0.85):
    '''
    Performs genetic algorithm approach for Knapsack Problem.
    '''
    #Initializa så att vi har slumpade, tillåtna individer
    population = initialize_population(items = items, max_weight = max_weight, population_size = population_size)
    population = sorted(population, key = lambda x: fitness_func(items, x, max_weight), reverse= True)


    for gen in range(nr_generations):
        population = selection(items = items, population = population, max_weight = max_weight, elitism=elitism)
        population.extend(cross_over(population))
        population[len(population)//2:] = mutate(population[len(population)//2:], generation=gen, mutation_reset= 60)
        population = sorted(population, key = lambda x: fitness_func(items, x, max_weight), reverse= True)

    for ind in range(5):
    #Samma individer - konvergerar mot samma lokala optimum. Ajaj! Får lösa med injection/diversity criteria
    #    print(np.linalg.norm(np.asarray(population[ind])- np.asarray(population[ind+1])))
        fitness_func(items, population[ind], max_weight = max_weight, verbose = True)

#Tuple generation
nr_of_elements = 100

upper_weight = 30
lower_weight = 10

upper_value = 200
lower_value = 20

max_weight = nr_of_elements*(upper_weight + lower_weight)/2

items = [(randint(lower_value,upper_value), randint(lower_weight,upper_weight)) for _ in range(nr_of_elements)]

knapsack_genetic(items = items, max_weight = max_weight , population_size = 600, elitism = 0.1, nr_generations=100, mutation_rate= 0.4)
knapsack_greedy(items, max_weight)




