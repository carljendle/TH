from random import randint
from typing import List

#Tuple generation
nr_of_elements = 200
items = [(randint(20,200), randint(10,30)) for _ in range(nr_of_elements)]

def random_knapsack(items: List, max_weight: int):
    '''
    Random Knapsack. Nothing to see.
    '''
    current_weight = 0
    current_value = 0
    return_items = []
    for value, weight in items:
        if current_weight + weight <= max_weight:
            current_weight += weight
            current_value += value
            return_items.append((value, weight, value/weight))
    print(f"Items selected: {return_items}")
    print(f"Value: {current_value}")
    print(f"Weight: {current_weight}")

def knapsack_sorted(items: List, max_weight: int, ascending: bool = False):
    '''
    Sorted knapsack, ascending or descending based on weight.
    '''
    if ascending:
        items = sorted(items, key = lambda x: x[1])
    else:
        items = sorted(items, key = lambda x: x[1], reverse = True)

    current_weight = 0
    current_value = 0
    return_items = []
    for value, weight in items:
        if current_weight + weight <= max_weight:
            current_weight += weight
            current_value += value
            return_items.append((value, weight, value/weight))
    print(f"Items selected: {return_items}")
    print(f"Value: {current_value}")
    print(f"Weight: {current_weight}")
    

def knapsack_fraction(items: List, max_weight: int) -> List:
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
    print(f"Items selected: {return_items}")
    print(f"Value: {current_value}")
    print(f"Weight: {current_weight}")

random_knapsack(items, 50)
knapsack_sorted(items, 50)
knapsack_sorted(items, 50, ascending=True)
knapsack_fraction(items, 50)


