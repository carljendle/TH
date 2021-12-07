import json
import numpy as np
from typing import List
import time
import sys
import random
import matplotlib.pyplot as plt

with open('flights_data.json') as json_file:
    data = json.load(json_file)


flights_tmp = data["Flights"]
flights = {}
for k,v in flights_tmp.items():
    key = tuple(k.split('-'))
    flights[key] = v

people = data["People"]

destination = data['Destination']



def print_schedule(schedule: List) -> None:
    '''
    '''
    flight_id = -1
    total_price = 0
    for i in range(len(schedule) // 2):
        name = people[i][0]
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][schedule[flight_id]]
        total_price += going[2]
        flight_id += 1
        returning = flights[(destination, origin)][schedule[flight_id]]
        total_price += returning[2]
        print('%10s%10s %5s-%5s U$%3s %5s-%5s U$%3s' % (name, origin, going[0], 
                                                        going[1], going[2], 
                                                        returning[0], returning[1], returning[2]))   
    print('Total price: ', total_price)


schedule = [1,2, 3,2, 7,3, 6,3, 2,4, 5,3]
print_schedule(schedule)



def get_minutes(hour: str):
    '''
    String to int representation for flight time.
    '''
    t = time.strptime(hour, '%H:%M')
    minutes = t[3] * 60 + t[4]

    return minutes

print(get_minutes('6:13'))
print(get_minutes('23:59'))
print(get_minutes('00:00'))




def fitness_function(solution):
    total_price = 0
    last_arrival = 0#Sämsta tänkbara värde
    first_departure = 1439

    flight_id = -1
    for i in range(len(solution) // 2):
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][solution[flight_id]]
        flight_id += 1
        returning = flights[(destination, origin)][solution[flight_id]]

        total_price += going[2]
        total_price += returning[2]

        if last_arrival < get_minutes(going[1]):
            last_arrival = get_minutes(going[1])
        if first_departure > get_minutes(returning[0]):
            first_departure = get_minutes(returning[0])




    total_wait = 0
    flight_id = -1
    for i in range(len(solution) // 2):
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][solution[flight_id]]
        flight_id += 1
        returning = flights[(destination, origin)][solution[flight_id]]

        total_wait += last_arrival - get_minutes(going[1])
        total_wait += get_minutes(returning[0]) - first_departure
  
    if last_arrival > first_departure:
        total_price += 50
  
    return total_price + total_wait



schedule = [1,4, 3,2, 7,3, 6,3, 2,4, 5,3]
print(fitness_function(schedule))

  
domain = [(0,9)] * (len(people) * 2)

def hill_climb(domain, fitness_function, initial = []):
    count = 0
    #Ifall vi har en genererad lösning från random search som är bättre än genomsnittet:
    if len(initial) > 0:
        solution = initial
    else:
        solution = [random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]

    while True:
        #Sätter upp grannar
        neighbors = []
        for i in range(len(domain)):
            if solution[i] > domain[i][0]:
                #Om det är maxvärde lägger vi inte till detta
                if solution[i] != domain[i][1]:
                    neighbors.append(solution[0:i] + [solution[i] + 1] + solution[i + 1:])
            if solution[i] < domain[i][1]:
                #Om det är minvärde lägger vi inte till detta
                if solution[i] != domain[i][0]:
                    neighbors.append(solution[0:i] + [solution[i] - 1] + solution[i + 1:])


        actual = fitness_function(solution)
        best = actual
        #Utvärdera värde i varje granne
        #print(len(neighbors))
        for i in range(len(neighbors)):
            count += 1
            cost = fitness_function(neighbors[i])
            if cost < best:
                best = cost
                solution = neighbors[i]
        #Bryt så fort vi inte ser en förbättring för någon granne. Vi har nått minima (sannolikt lokalt).
        if best == actual:
            break

    return solution

n_trials = 50

fitness_values = [fitness_function(hill_climb(domain, fitness_function)) for _ in range(n_trials)]

print(fitness_values)

plt.plot(np.arange(n_trials), fitness_values)
plt.xlabel("Run nr:")
plt.ylabel("Fitness Value")
plt.title("Hill Climb")
print(f"Minimum value: {min(fitness_values)}")
plt.show()
