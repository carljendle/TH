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
    Tar in lista av ID för flights för varje person:
    [flight ID tur person 0, flight ID retur person 0..]
    Exempel - [1,2, 3,2, 7,3, 6,3, 2,4, 5,3]
    '''
    flight_id = -1
    total_price = 0
    for i in range(len(schedule) // 2):
        #Hanterar en person i taget

        name = people[i][0]
        origin = people[i][1]
        flight_id += 1
        #Tur
        going = flights[(origin, destination)][schedule[flight_id]]
        total_price += going[2]
        flight_id += 1
        #Retur, samma person
        returning = flights[(destination, origin)][schedule[flight_id]]
        total_price += returning[2]
        print('%10s%10s %5s-%5s U$%3s %5s-%5s U$%3s' % (name, origin, going[0], 
                                                        going[1], going[2], 
                                                        returning[0], returning[1], returning[2]))   
    print('Total price: ', total_price)

#Exempel på flights: [tur person 0, retur person 0, tur person 1, retur person 1...]
schedule = [1,2, 3,2, 7,3, 6,3, 2,4, 5,3]
print_schedule(schedule)



def get_minutes(hour: str):
    '''
    String to int representation for flight time.
    '''
    t = time.strptime(hour, '%H:%M')
    minutes = t[3] * 60 + t[4]

    return minutes

#print(get_minutes('6:13'))
#print(get_minutes('23:59'))
#print(get_minutes('00:00'))




def fitness_function(solution):
    total_price = 0
    last_arrival = 0
    first_departure = 0#Sämsta tänkbara värde

    flight_id = -1
    for i in range(len(solution) // 2):
        origin = people[i][1]
        flight_id += 1
        going = flights[(origin, destination)][solution[flight_id]]
        flight_id += 1
        returning = flights[(destination, origin)][solution[flight_id]]

        total_price += going[2]
        total_price += returning[2]

        #Integervärde för ankomst, tur
        if last_arrival < get_minutes(going[1]):
            last_arrival = get_minutes(going[1])
        #Integervärde för avgång, retur
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

        #Hur länge måste personen vänta vid tur respektive retur?
        total_wait += last_arrival - get_minutes(going[1])
        #Vi har koll på first departure, ser hur stor summan utav alla personers väntetid på flygplatsen är.
        #Klockan slutar ticka för varje person så fort de gått ombord planet.
        total_wait += get_minutes(returning[0]) - first_departure
  
    #Balansen mellan total_price och total_wait kan modifieras här - vi skulle till exempel kunna returnera
    # a*total_price**b + c*total_wait**d för något a,b,c,d för att fästa större vikt vid respektive del
    return total_price + total_wait



schedule = [1,4, 3,2, 7,3, 6,3, 2,4, 5,3]
print(fitness_function(schedule))

  



def random_search(domain, fitness_function):
    #Börja med att sätta best_cost till sämsta tänkbara:
    best_cost = sys.maxsize
    #Slumpa massa lösningar och utvärdera dem en och en
    for i in range(1000):
        solution = [random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        cost = fitness_function(solution)
    if cost < best_cost:
        best_cost = cost
        best_solution = solution
    return best_solution

#Domain - det här är vårt sökområde. För varje person vet vi start och destination. Vi måste också kunna ha tillräckligt många
#element för att täcka varje persons tur- och returresa.
#Vi hade kunnat sätta ett generellt max/min-värde för varje domain - dock hade det inte fungerat ifall
#vi hade haft olika antal möjliga flights för varje person.
domain = [(0,9)] * (len(people)*2)

n_trials = 50

fitness_values = [fitness_function(random_search(domain, fitness_function)) for _ in range(n_trials)]

print(fitness_values)

plt.plot(np.arange(n_trials), fitness_values)
plt.xlabel("Run nr:")
plt.ylabel("Fitness Value")
plt.title("Random Search")
print(f"Minimum value: {min(fitness_values)}")
plt.show()
