from heapq import *
from typing import List


class Graph:
    """
    Grafklass som implementerar Dijkstra's för undirected, weighted graph.
    Inspirerad av https://stackabuse.com/dijkstras-algorithm-in-python/, dock med förändringar i form av:
    - Adjacency matrix ersatt av adjacency dict
    - Kartläggning av faktisk path, inte bara summerade weights
    - Borttrimmad redundancy
    - min heap som priority queue för att det låter fräckare
    """

    def __init__(self, num_of_vertices: int):
        """
        v - antal noder i grafen.
        edges - edges som sammankopplar noder med weight.
        visited - håller koll på vilka noder vi redan besökt så att vi inte lägger till dessa element i priority queuen.
        """
        self.v = num_of_vertices
        self.edges = {v:{} for v in range(self.v)}
        self.visited = {}


    def add_edge(self, u: int, v: int, weight: int):
        """
        Metod för att lägga till undirected edges i grafen.
        """
        self.edges[u][v] = weight
        self.edges[v][u] = weight

    def traverse(self, start_vertex: int, stop_vertex: int, path: List) -> List[int]:
        """
        One-liner practice för att få den faktiska pathen via rekursion.
        Om vi inte har nått hela vägen från vår slutnod till startnod kallar vi på traverse igen med vår nästa nod och lägger till nuvarande nod i vår path.
        Avbryter då vi nått startnoden och ger tillbaks den omvända ordningen inklusive startnod för att få vägen från start till slut.
        """

        return [*self.traverse(start_vertex, self.visited[stop_vertex], path), stop_vertex] if stop_vertex != start_node else list(reversed([*path,start_node]))


    def dijkstra(self, start_vertex: int, stop_vertex: int):
        """
        Dijkstra's ALgorithm för att hitta minsta avstånd från start_vertex till
        stop_vertex. Returnerar bäst observerade värden hittills för grafen.
        Kan givetvis göras snyggare med lite vilja och vaselin. Snabbare? Kanske, kanske inte.
        """
        D = {v:float('inf') for v in range(self.v)}
        D[start_vertex] = 0

        current_vertex = previous_vertex = start_vertex
        #Vi kör med heap för att det är coolare
        pq = []
        heappush(pq,(0, start_vertex, previous_vertex))
        #Vi fortsätter tills vi har besökt (inte bara observerat(!!!)) vårt mål
        while stop_vertex not in self.visited:
            #Vi loopar igenom vår heap tills vi når en nod vi inte besökt tidigare istället för att återbesöka
            while current_vertex in self.visited:
                try:
                    (dist, current_vertex, previous_vertex) = heappop(pq)
                except:
                    #Når vi slutet på listan har vi utvärderat alla vägar en gång och då bryter vi
                    return D, self.visited
            self.visited[current_vertex] = previous_vertex
            
            #Loopar inte igenom en hel adjacency matrix, utan kollar bara i de grannar som current_vertex faktiskt har
            for neighbor in self.edges[current_vertex]:
                distance = self.edges[current_vertex][neighbor]
                #Snabbare med dictionary än med linjärsökning som är i andra förslaget
                if neighbor not in self.visited:
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost < old_cost:
                        #Lägger till current_vertex också för att ha koll på vägen som har tagits
                        heappush(pq,(new_cost, neighbor, current_vertex))
                        D[neighbor] = new_cost
        return D, self.visited 


g = Graph(9)
inputs = [(1, 0, 4), (0, 6, 7),(1, 6, 11),(1, 7, 20),(1, 2, 9),(2, 3, 6),(2, 4, 2),(3, 4, 10),(3, 5, 5),
(4, 5, 15),(4, 7, 1),(4, 8, 5),(5, 8, 12),(6, 7, 1),(7, 8, 3)]
for inp in inputs:
    g.add_edge(*inp)


start_node = 0
stop_node = 5

D, visited = g.dijkstra(start_node,stop_node)

cool = g.traverse(0,5, [])
print(cool)
