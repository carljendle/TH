import numpy as np
class Graph:
    """
    Graph object med adjacency matrix. Finns det nackdelar med en sådan implementation?
    Vi anger antal noder, vilka edges vi har samt vilken väg vi har gått.
    """
    def __init__(self, num_of_vertices):

        self.v = num_of_vertices
        self.edges = np.zeros((num_of_vertices,num_of_vertices))
        self.travelled_path = []

    def add_edge(self, u, v, weight):
        """
        Undirected edge u <-> v med weight.
        """
        self.edges[u][v] = weight
        self.edges[v][u] = weight

    def normalize(self):
        """
        Normalisera vikter i varje rad så att sannolihetsvektorn
        summerar till 1.
        """
        for row in self.edges:
            row /= sum(row)

    def traverse(self, vertex: int, nr_of_steps: int) -> None:
        """
        Metod som går längs grafen och sparar ner stegen som tagits.
        Vertex anger startpunkt, nr_of_steps anger hur många steg som ska tas.
        """
        for _ in range(nr_of_steps):
            self.travelled_path.append(vertex)
            #Slumpa ny nod med sannolikhetsfördelning given av radvektor
            vertex = np.random.choice([x for x in range(self.v)], p = self.edges[vertex])






g = Graph(9)


g.add_edge(0, 1, 4)
g.add_edge(0, 6, 7)
g.add_edge(1, 6, 11)
g.add_edge(1, 7, 20)
g.add_edge(1, 2, 9)
g.add_edge(2, 3, 6)
g.add_edge(2, 4, 2)
g.add_edge(3, 4, 10)
g.add_edge(3, 5, 5)
g.add_edge(4, 5, 15)
g.add_edge(4, 7, 1)
g.add_edge(4, 8, 5)
g.add_edge(5, 8, 12)
g.add_edge(6, 7, 1)
g.add_edge(7, 8, 3)
g.normalize()
g.traverse(vertex = 2, nr_of_steps = 100)
print(f'Travelled path: {g.travelled_path}')
