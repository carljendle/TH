from heapq import *
import numpy as np
import time

def main():
    """Exempelkod för pseudo-heapsort för varierande size."""
    nr_of_runs = 40
    nr_of_exponents = 8
    for i in range(nr_of_exponents):
        nr_of_values = 10**i
        total_time = 0
        for _ in range(nr_of_runs):
            cool_list = list(np.arange(nr_of_values) + np.random.normal(nr_of_values))
            start = time.time()
            #List -> heap, O(n)
            heapify(cool_list)
            # O(n*log(n)) då varje heappop-call svarar mot log(n)
            sorted_values = [heappop(cool_list) for _ in range(nr_of_values)]
            stop = time.time()
            total_time += (stop - start)
        print(f'Averaged time over {nr_of_runs} runs for input size {nr_of_values}: {total_time/nr_of_runs} seconds.')

if __name__ == "__main__":
    main()




