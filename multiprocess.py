import numpy as np
import multiprocessing as mp
import time

def mean_range_parallel(nr_values: int, q: mp.Queue) -> None:
    """Parallellvariant där vi stoppar in q och antal values. 
    Tidskomplexitet O(n^2), space complexity O(n). Returnar None då vi inte har något return value."""

    array = np.arange(nr_values) + np.random.normal(nr_values)
    for i in range(len(array)):
        value = np.mean(array[:i+1])
    q.put(value)

def mean_range(nr_values: int) -> int:
    """Singel/sekventiell variant, stoppar bara in antal values.
    Tidskomplexitet O(n^2), space complexity O(n)."""

    array = np.arange(nr_values) + np.random.normal(nr_values)
    for i in range(len(array)):
        value = np.mean(array[:i+1])
    return value


def main():
    """Main som beräknar tid för sekventiella och parallella operationer för en ökande input size."""
    #Hur många cores har du?
    n_cores = mp.cpu_count()
    print(f'Number of cores: {n_cores}')
    for k in range(3):
        length_of_array = 3*10**k
        print(f'Number of steps: {length_of_array }')
        input_values = n_cores*[length_of_array]
        #Med parallellisering
        start = time.time()
        #Sätt upp Queue
        qout = mp.Queue()
        #Skicka in  till vår parallelliserade funktion som stoppar våra processer i kön
        processes = [mp.Process(target=mean_range_parallel, args=(val, qout))
                    for val in input_values]
        #Starta varje process
        for p in processes:
            p.start()
        #Sammanställ processer
        for p in processes:
            p.join()
        #Plocka ut resultat
        result = [qout.get() for p in processes]
        #print(f'Results: {result}')
        stop = time.time()
        print(f'Time with parallelisation: {stop - start}')

        #Utan parallellisering:
        start = time.time()
        my_serial_results = [mean_range(val) for val in input_values]
        stop = time.time()
        #print(f'Results: {my_serial_results}')
        print(f'Time without parallelisation: {stop - start}')



if __name__ == "__main__":
    #mp freeze support needed for Windows.
    mp.freeze_support()
    main()