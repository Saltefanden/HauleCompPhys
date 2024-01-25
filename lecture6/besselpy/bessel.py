import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool

def besselup(x: float, l: int) -> list[float]:
    j: list[float] = [0 for _ in range(l+1)]
    if x == 0: 
        j[0] = 1
        return j

    j[0] = np.sin(x)/x if x else 1

    if l == 0:
        return j

    j[1] = np.sin(x)/x**2 - np.cos(x)/x

    if l<2:
        return j

    for i in range(2,l+1):
        j[i] = (2*i - 1)/x * j[i-1] - j[i-2]

    return j


def besselup_multi(x: list[float], l: int) -> list[list[float]]:
    with Pool(processes=8) as pool: 
        y = pool.map(partial(besselup, l=l), x)
    return y


def besseldown(x: float, l: int) -> list[float]:
    Lmax = l * 3*np.sqrt(l)
    
     
    return [0]

def test_besselup():
    l: int=6
    N: int=1_000
    ulim: float=50
    start = time.perf_counter()
    x = [i * ulim/N for i in range(N)]
    y = [besselup(i, l) for i in x]  
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for singleprocess {N=}")
    stime = end - start
    start = time.perf_counter()
    z = besselup_multi(x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for multiprocess {N=}")
    mtime = end-start
    print(f"Multiprocessing is a factor of {stime/mtime} faster")

    plt.plot(x,y, label="single")
    plt.plot(x,z, linestyle="--", label="multi")
    plt.legend()
    plt.savefig("plot.png")

def main():
    test_besselup()

if __name__ == '__main__':
    main()
