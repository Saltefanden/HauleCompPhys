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

def besselup_single(x: list[float], l: int) -> list[list[float]]:
    j: list[list[float]] = [[] for _ in range(l+1)]
    j[0] = [np.sin(_x)/_x if _x else 1. for _x in x]
    if l == 0:
        return j
    j[1] = [np.sin(_x)/_x**2 - np.cos(_x)/_x if _x else 0. for _x in x]
    if l<2:
        return j
    for i in range(2,l+1):
        j[i] = [(2*i - 1)/_x * j[i-1][idx] - j[i-2][idx] if _x else 0. for idx, _x in enumerate(x)]
    return j

def besselup_np(x: np.ndarray, l:int) -> np.ndarray:
    j = np.zeros((l+1, x.shape[0]))
    j[0] = np.sin(x)/x
    j[1] = np.sin(x)/x**2 - np.cos(x)/x
    for i in range(2, l+1):
        j[i] = (2*i-1)/x * j[i-1] - j[i-2]

    return j

def besselup_np_colmajor(x: np.ndarray, l:int) -> np.ndarray:
    j = np.zeros((x.shape[0],l+1), order="F")
    j[:,0] = np.sin(x)/x
    j[:,1] = np.sin(x)/x**2 - np.cos(x)/x
    for i in range(2, l+1):
        j[:,i] = (2*i-1)/x * j[:,i-1] - j[:,i-2]

    return j


def besseldown(x: float, l: int) -> list[float]:
    Lmax = l * 3*np.sqrt(l)
    return [0]

def test_besselup():
    l: int=6
    N: int=1_000
    ulim: float=50
    x = [i * ulim/N for i in range(N)]
    start = time.perf_counter()
    y = [besselup(i, l) for i in x]  
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for singleprocess {N=}")
    stime = end - start

    start = time.perf_counter()
    z = besselup_multi(x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for multiprocess {N=}")
    mtime = end-start

    start = time.perf_counter()
    _z = list(zip(*besselup_single(x, l)))
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for Singlepass {N=}")
    vtime = end-start

    _x = np.linspace(0, ulim, N)
    start = time.perf_counter()
    __z = besselup_np(_x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy {N=}")
    ntime = end-start
    
    start = time.perf_counter()
    ___z = besselup_np_colmajor(_x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy {N=}")
    nctime = end-start

    print(f"Multiprocessing is a factor of {stime/mtime} faster")
    print(f"Singlepass is a factor of {stime/vtime} faster")
    print(f"Numpy is a factor of {stime/ntime} faster")
    print(f"Numpy colmajor is a factor of {stime/nctime} faster")

    plt.plot(x,y, label="single")
    # plt.plot(x,z, linestyle="--", label="multi")
    plt.plot(x,_z, linestyle="--", label="multi")
    plt.plot(_x, __z.T, label="numpy")
    plt.plot(_x, ___z, label="numpy")
    plt.legend()
    plt.savefig("plot.png")

def main():
    test_besselup()

if __name__ == '__main__':
    main()
