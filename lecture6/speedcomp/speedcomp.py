import os
from cmodule.lib.besselcc import besselc
import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit

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
    lx = len(x)
    j: list[list[float]] = [[0.]*lx for _ in range(l+1)]
    j[0] = [np.sin(_x)/_x if _x else 1. for _x in x]
    if l == 0:
        return j
    j[1] = [np.sin(_x)/_x**2 - np.cos(_x)/_x if _x else 0. for _x in x]
    if l<2:
        return j
    for i in range(2,l+1):
        j[i] = [(2*i - 1)/_x * j[i-1][idx] - j[i-2][idx] if _x else 0. for idx, _x in enumerate(x)]
    return j

def besselup_single_colmajor(x: list[float], l: int) -> list[list[float]]:
    lx = l+1
    j: list[list[float]] = [[0.]*lx for _ in range(len(x))]
    for idx,_x in enumerate(x): 
        if not _x:
            j[idx][0] = 1
            if lx == 1:
                continue
            j[idx][1] = 0
            if lx == 2:
                continue
            for i in range(2, lx):
                j[idx][i] = 0
            continue

        j[idx][0] = np.sin(_x) / _x 
        if lx == 1:
            return j
        j[idx][1] = np.sin(_x)/_x**2 - np.cos(_x)/_x 
        if lx == 2:
            return j 
        for i in range(2, lx):
            j[idx][i] = (2*i - 1)/_x * j[idx][i-1] - j[idx][i-2] 
        
    return j


def besselup_np(x: np.ndarray, l:int) -> np.ndarray:
    j = np.zeros((l+1, x.shape[0]), order="C")
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

@njit
def besselup_np_jit(x: np.ndarray, l:int, j:np.ndarray) -> np.ndarray:
    j[0] = np.sin(x)/x
    j[1] = np.sin(x)/x**2 - np.cos(x)/x
    for i in range(2, l+1):
        j[i] = (2*i-1)/x * j[i-1] - j[i-2]

    return j

@njit
def besselup_np_colmajorjit(x: np.ndarray, l:int, j: np.ndarray) -> np.ndarray:
    j[:,0] = np.sin(x)/x
    j[:,1] = np.sin(x)/x**2 - np.cos(x)/x
    for i in range(2, l+1):
        j[:,i] = (2*i-1)/x * j[:,i-1] - j[:,i-2]

    return j

def time_numpycol():
    l: int=50
    N: int=10000000
    ulim: float=50
    _x = np.linspace(0, ulim, N)
    start = time.perf_counter()
    z = besselup_np_colmajor(_x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy colmajor {N=}")

def time_besselc():
    l: int=50
    N: int=10000000
    ulim: float=50
    z = np.zeros(N*(l+1))
    if not "OMP_NUM_THREADS" in os.environ:
        os.environ["OMP_NUM_THREADS"] = "6"
    start = time.perf_counter()
    besselc(z, N, ulim, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for c extension {N=}", end= " ")
    print(f"With OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")


def test_besselup():
    np.seterr(divide = 'ignore', invalid="ignore")
    l: int=50
    N: int=1000000
    ulim: float=50
    x = [i * ulim/N for i in range(N)]
    start = time.perf_counter()
    y = [besselup(i, l) for i in x]  
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for singleprocess {N=}")
    stime = end - start

    # start = time.perf_counter()
    # z = besselup_multi(x, l)
    # end = time.perf_counter()
    # print(f"Elapsed time {end - start} for multiprocess {N=}")
    mtime = end-start

    start = time.perf_counter()
    # z = list(zip(*besselup_single(x, l)))
    z = besselup_single(x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for Singlepass {N=}")
    vtime = end-start

    start = time.perf_counter()
    # z = list(zip(*besselup_single(x, l)))
    z = besselup_single_colmajor(x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for Singlepasscolmajor {N=}")
    vcoltime = end-start

    _x = np.linspace(0, ulim, N)
    start = time.perf_counter()
    z = besselup_np(_x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy {N=}")
    ntime = end-start
    
    start = time.perf_counter()
    z = besselup_np_colmajor(_x, l)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy colmajor {N=}")
    nctime = end-start

    start = time.perf_counter()
    # To avoid cheating by caching the result itself use a different value
    # j = np.zeros((l+2, _x.shape[0]), order="C")
    # z = besselup_np_jit(_x, l+1, j)
    j = np.zeros((l+1, _x.shape[0]), order="C")
    z = besselup_np_jit(_x, l, j)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy jit {N=}")
    njittime = end-start

    start = time.perf_counter()
    z = besselup_np_jit(_x, l, j)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy jitsecond pass {N=}")
    njittime2 = end-start

    start = time.perf_counter()
    j = np.zeros((_x.shape[0],l+1), order="F")
    z = besselup_np_colmajorjit(_x, l, j)
    # To avoid cheating by caching the result itself use a different value
    # j = np.zeros((_x.shape[0],l+2), order="F")
    # z = besselup_np_colmajorjit(_x, l+1, j)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy colmajor jit {N=}")
    ncjittime = end-start

    start = time.perf_counter()
    z = besselup_np_colmajorjit(_x, l, j)
    end = time.perf_counter()
    print(f"Elapsed time {end - start} for numpy colmajor jitsecond pass {N=}")
    ncjittime2 = end-start

    z = np.zeros(N*(l+1))
    os.environ["OMP_NUM_THREADS"] = "1" 
    start = time.perf_counter()
    besselc(z, N, ulim, l)
    end = time.perf_counter()
    comp1 = end-start
    print(f"Elapsed time {end - start} for c extension {N=}", end= " ")
    print(f"With OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")

    z = np.zeros(N*(l+1))
    os.environ["OMP_NUM_THREADS"] = "12"
    start = time.perf_counter()
    besselc(z, N, ulim, l)
    end = time.perf_counter()
    comp12 = end-start
    print(f"Elapsed time {end - start} for c extension {N=}", end= " ")
    print(f"With OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")

    print(f"Multiprocessing is a factor of {stime/mtime} faster")
    print(f"Singlepass is a factor of {stime/vtime} faster")
    print(f"Singlepass is a factor of {stime/vcoltime} faster")
    print(f"Numpy is a factor of {stime/ntime} faster")
    print(f"Numpy colmajor is a factor of {stime/nctime} faster")
    print(f"Numpy jit is a factor of {stime/njittime} faster")
    print(f"Numpy colmajorjit is a factor of {stime/ncjittime} faster")
    print(f"2nd passNumpy jit is a factor of {stime/njittime2} faster")
    print(f"2nd passNumpy colmajorjit is a factor of {stime/ncjittime2} faster")
    print(f"C extension is a factor of {stime/comp1} faster")
    print(f"C extension with 12 threads is a factor of {stime/comp12} faster")


def main():
    # time_besselc()
    # time_numpycol()
    # time_numpycol()
    # time_besselc()
    # time_besselc()
    test_besselup()

if __name__ == '__main__':
    main()
