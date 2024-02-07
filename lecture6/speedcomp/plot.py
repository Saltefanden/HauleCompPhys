import numpy as np
import matplotlib.pyplot as plt
from speedcomp import besselup
import time


f =  open("./res.dat", "r")
z = [[float(element) for element in line.split()] for line in f.readlines()]
z = list(zip(*z))

l: int=50
N: int=1000
ulim: float=50
x = [i * ulim/N for i in range(N)]
start = time.perf_counter()
y = [besselup(i, l) for i in x]
y = list(zip(*y))
end = time.perf_counter()
print(f"Elapsed time {end - start} for singleprocess {N=}")
stime = end - start

def plotnbessel(n):
    plt.plot(z[n])
    plt.plot(y[n])
    plt.savefig(f"cplot{n}.png")
    plt.clf()


for i in range(5):
    plotnbessel(i)

