import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")

def besselup(x: np.ndarray, l: int) -> np.ndarray:
    j0 = np.sin(x)/x
    if l == 0:
        return j0
    j1 = np.sin(x)/x**2 - np.cos(x)/x

    for i in range(2, l+1):
        j0, j1 = j1, (2*i-1)/x * j1 - j0

    return np.nan_to_num(j1) 


def besseldown(x: np.ndarray, l: int) -> np.ndarray:
    lstart = l + int(np.sqrt(3)*l)
    print(lstart)
    j1 = np.zeros(x.shape)
    j0 = np.ones(x.shape)
    jl = np.sin(x)/x

    for i in range(lstart, 0, -1): 
        j0, j1 = (2*i + 1)/x * j0 - j1, j0
        if i == l:
            jl = j1
    
    true_j0 = np.sin(x)/x
    scalefactor = true_j0 / j0

    return np.nan_to_num(jl* scalefactor)

def bessel(x: np.ndarray, l: int) -> np.ndarray:
    pivot = np.searchsorted(x, l)
    xl, xu = x[:pivot], x[pivot:] 
    return np.concatenate((besseldown(xl, l), besselup(xu, l)))



x = np.linspace(0, 50, 100)
l = 15

z = besselup(x, l)
Z = besseldown(x, l)
Y = bessel(x, l)

plt.plot(x, z, label="up", linestyle = "--")
plt.plot(x, Z, label="down", linestyle = ':', color="r")
plt.plot(x, Y, label="Miller", color="k")
ymin, ymax = min(Y), max(Y)
plt.ylim([ymin, ymax])
plt.legend()
plt.savefig("res.png")
