import numpy as np
import lib.besselcc as bs

N: int = 1_000
ulim: float = 50.
l: int = 50
z = np.zeros((l+1)*N)
bs.besselc(z, N, ulim, l)

# print(z)
# print(z[0])




