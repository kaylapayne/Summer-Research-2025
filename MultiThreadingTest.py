from Corrfunc.theory import DD
import numpy as np
import time

# Generate some random 3D points (e.g., 10000 points)
N = 1000000
np.random.seed(0)
X = np.random.uniform(0, 1000, N)
Y = np.random.uniform(0, 1000, N)
Z = np.random.uniform(0, 1000, N)

# Use a simple binfile for distances (in Mpc/h)
binfile = np.linspace(0, 50, 21).astype(np.float64)  # 20 bins from 0 to 50

def run_dd(nthreads):
    start = time.time()
    results = DD(autocorr=1, nthreads=nthreads, binfile=binfile,
                 X1=X, Y1=Y, Z1=Z, periodic=True, boxsize=1000.0)
    end = time.time()
    print(f"Threads: {nthreads}, Time: {end - start:.3f} seconds")
    return results

print("Testing Corrfunc DD with different thread counts...\n")

for nthreads in [4, 8, 10, 12, 16]:
    run_dd(nthreads)