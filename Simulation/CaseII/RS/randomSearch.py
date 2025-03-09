""" 
    Generating 5000 random samples for Case II
    
"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2025, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"

from simulator import simulator
import matplotlib.pyplot as plt
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import pickle

#cleaning up  the data files
try:
    os.remove("outs.dat")
except OSError:
    pass
try:
    os.remove("samples.dat")
except OSError:
    pass

samples = []
print("sampling", flush=True)
for i in range(0, 5000):
    samples.append(np.random.randint(low=0, high=2, size=23))

print("sampling done", flush=True)

num_cores = multiprocessing.cpu_count()
outs = Parallel(n_jobs=num_cores)(delayed(simulator.evaluate)(samples[i]) for i in range(0, 5000))

print("done", flush=True)

f = open('outs.dat', 'ab')
pickle.dump(outs , f)
f.close()

f = open('samples.dat', 'ab')
pickle.dump(samples , f)
f.close()

n, bins, patches = plt.hist(x=outs, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)#, grid=True)
plt.xlabel('Force')
plt.ylabel('Counts')
plt.title('Random Search')
#plt.xlim([0, 8])
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.show()
plt.savefig("histogram.jpg", dpi=300)
