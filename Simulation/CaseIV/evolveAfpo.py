""" 
    Age Fitness Pareto Optimization
    Multiobjective Optimization
    Set the number of fitnesses on line 27
    
"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2025, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"

from afpo import AFPO
import os
import constants as c
import random
import numpy as np

import subprocess as sub
import sys

# input the seed
seed = int(sys.argv[1])

#cleaning up the data files
sub.call(f"mkdir data", shell=True)
sub.call(f"rm -rf data/gens{seed}", shell=True)
sub.call(f"mkdir data/gens{seed}/", shell=True)

try:
    os.remove(f"data/lastGeneration{seed}.dat")
except OSError:
    pass
try:
    os.remove(f"data/avgFitness{seed}.dat")
except OSError:
    pass
try:
    os.remove(f"data/bests{seed}.dat")
except OSError:
    pass

print("*********************************************************", flush=True)
print("run: "+str(seed), flush=True)
random.seed(seed)
np.random.seed(seed)
afpo = AFPO(seed, 2)
afpo.evolve()
