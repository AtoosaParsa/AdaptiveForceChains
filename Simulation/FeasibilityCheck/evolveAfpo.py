""" 
    Age Fitness Pareto Optimization
    
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
import subprocess as sub
import sys

#cleaning up  the data files

sub.call(f"mkdir data", shell=True)

try:
    os.remove("data/savedRobotsLastGenAfpoSeed.dat")
except OSError:
    pass
try:
    os.remove("data/avgFitnessAfpoSeed.dat")
except OSError:
    pass
try:
    os.remove("data/savedRobotsAfpoSeed.dat")
except OSError:
    pass

runs = c.RUNS
for r in range(1, runs+1):
    print("*********************************************************", flush=True)
    print("run: "+str(r), flush=True)
    randomSeed = r
    random.seed(r)
    afpo = AFPO(randomSeed)
    afpo.Evolve()
    #afpo.Show_Best_Genome()
        


