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

from simulator import simulator
import constants as c
import random
import math
import numpy as np
import sys
import pickle

class INDIVIDUAL:
    def __init__(self, i):
        self.N_heavy = 9
        self.N = 23
        self.genome = np.random.randint(low=0, high=2, size=self.N)
        self.fitness = 0
        self.ID = i
    
    def Compute_Fitness(self, show=False):
        self.fitness = simulator.evaluate(self.genome)
        if show:
            simulator.showPacking(self.genome)
            print("fitness is:")
            print(self.fitness)
        return self.fitness
    
    def Mutate(self):
        mutationRate = 0.05
        probToMutate = np.random.choice([False, True], size=self.genome.shape, p=[1-mutationRate, mutationRate])
        candidate = np.where(probToMutate, 1-self.genome, self.genome)
        
        self.genome = candidate       
    
    def Print(self):
        print('[', self.ID, self.fitness, ']', end=' ')
        
    
    def Save(self):
        f = open('data/savedFitnessSeed.dat', 'ab')
        pickle.dump(self.fitness , f)
        f.close()
    
    def SaveBest(self):
        f = open('data/savedBestsSeed.dat', 'ab')
        pickle.dump(self.genome , f)
        f.close()
