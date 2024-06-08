from simulator4 import simulator
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
        f = open('savedFitnessSeed.dat', 'ab')
        pickle.dump(self.fitness , f)
        f.close()
    
    def SaveBest(self):
        f = open('savedBestsSeed.dat', 'ab')
        pickle.dump(self.genome , f)
        f.close()
