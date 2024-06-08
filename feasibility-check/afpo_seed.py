#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:19:50 2021

@author: atoosa
"""

import constants as c
import copy
import numpy as np
import operator
import statistics
import pickle
import os
from joblib import Parallel, delayed
import multiprocessing

from genome_seed import GENOME 

class AFPO:

    def __init__(self,randomSeed):        
        self.randomSeed = randomSeed
        self.currentGeneration = 0
        self.nextAvailableID = 0

        self.genomes = {}

        for populationPosition in range(c.popSize):

            self.genomes[populationPosition] = GENOME(self.nextAvailableID)
            self.nextAvailableID = self.nextAvailableID + 1

    def Evolve(self):
        self.Perform_First_Generation()
        for self.currentGeneration in range(1,c.numGenerations):
            self.Perform_One_Generation()
            #try:
            #    os.remove("savedRobotsLastGenAfpoSeed.dat")
            #except OSError:
            #    pass
            #self.SaveLastGen()
        self.SaveLastGen()

# -------------------------- Private methods ----------------------

    def Age(self):
        for genome in self.genomes:
            self.genomes[genome].Age()

    def Aggressor_Dominates_Defender(self,aggressor,defender):
        return self.genomes[aggressor].Dominates(self.genomes[defender])

    def Choose_Aggressor(self):
        return np.random.randint(c.popSize)

    def Choose_Defender(self,aggressor):
        defender = np.random.randint(c.popSize)

        while defender == aggressor:
            defender = np.random.randint(c.popSize)
        return defender

    def Contract(self):
        while len(self.genomes) > c.popSize:
            aggressorDominatesDefender = False
            while not aggressorDominatesDefender:
                aggressor = self.Choose_Aggressor()
                defender  = self.Choose_Defender(aggressor)
                aggressorDominatesDefender = self.Aggressor_Dominates_Defender(aggressor,defender)
            for genomeToMove in range(defender,len(self.genomes)-1):
                self.genomes[genomeToMove] = self.genomes.pop(genomeToMove+1)

    def Evaluate_Genomes(self):
    	num_cores = multiprocessing.cpu_count()
    	print("we have")
    	print(num_cores)
    	print("cores")
    	outs = Parallel(n_jobs=num_cores)(delayed(self.genomes[genome].Evaluate)() for genome in self.genomes)
    	#print("done")
    	#print(type(outs))
    		
    	#self.genomes = copy.deepcopy(np.array(outs))
    	#self.genomes = dict(zip(list(range(0, c.popSize)), outs))
    	#print(type(self.genomes))
    	
    	for genome in self.genomes:
    	#	print(genome)
    	#	print(self.genomes[genome].indv.genome)
    	#	print(outs[genome])
    		self.genomes[genome].fitness = outs[genome]
    		self.genomes[genome].indv.fitness = -outs[genome]

    def Expand(self):
        popSize = len(self.genomes)
        for newGenome in range( popSize , 2 * popSize - 1 ):
            spawner = self.Choose_Aggressor()
            self.genomes[newGenome] = copy.deepcopy(self.genomes[spawner])
            self.genomes[newGenome].Set_ID(self.nextAvailableID)
            self.nextAvailableID = self.nextAvailableID + 1
            self.genomes[newGenome].Mutate()

    def Find_Best_Genome(self):
        genomesSortedByFitness = sorted(self.genomes.values(), key=operator.attrgetter('fitness'),reverse=False)
        return genomesSortedByFitness[0]

    def Find_Avg_Fitness(self):
        add = 0        
        for g in self.genomes:
            add += self.genomes[g].fitness
        return  add/c.popSize
    
    def Inject(self):
        popSize = len(self.genomes)
        self.genomes[popSize-1] = GENOME(self.nextAvailableID)
        self.nextAvailableID = self.nextAvailableID + 1

    def Perform_First_Generation(self):
        self.Evaluate_Genomes()
        self.Print()
        self.Save_Best()
        self.Save_Avg()

    def Perform_One_Generation(self):
        self.Expand()
        self.Age()
        self.Inject()
        self.Evaluate_Genomes()
        self.Contract()
        self.Print()
        self.Save_Best()
        self.Save_Avg()

    def Print(self):
        print('Generation ', end='', flush=True)
        print(self.currentGeneration, end='', flush=True)
        print(' of ', end='', flush=True)
        print(str(c.numGenerations), end='', flush=True)
        print(': ', end='', flush=True)

        bestGenome = self.Find_Best_Genome()
        bestGenome.Print()

    def Save_Best(self):
        bestGenome = self.Find_Best_Genome()
        bestGenome.Save(self.randomSeed)
    
    def SaveLastGen(self):
        genomesSortedByFitness = sorted(self.genomes.values(), key=operator.attrgetter('fitness'),reverse=False)
        f = open('savedRobotsLastGenAfpoSeed.dat', 'ab')
        pickle.dump(genomesSortedByFitness, f)
        f.close()

    def Save_Avg(self):
        f = open('avgFitnessAfpoSeed.dat', 'ab')
        avg = self.Find_Avg_Fitness()
        print('Average ' + str(avg))
        print()
        #f.write("%.3f\n" % avg)
        pickle.dump(avg, f)
        f.close()
        
    def Show_Best_Genome(self):
        bestGenome = self.Find_Best_Genome()
        bestGenome.Show()
