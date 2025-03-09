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

import constants as c
import copy
import numpy as np
import operator
import statistics
import pickle
from joblib import Parallel, delayed
import multiprocessing
import time


from genome import GENOME 

class AFPO:

    def __init__(self, randomSeed, num_fitnesses):        
        self.randomSeed = randomSeed
        self.currentGeneration = 0
        self.num_fitnesses = num_fitnesses
        self.nextAvailableID = 0

        self.genomes = {}

        for populationPosition in range(c.popSize):
            self.genomes[populationPosition] = GENOME(self.num_fitnesses, self.nextAvailableID)
            self.nextAvailableID = self.nextAvailableID + 1
        

    def evolve(self):
        self.performFirstGen()
        for self.currentGeneration in range(1, c.numGenerations):
            self.performOneGen()
            # making a checkpoint every 100 gens - save the whole population
            #if (self.currentGeneration % 100 == 0):
            self.SaveGen(self.currentGeneration)
        self.SaveLastGen()

    def aging(self):
        for genome in self.genomes:
            self.genomes[genome].aging()


    def reducePopulation(self):
        pf = self.paretoFront()
        pf_size = len(pf)

        print("pf_size: "+str(pf_size), flush=True)

        counter = 0

        # remove individuals until the population size is popSize or size of the pareto front
        while len(self.genomes) > c.popSize: #max(pf_size, c.popSize):

            #print(counter, flush=True)
            counter = counter + 1
            
            pop_size = len(self.genomes)
            ind1 = np.random.randint(pop_size)
            ind2 = np.random.randint(pop_size)
            while ind1 == ind2:
                ind2 = np.random.randint(pop_size)

            # find the dominated genome and remove it
            if self.genomes[ind1].dominates(self.genomes[ind2]):  # ind1 dominates
                for i in range(ind2, len(self.genomes) - 1):
                    self.genomes[i] = self.genomes.pop(i + 1)

            elif self.genomes[ind2].dominates(self.genomes[ind1]):  # ind2 dominates
                for i in range(ind1, len(self.genomes) - 1):
                    self.genomes[i] = self.genomes.pop(i + 1)

            elif pf_size > c.popSize:
                # if the distance between the two genomes is small, keep the one with higher metric
                if self.genomes[ind1].distance(self.genomes[ind2]):
                    print("there are close genomes!", flush=True)
                    if self.genomes[ind1].metric >= self.genomes[ind2].metric: # keep ind1
                        for i in range(ind2, len(self.genomes) - 1):
                            self.genomes[i] = self.genomes.pop(i + 1)
                    else: # keep ind2
                        for i in range(ind1, len(self.genomes) - 1):
                            self.genomes[i] = self.genomes.pop(i + 1)
                
                elif counter > 2*c.popSize: # to prevent the loop to go forever, if we've been here for a while
                    print("pareto front saturating, try increasing the popSize", flush=True) 
                    for i in range(ind2, len(self.genomes) - 1): # just randomly keep one of the two
                        self.genomes[i] = self.genomes.pop(i + 1)



    def evaluateAll(self):
        num_cores = multiprocessing.cpu_count()
        print("we have", flush=True)
        print(num_cores)
        print("cores", flush=True)
        outs = Parallel(n_jobs=num_cores)(delayed(self.genomes[genome].evaluate)() for genome in self.genomes)
        
        for genome in self.genomes:
            self.genomes[genome].fitnesses = outs[genome]
            
    def evaluateInput(self, individuals):
        num_cores = multiprocessing.cpu_count()
        print("we have", flush=True)
        print(num_cores)
        print("cores", flush=True)
        outs = Parallel(n_jobs=num_cores)(delayed(indv.evaluate)() for indv in individuals)
        
        i = 0
        for indv in individuals:
            individuals[i].fitnesses = outs[i]
            i = i + 1
        
        return individuals

    def increasePopulation(self, children):
        popSize = len(self.genomes)
        j = 0
        for i in range(popSize , 2*popSize):
            self.genomes[i] = children[j]
            j = j + 1

    def breeding(self):
        # increase the population to 2*popSize-1 by adding random genomes and mutating them
        popSize = len(self.genomes)
        children = []
        for newGenome in range(popSize , 2*popSize-1):
            randGen = np.random.randint(popSize)
            child = copy.deepcopy(self.genomes[randGen])
            child.mutate()
            child.ID = self.nextAvailableID
            self.nextAvailableID = self.nextAvailableID + 1
            children.append(child)
        return children

    def findBestGenome(self):
        best = None
        for g in self.genomes:
            if best is None:
                best = self.genomes[g]
            if self.genomes[g] is not None and self.genomes[g].dominatesAll(best):
                best = self.genomes[g]
        return best

    def findAvgFitness(self):
        add = np.zeros(self.num_fitnesses)   
        for g in self.genomes:
            add += np.array(self.genomes[g].fitnesses)
        return  np.round(add/len(self.genomes), decimals=2)
    
    def injectOne(self, children):
        # add one ranom genome
        children.append(GENOME(self.num_fitnesses, self.nextAvailableID))
        self.nextAvailableID = self.nextAvailableID + 1
        return children

    def performFirstGen(self):
        self.evaluateAll()
        self.printing()
        self.saveBest()
        self.saveAvg()

    def performOneGen(self):
        print("pop size: "+str(len(self.genomes)), flush=True)
        print("aging", flush=True)
        self.aging()
        print("breeding", flush=True)
        children = self.breeding()
        print("injecting", flush=True)
        children = self.injectOne(children)
        print("evaluateing", flush=True)
        children = self.evaluateInput(children)
        print("entend the population")
        self.increasePopulation(children)
        print("reducing", flush=True)
        self.reducePopulation()
        print("Printing", flush=True)
        self.printing()
        print("Saving best", flush=True)
        self.saveBest()
        print("save avg", flush=True)
        self.saveAvg()

    def printing(self):
        print('Generation ', end='', flush=True)
        print(self.currentGeneration, end='', flush=True)
        print(' of ', end='', flush=True)
        print(str(c.numGenerations), end='', flush=True)
        print(': ', end='', flush=True)

        bestGenome = self.findBestGenome()
        bestGenome.genomePrint()

    def saveBest(self):
        bestGenome = self.findBestGenome()
        f = open(f'data/bests{self.randomSeed}.dat', 'ab')
        pickle.dump(bestGenome , f)
        f.close()
    
    def SaveLastGen(self):
        f = open(f'data/lastGeneration{self.randomSeed}.dat', 'ab')
        pickle.dump(self.genomes, f)
        f.close()

    def SaveGen(self, gen):
        f = open(f'data/gens{self.randomSeed}/gen{gen}.dat', 'ab')
        pickle.dump(self.genomes, f)
        f.close()

    def saveAvg(self):
        f = open(f'data/avgFitness{self.randomSeed}.dat', 'ab')
        avg = self.findAvgFitness()
        print('Average ' + str(avg))
        print()
        pickle.dump(avg, f)
        f.close()
        
    def showBestGenome(self):
        bestGenome = self.findBestGenome()
        bestGenome.genomeShow()

    def paretoFront(self):
        pareto_front = []

        for i in self.genomes:
            i_is_dominated = False
            for j in self.genomes:
                if i != j:
                    if self.genomes[j].dominates(self.genomes[i]):
                        i_is_dominated = True
            if not i_is_dominated:
                pareto_front.append(self.genomes[i])

        return pareto_front
