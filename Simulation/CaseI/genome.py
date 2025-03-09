""" 
    Age Fitness Pareto Optimization
    genome is an array of binary numbers
    need to change the dimension of the grid on line 19
    change lines 106, 110 and 113 depending on the objectives
    
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
import uuid

class GENOME:

    def __init__(self, num_fitnesses, id):
        self.ID = id #self.set_uuid()
        self.age = 0
        # total number of the particles in the grid
        self.N = 23
        self.genome = np.random.randint(low=0, high=2, size=self.N)
        self.fitnesses = [0.0 for x in range(num_fitnesses)]
        # determines if this individual was already evaluated
        self.needs_eval = True
        # metric
        self.metric = [0.0 for x in range(num_fitnesses)]

    def aging(self):
        self.age = self.age + 1

    def dominates(self, other):
        # returns True if self dominates other param other, False otherwise.

        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.get_minimize_vals()
        other_max_traits = other.get_maximize_vals()

        # all min traits must be at least as small as corresponding min traits
        if list(filter(lambda x: x[0] > x[1], zip(self_min_traits, other_min_traits))):
            return False

        # all max traits must be at least as large as corresponding max traits
        if list(filter(lambda x: x[0] < x[1], zip(self_max_traits, other_max_traits))):
            return False

        # any min trait smaller than other min trait
        if list(filter(lambda x: x[0] < x[1], zip(self_min_traits, other_min_traits))):
            return True

        # any max trait larger than other max trait
        if list(filter(lambda x: x[0] > x[1], zip(self_max_traits, other_max_traits))):
            return True

        # all fitness values are the same, default to return False.
        return self.ID < other.ID

    def dominates2(self, other):
        # new function for plotting the pareto front without considering the age
        # returns True if self dominates other param other, False otherwise.

        self_min_traits = [] #self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = [] #other.get_minimize_vals()
        other_max_traits = other.get_maximize_vals()

        # all min traits must be at least as small as corresponding min traits
        if list(filter(lambda x: x[0] > x[1], zip(self_min_traits, other_min_traits))):
            return False

        # all max traits must be at least as large as corresponding max traits
        if list(filter(lambda x: x[0] < x[1], zip(self_max_traits, other_max_traits))):
            return False

        # any min trait smaller than other min trait
        if list(filter(lambda x: x[0] < x[1], zip(self_min_traits, other_min_traits))):
            return True

        # any max trait larger than other max trait
        if list(filter(lambda x: x[0] > x[1], zip(self_max_traits, other_max_traits))):
            return True

        # all fitness values are the same, default to return False.
        return self.ID < other.ID

    def distance(self, other):
        # checks the distance in genotype space, if they are close enough, returns true
        dist = np.abs(self.genome - other.genome)
        threshold = 1
        return (dist<=threshold).all()

    def dominatesAll(self, other):
        # used for printing generation summary
        dominates = True
        for index in range(len(self.fitnesses)):
            dominates = dominates and (self.fitnesses[index] > other.fitnesses[index])
        return dominates

    def evaluate(self):
        if self.needs_eval == True:
            #sim = simulator(self.genome)
            output = simulator.evaluate(self.genome)
            self.fitnesses = output
            self.metric = output[0]
            self.needs_eval = False
        return self.fitnesses

    def get_minimize_vals(self):
        return [self.age]

    def get_maximize_vals(self):
        return [self.fitnesses[0]]

    def mutate(self):
        self.needs_eval = True
        mutationRate = 0.05
        probToMutate = np.random.choice([False, True], size=self.genome.shape, p=[1-mutationRate, mutationRate])
        candidate = np.where(probToMutate, 1-self.genome, self.genome)

        while np.all(candidate == self.genome):
            probToMutate = np.random.choice([False, True], size=self.genome.shape, p=[1-mutationRate, mutationRate])
            candidate = np.where(probToMutate, 1-self.genome, self.genome)
        
        self.genome = candidate

        self.fitnesses = [0.0 for x in range(len(self.fitnesses))]
        #self.age = 0
        #self.set_uuid()

    def genomePrint(self):
        print(' [fitness: ' , end = '' )
        print(self.fitnesses , end = '' )

        print(' age: ', end = '' )
        print(str(self.age)+']', end = '' )

        print()

        print(self.genome)

        print()

    def genomeShow(self):
        print("fitness is: ")
        print(self.fitnesses)

    def set_uuid(self):
        self.ID = uuid.uuid1()
        return self.ID
