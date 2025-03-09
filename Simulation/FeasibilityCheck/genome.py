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
from individual import INDIVIDUAL
import pickle

class GENOME:

    def __init__(self,ID,fitness=c.worstFitness):
        self.Set_ID(ID)
        self.indv = INDIVIDUAL(ID)
        self.age = 0
        self.fitness = fitness

    def Age(self):
        self.age = self.age + 1

    def Dominates(self,other):
        if self.Get_Fitness() <= other.Get_Fitness():
            if self.Get_Age() <= other.Get_Age():
                equalFitnesses = self.Get_Fitness() == other.Get_Fitness()
                equalAges      = self.Get_Age()     == other.Get_Age()
                if not equalFitnesses and equalAges:
                    return True
                else:
                    return self.Is_Newer_Than(other)
            else:
                return False
        else:
            return False

    def Evaluate(self):
        f = self.indv.Compute_Fitness()
        self.fitness = -f
        return self.fitness
        
    def Get_Age(self):
        return self.age

    def Get_Fitness(self):
        return self.fitness

    def Mutate(self):
        self.indv.Mutate()

    def Print(self):
        print(' fitness: ' , end = '' )
        print(self.fitness , end = '' )

        print(' age: '     , end = '' )
        print(self.age     , end = '' )

        print()

    def Save(self,randomSeed):
        f = open('data/savedRobotsAfpoSeed.dat', 'ab')
        pickle.dump(self.indv , f)
        f.close()
        pass

    def Set_ID(self,ID):
        self.ID = ID

    def Show(self):
        self.indv.Compute_Fitness(True)

# -------------------- Private methods ----------------------

    def Get_ID(self):
        return self.ID

    def Is_Newer_Than(self,other):
        return self.Get_ID() > other.Get_ID()
