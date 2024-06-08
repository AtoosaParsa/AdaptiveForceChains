# multi-objective optimization to solve the checkerboard problem

# imports for DEAP
import time, array, random, copy, math
import numpy as np
from deap import algorithms, base, benchmarks, tools, creator
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

# imports for the simulator
from simulator6 import simulator
import random
import matplotlib.pyplot as plt
import pickle
from os.path import exists
from scoop import futures
import multiprocessing
import os
    
#cleaning up  the data files
try:
    os.remove("results.pickle")
except OSError:
    pass
try:
    os.remove("logs.pickle")
except OSError:
    pass
try:
    os.remove("hofs.pickle")
except OSError:
    pass
try:
    os.remove("hostfile")
except OSError:
    pass
try:
    os.remove("scoop-python.sh")
except OSError:
    pass

# start of the optimization:
random.seed(a=42)

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 46)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", simulator.evaluate)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# parallelization?
toolbox.register("map", futures.map)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
# also save the population of each generation
stats.register("pop", copy.deepcopy)

def main():
    toolbox.pop_size = 100
    toolbox.max_gen = 800
    toolbox.mut_prob = 0.8

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    hof = tools.HallOfFame(1, similar=np.array_equal) #can change the size

    def run_ea(toolbox, stats=stats, verbose=True, hof=hof):
        pop = toolbox.population(n=toolbox.pop_size)
        pop = toolbox.select(pop, len(pop))
        return algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.pop_size, 
                                        lambda_=toolbox.pop_size, 
                                        cxpb=1-toolbox.mut_prob, #: no cross-over?
                                        mutpb=toolbox.mut_prob, 
                                        stats=stats, 
                                        ngen=toolbox.max_gen, 
                                        verbose=verbose,
                                        halloffame=hof)

    res,log = run_ea(toolbox, stats=stats, verbose=True, hof=hof)

    return res, log, hof


if __name__ == '__main__':

    print("starting")
    res, log, hof = main()
    print("done")
    pickle.dump(res, open('results.pickle', 'wb'))
    pickle.dump(log, open('logs.pickle', 'wb'))
    pickle.dump(hof, open('hofs.pickle', 'wb'))

    avg = log.select("avg")
    std = log.select("std")
    avg_stack = np.stack(avg, axis=0)
    avg_f1 = avg_stack[:, 0]
    avg_f2 = avg_stack[:, 1]
    std_stack = np.stack(std, axis=0)
    std_f1 = std_stack[:, 0]
    std_f2 = std_stack[:, 1]

    plt.figure(figsize=(6.4,4.8))
    plt.plot(avg_f1, color='blue')
    plt.fill_between(list(range(0, toolbox.max_gen+1)), avg_f1-std_f1, avg_f1+std_f1, color='cornflowerblue', alpha=0.2)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness of Individuals in the Population - F1", fontsize='small')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("avg_F1.jpg", dpi = 300)

    plt.figure(figsize=(6.4,4.8))
    plt.plot(avg_f2, color='blue')
    plt.fill_between(list(range(0, toolbox.max_gen+1)), avg_f2-std_f2, avg_f2+std_f2, color='cornflowerblue', alpha=0.2)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness of Individuals in the Population - F2", fontsize='small')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("avg_F2.jpg", dpi = 300)
