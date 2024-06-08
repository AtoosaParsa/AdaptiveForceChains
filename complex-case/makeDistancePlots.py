## plot stuff after loading everything from the pickled files for MOO

import time, array, random, copy, math
import numpy as np
from deap import algorithms, base, benchmarks, tools, creator
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
#import seaborn
#import pandas as pd
import random
import pickle
from os.path import exists
import os
from simulator6 import simulator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# deap setup:
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
#toolbox.register("map", futures.map)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
# also save the population of each generation
stats.register("pop", copy.deepcopy)

toolbox.pop_size = 100
toolbox.max_gen = 1000
toolbox.mut_prob = 0.8

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

hof = tools.HallOfFame(1, similar=np.array_equal) #can change the size

# load pareto front from the file
indices = pickle.load(open('indices.pickle', 'rb'))
outputs = pickle.load(open('outputs.pickle', 'rb'))

# print all of pareto front individuals
for i in range(len(outputs)):
    print(str(i) + ": " + str(outputs[i]))

plt.figure(figsize=(4,4))
i = 0
colors = []
xs = []
ys = []
for output in outputs:
    colors.append(np.sum(1-np.array(indices[i])))
    xs.append(output[2])
    ys.append(output[1]+output[0])
    #plt.annotate(str(i), (output[0], output[1]))
    i = i+1
plt.scatter(x=xs, y=ys, c=colors, s=100, edgecolors='black', linewidth = 0.5, cmap='Reds')
plt.xlabel('Switching Particles', fontsize=28)
plt.ylabel('$(f1*f2*f3)_{c1} + (f1*f2)_{c2}$', fontsize=28)
plt.title("solutions from the last generation", fontsize=28)
plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.clim(min(colors), max(colors))
cbar=plt.colorbar()
cbar.set_label("Soft Particles", fontsize=24)
cbar.ax.tick_params(labelsize=24)
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
i = 0
colors = []
xs = []
ys = []
for output in outputs:
    colors.append(output[2])
    xs.append(output[0])
    ys.append(output[1])
    #plt.annotate(str(i), (output[0], output[1]))
    i = i+1
plt.scatter(x=xs, y=ys, c=colors, s=100, edgecolors='black', linewidth = 0.5, cmap='Blues')
plt.xlabel('$(f1*f2*f3)_{c1}$', fontsize=28)
plt.ylabel('$(f1*f2)_{c2}$', fontsize=28)
plt.title("solutions from the last generation", fontsize=28)
plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontsize=18)
plt.yticks(fontsize=24)
plt.clim(min(colors), max(colors))
cbar=plt.colorbar()
cbar.set_label("Switching Particles", fontsize=24)
cbar.ax.tick_params(labelsize=24)
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
i = 0
colors = []
xs = []
ys = []
for output in outputs:
    colors.append(np.sum(1-np.array(indices[i])))
    xs.append(output[0])
    ys.append(output[1])
    #plt.annotate(str(i), (output[0], output[1]))
    i = i+1
plt.scatter(x=xs, y=ys, c=colors, s=100, edgecolors='black', linewidth = 0.5, cmap='Greens')
plt.xlabel('$(f1*f2*f3)_{c1}$', fontsize=28)
plt.ylabel('$(f1*f2)_{c2}$', fontsize=28)
plt.title("solutions from the last generation", fontsize=28)
plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontsize=18)
plt.yticks(fontsize=24)
plt.clim(min(colors), max(colors))
cbar=plt.colorbar()
cbar.set_label("Soft Particles", fontsize=24)
cbar.ax.tick_params(labelsize=24)
plt.tight_layout()
plt.show()