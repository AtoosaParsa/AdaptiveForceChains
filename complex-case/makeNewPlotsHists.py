## plot stuff after loading everything from the pickled files for MOO
## making new pareto front plots

import time, array, random, copy, math
import numpy as np
from deap import algorithms, base, benchmarks, tools, creator
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
#import seaborn
#import pandas as pd
import random
import pickle
from os.path import exists
import os
from simulator6 import simulator
import topsispy as tp

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#import matplotlib.font_manager

#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Tahoma']

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
toolbox.max_gen = 800
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

# best from previous simulator
#print(simulator.showPackingWithContacts([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]))

# set up the axis and subplots
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# plot the scatter plot
i = 0
xs = []
ys = []
for output in outputs:
    xs.append(output[1])
    ys.append(output[0])
    #plt.annotate(str(i), (output[1], output[0]))
    if i == 0:
        xmark = output[1]
        ymark = output[0]
    i = i+1
main_ax.scatter(x=xs, y=ys, c='blue', s=150, alpha=0.5, edgecolors='black', linewidth = 0.5)
main_ax.scatter(x=xmark, y=ymark, c='red', s=170, alpha=1, edgecolors='black', linewidth = 0.5)
main_ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)
main_ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
main_ax.minorticks_on()
main_ax.xaxis.set_major_formatter(plt.NullFormatter())
main_ax.yaxis.set_minor_formatter(plt.NullFormatter())
main_ax.tick_params(axis='y', which='both', labelsize=0)
main_ax.tick_params(axis='x', which='both', labelsize=0)
main_ax.set_xlim((2.00e-2, 0.110))
main_ax.set_ylim((0.000, 0.012))

# plot the histograms
with open('O1.dat', "rb") as f:
    o1 = pickle.load(f)
f.close()
_, bins, _ = y_hist.hist(x=o1, bins='auto', color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True) #, label='random') #, grid=True)histtype='stepfilled',
#_, bins, _ = y_hist.hist(x=ys, bins=bins, color='blue', alpha=0.4, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True, label='evolved') #, grid=True)histtype='stepfilled',
y_hist.invert_xaxis()
y_hist.minorticks_on()
y_hist.set_ylabel('$O_1$', fontsize=42)
formatter = matplotlib.ticker.FormatStrFormatter('%1.2e')
#formatter2 = matplotlib.ticker.FormatStrFormatter('%1.4f')
y_hist.yaxis.set_major_formatter(formatter)
#y_hist.yaxis.set_minor_formatter(formatter2)
y_hist.tick_params(axis='y', which='major', labelsize=20)
#y_hist.tick_params(axis='y', which='minor', labelsize=14)
#y_hist.legend(loc='upper left')
y_hist.set_ylim((0.000, 0.012))

with open('O2.dat', "rb") as f:
    o2 = pickle.load(f)
f.close()
_, bins, _ = x_hist.hist(x=o2, bins='auto', color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='vertical', histtype='stepfilled', density=True) #, label='random') #, grid=True)histtype='stepfilled',
x_hist.invert_yaxis()
x_hist.minorticks_on()
x_hist.set_xlabel('$O_2$', fontsize=42)
formatter = matplotlib.ticker.FormatStrFormatter('%1.2e')
x_hist.xaxis.set_major_formatter(formatter)
x_hist.tick_params(axis='x', which='major', labelsize=16)
x_hist.set_xlim((2.00e-2, 0.110))

print(simulator.showPackingWithContacts(indices[0]))

##############################
# find the best solution from the pareto front
data = []
for output in outputs:
    data.append([output[0]*output[1], output[2]])
weights = [0.5, 0.5]
sign = [1, -1]
bestIndex, ranks = tp.topsis(data, weights, sign)
simulator.showPackingWithContacts(indices[bestIndex])
print(outputs[bestIndex])

# set up the axis and subplots
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

plt.figure(figsize=(4,4))
i = 0
xs = []
ys = []
for output in outputs:
    xs.append(output[0]*output[1])
    ys.append(output[2])
    #plt.annotate(str(i), (output[0]*output[1], output[2]))
    if i == bestIndex:
        xmark = output[0]*output[1]
        ymark = output[2]
    i = i+1
main_ax.scatter(x=xs, y=ys, c='blue', s=150, alpha=0.5, edgecolors='black', linewidth = 0.5)
main_ax.scatter(x=xmark, y=ymark, c='red', s=170, alpha=1, edgecolors='black', linewidth = 0.5)
main_ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)
main_ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
main_ax.minorticks_on()
main_ax.xaxis.set_major_formatter(plt.NullFormatter())
main_ax.yaxis.set_minor_formatter(plt.NullFormatter())
main_ax.tick_params(axis='y', which='both', labelsize=0)
main_ax.tick_params(axis='x', which='both', labelsize=0)
main_ax.set_xlim((0.0, 0.0012))
main_ax.set_ylim((-5, 23))

# plot the histograms
with open('O3.dat', "rb") as f:
    o3 = pickle.load(f)
f.close()
_, bins, _ = y_hist.hist(x=o3, bins=15, color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True) #, label='random') #, grid=True)histtype='stepfilled',
#_, bins, _ = y_hist.hist(x=ys, bins=bins, color='blue', alpha=0.4, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True, label='evolved') #, grid=True)histtype='stepfilled',
y_hist.invert_xaxis()
y_hist.minorticks_on()
y_hist.set_ylabel('$O_3$', fontsize=42)
formatter = matplotlib.ticker.FormatStrFormatter('%1.0f')
#formatter2 = matplotlib.ticker.FormatStrFormatter('%1.4f')
y_hist.yaxis.set_major_formatter(formatter)
#y_hist.yaxis.set_minor_formatter(formatter2)
y_hist.tick_params(axis='y', which='major', labelsize=20)
#y_hist.tick_params(axis='y', which='minor', labelsize=14)
#y_hist.legend(loc='upper left')
y_hist.set_ylim((-5, 23))

with open('O1O2.dat', "rb") as f:
    o1o2 = pickle.load(f)
f.close()
_, bins, _ = x_hist.hist(x=o1o2, bins='auto', color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='vertical', histtype='stepfilled', density=True) #, label='random') #, grid=True)histtype='stepfilled',
x_hist.invert_yaxis()
x_hist.minorticks_on()
x_hist.set_xlabel('$O_1 * O_2$', fontsize=42)
formatter = matplotlib.ticker.FormatStrFormatter('%1.2e')
x_hist.xaxis.set_major_formatter(formatter)
x_hist.tick_params(axis='x', which='major', labelsize=16)
x_hist.set_xlim((0.0, 0.0012))

#print(simulator.showPackingWithContacts(indices[64]))

###########################
# find the best solution from the pareto front
data = []
for output in outputs:
    data.append([output[0]*output[1], output[3]])
weights = [0.5, 0.5]
sign = [1, -1]
bestIndex, ranks = tp.topsis(data, weights, sign)
simulator.showPackingWithContacts(indices[bestIndex])
print(outputs[bestIndex])

# set up the axis and subplots
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

i = 0
xs = []
ys = []
for output in outputs:
    xs.append(output[0]*output[1])
    ys.append(output[3])
    #plt.annotate(str(i), (output[0]*output[1], output[3]))
    if i == bestIndex:
        xmark = output[0]*output[1]
        ymark = output[3]
    i = i+1
main_ax.scatter(x=xs, y=ys, c='blue', s=150, alpha=0.5, edgecolors='black', linewidth = 0.5)
main_ax.scatter(x=xmark, y=ymark, c='red', s=170, alpha=1, edgecolors='black', linewidth = 0.5)
main_ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)
main_ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
main_ax.minorticks_on()
main_ax.xaxis.set_major_formatter(plt.NullFormatter())
main_ax.yaxis.set_minor_formatter(plt.NullFormatter())
main_ax.tick_params(axis='y', which='both', labelsize=0)
main_ax.tick_params(axis='x', which='both', labelsize=0)
main_ax.set_xlim((0.0, 0.0012))
main_ax.set_ylim((-50, 450))

# plot the histograms
with open('O4.dat', "rb") as f:
    o4 = pickle.load(f)
f.close()
_, bins, _ = y_hist.hist(x=o4, bins=20, color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True) #, label='random') #, grid=True)histtype='stepfilled',
#_, bins, _ = y_hist.hist(x=ys, bins=bins, color='blue', alpha=0.4, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True, label='evolved') #, grid=True)histtype='stepfilled',
y_hist.invert_xaxis()
y_hist.minorticks_on()
y_hist.set_ylabel('$O_4$', fontsize=42)
formatter = matplotlib.ticker.FormatStrFormatter('%1.0f')
#formatter2 = matplotlib.ticker.FormatStrFormatter('%1.4f')
y_hist.yaxis.set_major_formatter(formatter)
#y_hist.yaxis.set_minor_formatter(formatter2)
y_hist.tick_params(axis='y', which='major', labelsize=20)
#y_hist.tick_params(axis='y', which='minor', labelsize=14)
#y_hist.legend(loc='upper left')
y_hist.set_ylim((-50, 450))

with open('O1O2.dat', "rb") as f:
    o1o2 = pickle.load(f)
f.close()
_, bins, _ = x_hist.hist(x=o1o2, bins='auto', color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='vertical', histtype='stepfilled', density=True) #, label='random') #, grid=True)histtype='stepfilled',
x_hist.invert_yaxis()
x_hist.minorticks_on()
x_hist.set_xlabel('$O_1 * O_2$', fontsize=42)
formatter = matplotlib.ticker.FormatStrFormatter('%1.2e')
x_hist.xaxis.set_major_formatter(formatter)
x_hist.tick_params(axis='x', which='major', labelsize=16)
x_hist.set_xlim((0.0, 0.0012))

plt.show()
#print(simulator.showPackingWithContacts(indices[39]))
