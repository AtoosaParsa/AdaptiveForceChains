import pickle
import matplotlib.pyplot as plt
from simulator4 import simulator
import constants as c
import numpy

runs = c.RUNS
gens = c.numGenerations
fitnesses = numpy.zeros([runs, gens])
temp = []
individuals = []
with open('savedRobotsAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        for g in range(1, gens+1):
            try:
                if g == gens:
                    individuals.append(pickle.load(f))
                    temp.append(individuals[0].fitness)
                else:
                    temp.append(pickle.load(f).fitness)
            except EOFError:
                break
        fitnesses[r-1] = temp
        temp = []
f.close()

mean_f = numpy.mean(fitnesses, axis=0)
std_f = numpy.std(fitnesses, axis=0)

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), mean_f, color='blue', linewidth=2)
plt.fill_between(list(range(1, gens+1)), mean_f-std_f, mean_f+std_f, color='cornflowerblue', alpha=0.3, linewidth=1)
plt.xlabel("Generations", fontsize=32)
plt.ylabel("Best Fitness", fontsize=32)
plt.title("Fitness of the Best Individual in the Population", fontsize=32)
plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.tight_layout()
#plt.legend(['two robot', 'three robots'], loc='upper left')
plt.ylim([0.11, 0.36])
plt.show()

print(mean_f[gens-1])

## plot the average fitness
temp = []
fitnesses = numpy.zeros([runs, gens])
with open('avgFitnessAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        for g in range(1, gens+1):
            try:
                temp.append(pickle.load(f))
            except EOFError:
                break
        fitnesses[r-1] = temp
        temp = []
f.close()

mean_f = numpy.mean(fitnesses, axis=0)
std_f = numpy.std(fitnesses, axis=0)

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), -mean_f, color='blue', linewidth=2)
plt.fill_between(list(range(1, gens+1)), -(mean_f-std_f), -(mean_f+std_f), color='cornflowerblue', alpha=0.3, linewidth=1)
plt.xlabel("Generations", fontsize=32)
plt.ylabel("Best Fitness", fontsize=32)
plt.title("Average Fitness of Individuals in the Population", fontsize=32)
plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.tight_layout()
#plt.legend(['two robot', 'three robots'], loc='upper left')
plt.ylim([0.11, 0.36])
plt.show()

print(mean_f[gens-1])

# running the drill and the previous best
inds = numpy.zeros(23)
for i in [0, 15, 4, 18, 8, 17, 7, 6, 19, 20, 10, 9, 21, 12, 3]:
    inds[i]= 1
print(simulator.showPackingWithContacts(inds))

# running the best individuals
bests = numpy.zeros([runs, gens])
temp = []
rubish = []
with open('savedRobotsLastGenAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        # best individual of last generation
        best = temp[0]
        #simulator.showPacking(best.indv.genome)
        #print(simulator.evaluate(best.indv.genome))
        print(best.indv.genome)
        print(simulator.showPackingWithContacts(best.indv.genome))
        temp = []
f.close()

# running all of the individuals of the last generation - plotting the pareto front

bests = numpy.zeros([runs, gens])
temp = []
rubish = []
with open('savedRobotsLastGenAfpoSeed.dat', "rb") as f:
    ages = []
    fits = []
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        for i in range(0, c.popSize):
            if (i < 15): # just show the best 15, there are 30 individuals, show 15 of them.
                print(simulator.showPackingWithContacts(temp[i].indv.genome))
            ages.append(temp[i].age)
            fits.append(-temp[i].fitness)
        
        plt.figure(figsize=(6.4,4.8))
        plt.plot(ages, fits, 'o', color='black')
        plt.xlabel("Age")
        plt.ylabel("Fitness")
        plt.title("Last Generation", fontsize='small')
        plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        temp = []
        ages = []
        fits = []
f.close()
