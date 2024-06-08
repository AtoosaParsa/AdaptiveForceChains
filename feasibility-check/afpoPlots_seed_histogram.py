import pickle
import matplotlib.pyplot as plt
from simulator4 import simulator
import constants as c
import numpy
import matplotlib
from scipy.stats import norm

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

# set up the axis and subplots
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
#x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.plot(list(range(1, gens+1)), mean_f, color='red', linewidth=2, label='best')
main_ax.fill_between(list(range(1, gens+1)), mean_f-std_f, mean_f+std_f, color='red', alpha=0.3, linewidth=1)

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

main_ax.plot(list(range(1, gens+1)), -mean_f, color='blue', linewidth=2, label='average')
main_ax.fill_between(list(range(1, gens+1)), -(mean_f-std_f), -(mean_f+std_f), color='cornflowerblue', alpha=0.3, linewidth=1)
main_ax.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
main_ax.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
main_ax.minorticks_on()
main_ax.legend(loc='upper right')
main_ax.set_ylim([0.0, 0.36])
main_ax.set_xlabel('Generations', fontsize=24)
main_ax.tick_params(axis='y', which='both', labelsize=0)
main_ax.tick_params(axis='x', which='both', labelsize=20)

# plot the histogram
with open('RS.dat', "rb") as f:
    rs = pickle.load(f)
f.close()
_, bins, _ = y_hist.hist(x=rs, bins='auto', color='gray', alpha=0.7, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True, label='random') #, grid=True)histtype='stepfilled',
#_, bins, _ = y_hist.hist(x=ys, bins=bins, color='blue', alpha=0.4, rwidth=0.85, cumulative=False, orientation='horizontal', histtype='stepfilled', density=True, label='evolved') #, grid=True)histtype='stepfilled',
y_hist.invert_xaxis()
y_hist.minorticks_on()
y_hist.set_ylabel('Fitness', fontsize=24)
formatter = matplotlib.ticker.FormatStrFormatter('%1.3f')
#formatter2 = matplotlib.ticker.FormatStrFormatter('%1.4f')
y_hist.yaxis.set_major_formatter(formatter)
#y_hist.yaxis.set_minor_formatter(formatter2)
y_hist.tick_params(axis='y', which='major', labelsize=20)
#y_hist.tick_params(axis='y', which='minor', labelsize=14)
y_hist.legend(loc='upper left')
y_hist.set_ylim([0.0, 0.36])

# fitting a normal distribution
#mu, std = norm.fit(rs)
#xmin, xmax = y_hist.get_ylim()
#x = numpy.linspace(xmin, xmax, 1000)
#p = norm.pdf(x, mu, std)

#plt.plot(x, p, linewidth=2)
#myText = "Mean={:.2f}, STD={:.2f}".format(mu, std)
#y_hist.hlines(y=0.12, xmin=0, xmax=10, linewidth=1, linestyle='--', color='limegreen', alpha=0.9)
#y_hist.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=28, color='limegreen', fontweight='bold')


plt.show()