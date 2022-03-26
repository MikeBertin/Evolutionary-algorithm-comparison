##!/usr/bin/env python
# coding: utf-8

# Methinks evolutionary algorithm with and without crossover comparison with hillclimber

# Import libraries #
import numpy as np
import pandas as pd
import string
import random
import matplotlib as mpl    # Required to make graph plot from CLI along - see below 
mpl.use('tkagg')            # From: https://github.com/matplotlib/matplotlib/issues/18953 - to make plot work 
import matplotlib.pyplot as plt

# Global variables
EVALUATION, ODDS, GOAL = 0, 1, False
np.random.seed()   

# Return a random [ascii + ' '] char
def random_char():
    return random.choice(string.ascii_letters + ' ')


# Return random 1D array of random_char() of size = target string
def random_string(target):
    return [random_char() for letter in target]


# Return 2D array of random char arrays
def make_population(population_size, target):
    return [random_string(target) for i in range(population_size)]    


# Return string labels array
def make_labels(population_size):
    return ["Str "+ str(i+1) + ":" for i in range(population_size)]


# Return Data frame to store population with index labels
def make_frame(population_size, target):
    return pd.DataFrame(make_population(population_size, target), index = make_labels(population_size))


# Check if fitness has reached the target length
def goal_check(fitness, target, string_index):
    
    if fitness != len(target):
        return False
    else:
        print('\n   Goal: String', string_index, 'wins!')
        return True


# Compare strings - if identical set Goal=true - return fitness as int
def fitness_test(candidate, target, string_index):
    
    global EVALUATION, GOAL
    EVALUATION += 1
    print('   Fitness evaluations:'+str(EVALUATION), end='\r', flush=True)
    # Count identical elements of arrays
    fitness = sum(candidate[i] == target[i] for i in range(len(target)))
    GOAL = goal_check(fitness, target, string_index) 
    
    return fitness


# Group fitness - test fitness of each string and return frame with fitness values
def group_fitness_test(population, population_size, target):
    fitness_array = [fitness_test(population.iloc[i], target, i) for i in range(population_size)]
    return pd.DataFrame(fitness_array, index = make_labels(population_size), columns = ['Fitness'])


# Mutate letters in string with probability 1/target string length
def mutate(individual, target):
    return [random_char() if random.random() < ODDS**-1 else individual[i] for i in range(len(target))]


# Hill climber - mutate each individual and replace if mutant fitter than parent
def hillclimb(group, fitness, population_size, target):
    
    for i in range(population_size):
        mutant = mutate(group.iloc[i], target)
        mutant_fitness = fitness_test(mutant, target, i)
        # Replace parent with mutant if fitness greater than stored value of parent fitness
        if  mutant_fitness >= fitness.iloc[i]['Fitness']:
            group.iloc[i] = mutant
            fitness.iloc[i] = mutant_fitness
        if GOAL is True: return group, fitness
        
    return group, fitness


# GA with no crossover - select fittest of two random parents and mutate, replace least fit of two other randoms
def ga_no_cross(group, fitness, population_size, target):
    
    randoms = [np.random.randint(0, population_size) for i in range(4)]  # Make array of four random ints
    parents = [group.iloc[randoms[i]] for i in range(2)]                # Select two random parents
    
    # Mutate the fittest parent 
    if fitness.iloc[randoms[0]]['Fitness'] > fitness.iloc[randoms[1]]['Fitness']:
        mutant = mutate(parents[0], target)
        mutant_fitness = fitness_test(mutant, target, randoms[0])
    else:
        mutant = mutate(parents[1], target)
        mutant_fitness = fitness_test(mutant, target, randoms[1])
    
    # Replace the least fit of two random individuals with the mutant
    if  fitness.iloc[randoms[2]]['Fitness'] < fitness.iloc[randoms[3]]['Fitness']:
        group.iloc[randoms[2]] = mutant
        fitness.iloc[randoms[2]] = mutant_fitness
    else: 
        group.iloc[randoms[3]] = mutant
        fitness.iloc[randoms[3]] = mutant_fitness
    
    return group, fitness


# Crossover function - combine two parent strings randomly with 50:50 chance
def crossover(parent1, parent2):
    return [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]


# GA with crossover
def ga_with_cross(group, fitness, population_size, target):

    randoms = [np.random.randint(0, population_size) for i in range(6)]  # Make array of six random ints
    parents = [group.iloc[randoms[i]] for i in range(4)]                # Select four random parents
    
    # Select fittest two parents from two pairs of parents
    if fitness.iloc[randoms[0]]['Fitness'] > fitness.iloc[randoms[1]]['Fitness']:
        chosen_parent_1 = parents[0]
    else:
        chosen_parent_1 = parents[1]
    
    if fitness.iloc[randoms[2]]['Fitness'] > fitness.iloc[randoms[3]]['Fitness']:
        chosen_parent_2 = parents[2]
    else:
        chosen_parent_2 = parents[3]
    
    # Mutate crossover to make mutant child of two parents
    mutant_child = mutate(crossover(chosen_parent_1, chosen_parent_2), target)
    
    # Replace least fit of randomly selected pair with mutant child and its fitness
    if  fitness.iloc[randoms[4]]['Fitness'] < fitness.iloc[randoms[5]]['Fitness']:
        group.iloc[randoms[4]] = mutant_child
        fitness.iloc[randoms[4]] = fitness_test(mutant_child, target, randoms[4])
    else: 
        group.iloc[randoms[5]] = mutant_child
        fitness.iloc[randoms[5]] = fitness_test(mutant_child, target, randoms[5])
    
    return group, fitness


# Return new random population and test group fitness
def initialise(population_size, target):
    new_group = make_frame(population_size, target)
    initial_fitness = group_fitness_test(new_group, population_size, target)
    return new_group, initial_fitness


# Iterate until GOAL fitness condition or counter limit is satisfied
def evolve(method, population_size, target):
     
    global EVALUATION, GOAL
    EVALUATION, GOAL = 0, False
    
    group, fitness = initialise(population_size, target)

    while GOAL is False:
        if method == 'Hillclimber':
            group, fitness = hillclimb(group, fitness, population_size, target)
        elif method == 'GA':
            group, fitness = ga_no_cross(group, fitness, population_size, target)
        elif method == 'GA with crossover':
            group, fitness = ga_with_cross(group, fitness, population_size, target)
        else:
            break
    
    return EVALUATION


# Loop for each method and odds - call evolve function (average samples) - returns a 2D matrix
def run_average(population_size, target, runs, step, average_no, odds_array, methods):
        
    global ODDS
    run_averages = [[0 for x in odds_array] for y in methods]
    
    for i, method in enumerate(methods):
        print(' Method:', method)
        for j, odds in enumerate(odds_array):
            print('  Mutation odds:', odds)
            samples = []
            ODDS = odds*len(target)
            
            for k in range(average_no):
                samples.append(evolve(method, population_size, target))
            run_averages[i][j] = round(sum(samples) / average_no)
            
    return run_averages


def main():

    # Collect 3D array of data varying population size, number of samples per average and odds
    methods = ['GA', 'GA with crossover', 'Hillclimber']

    # Get user input from CLI
    population_size = int(input('Please enter population start size: '))
    runs = int(input('Please enter number of runs: '))
    step = int(input('Please enter population increment size: '))
    average_no = int(input('Please enter number of averages: '))
    # Odds values for mutation - More or less likely relative to 1/target length
    str_arr = input('Mutation odds array (space seperated): ').split(' ')
    odds_array = [float(num) for num in str_arr]
    target = input('Please enter target: ')

    data = []
    print('Averages:', average_no, ' Runs:', runs, ' Step:', step,  ' Target:', target)
    # Vary the population size and collect in an array
    for i in range(0, runs*step, step):
        print('Population size:', population_size + i)
        data.append(run_average(population_size + i, target, runs, step, average_no, odds_array, methods))

    # Make labels for frame and graph
    population_labels = [str(population_size + i) for i in range(0, runs*step, step)] 
    
    # Make frame to check data
    df = pd.DataFrame(data, index = population_labels, columns = methods)
    pd.set_option('display.colheader_justify', 'center')
    df.index.name = 'Population size'
    print(df)

    # Plot generations/evaluations vs. population size for each method in seperate graph - different lines are odds
    fig, axs = plt.subplots(3,1, figsize=(15, 10), sharey=False)
    for i, ax in enumerate(axs.flat):
        ax.plot(population_labels, [data[:][j][i] for j in range(runs)])
        ax.set_title(str(methods[i]))
    # Format graph
    plt.setp(axs[:], ylabel='Fitness evaluations') #set identical y axis labels
    plt.setp(axs[len(methods) - 1], xlabel='Population size') #set common x axis label
    ax.legend(['Odds: '+str(odds) for odds in odds_array]) 
    plt.savefig("test.png")
    plt.show()


if __name__ == "__main__":
    main()
