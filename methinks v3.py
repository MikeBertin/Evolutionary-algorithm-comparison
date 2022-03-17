##!/usr/bin/env python
# coding: utf-8

# In[1]:


# Methinks evolutionary algorithm with and without crossover comparison with hillclimber


# In[2]:


import os
#print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
#print("PATH:", os.environ.get('PATH'))


# In[3]:


# Import libraries #
import numpy as np
import string
import pandas as pd
import random


# In[4]:


# Global variables
GOAL = False
EVALUATION = 0
ODDS = 1
np.random.seed()


# In[5]:


# Return a random [ascii + ' '] char
def random_char():
    return random.choice(string.ascii_letters + ' ')


# In[6]:


# Return random 1D array of random_char() of size = target string
def random_string(target):
    return [random_char() for letter in target]


# In[7]:


# Return 2D array of random char arrays
def make_population(population_size, target):
    return [random_string(target) for i in range(population_size)]    


# In[8]:


# Return string labels array
def make_labels(population_size):
    return ["Str "+ str(i+1) + ":" for i in range(population_size)]


# In[9]:


# Return Data frame to store population with index labels
def make_frame(population_size, target):
    return pd.DataFrame(make_population(population_size, target), index = make_labels(population_size))


# In[10]:


# Goal check
def goal_check(fitness, target, string_index, candidate):
    if fitness == len(target):
        print("\n   Goal: String", string_index, "wins!")
        return True
    else:
        return False


# In[11]:


# Compare strings - if identical set Goal=true - return fitness as int
def fitness_test(candidate, target, string_index):
    
    global EVALUATION
    global GOAL
    
    fitness = sum(candidate[i]==target[i] for i in range(len(target))) # Count identical elements of arrays
    EVALUATION += 1   # Count number of fitness evaluations
    
    print('   Fitness evaluations: %d' % EVALUATION, end='\r', flush=True)
    GOAL = goal_check(fitness, target, string_index, candidate) # Check goal condition
    
    return fitness


# In[12]:


# Group fitness - test fitness of each string and return frame with fitness values
def group_fitness_test(population, population_size, target):
    
    fitness_array = [fitness_test(population.iloc[i], target, i) for i in range(population_size)]
    
    return pd.DataFrame(fitness_array, index = make_labels(population_size), columns = ["Fitness"])


# In[13]:


# Mutate letters in string with probability 1/target string length
def mutate(individual, target):
    return [random_char() if random.random() < ODDS**-1 else individual[i] for i in range(len(target))]


# In[14]:


# Hill climber - mutate each individual and replace if mutant fitter than parent
def hillclimb(group, fitness, population_size, target):
    
    global GOAL
    
    for i in range(population_size):
        
        mutant = mutate(group.iloc[i], target) # Mutate ith member of group
        mutant_fitness = fitness_test(mutant, target, i) # Test fitness of ith member of group
        
        # Replace parent with mutant if fitness greater than stored value of parent fitness
        if  mutant_fitness >= fitness.iloc[i]['Fitness'] :
            group.iloc[i] = mutant
            fitness.iloc[i] = mutant_fitness
            
        if GOAL == True: return group, fitness # Check goal condition and return if true

    return group, fitness


# In[15]:


# GA with no crossover - select fittest of two random parents and mutate, replace least fit of two other randoms
def ga_no_cross(group, fitness, population_size, target):
    
    # Make four random ints
    random1 = np.random.randint(0,population_size)
    random2 = np.random.randint(0,population_size)
    random3 = np.random.randint(0,population_size)
    random4 = np.random.randint(0,population_size)
   
    # Select two random parents
    parent1 = group.iloc[random1]
    parent2 = group.iloc[random2]
    
    # Compare fitness values for parents and mutate the fittest
    if fitness.iloc[random1]['Fitness'] > fitness.iloc[random2]['Fitness'] :
        mutant = mutate(parent1, target)
        mutant_fitness = fitness_test(mutant, target, random1)
    else:
        mutant = mutate(parent2, target)
        mutant_fitness = fitness_test(mutant, target, random2)
    
    # Compare two other random individuals and replace the least fit with the mutant
    if  fitness.iloc[random3]['Fitness'] < fitness.iloc[random4]['Fitness'] :
        group.iloc[random3] = mutant
        fitness.iloc[random3] = mutant_fitness
    else: 
        group.iloc[random4] = mutant
        fitness.iloc[random4] = mutant_fitness
    
    return group, fitness


# In[16]:


# Crossover function - combine two parent strings randomly with 50:50 chance
def crossover(parent1, parent2):
    return [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]


# In[17]:


# GA with crossover
def ga_with_cross(group, fitness, population_size, target):
    
    # Make six random ints
    random1 = np.random.randint(0,population_size)
    random2 = np.random.randint(0,population_size)
    random3 = np.random.randint(0,population_size)
    random4 = np.random.randint(0,population_size)
    random5 = np.random.randint(0,population_size)
    random6 = np.random.randint(0,population_size)
    
    # Select four random parents
    parent1 = group.iloc[random1]
    parent2 = group.iloc[random2]
    parent3 = group.iloc[random3]
    parent4 = group.iloc[random4]
    
    # Compare fitness values for parents and chose the fittest
    if fitness.iloc[random1]['Fitness'] > fitness.iloc[random2]['Fitness'] :
        chosen_parent1 = parent1
    else:
        chosen_parent1 = parent2
    
    # Compare fitness values for parents and chose the fittest
    if fitness.iloc[random3]['Fitness'] > fitness.iloc[random4]['Fitness'] :
        chosen_parent2 = parent3
    else:
        chosen_parent2 = parent4
    
    # Make child using crossover and mutate 
    child = crossover(chosen_parent1, chosen_parent2)
    mutant_child = mutate(child, target)
    
    # Compare two other random individuals, replace the least fit with mutant child and update fitness array
    if  fitness.iloc[random5]['Fitness'] < fitness.iloc[random6]['Fitness'] :
        group.iloc[random5] = mutant_child
        fitness.iloc[random5] = fitness_test(mutant_child, target, random5)
    else: 
        group.iloc[random6] = mutant_child
        fitness.iloc[random6] = fitness_test(mutant_child, target, random6)
    
    return group, fitness


# In[18]:


# Initialise new random population and test group fitness
def initialise(population_size, target):
    
    new_group = make_frame(population_size, target) # Make new random population
    initial_fitness = group_fitness_test(new_group, population_size, target) # Test fitness of group
    
    return new_group, initial_fitness


# In[19]:


# Iterate until GOAL fitness condition or counter limit is satisfied

def evolve(method, population_size, target):
     
    global EVALUATION 
    global GOAL
    
    # Initialise variables
    EVALUATION = generation = 0
    GOAL = False # Set GOAL state false
    np.random.seed() # Seed pseudorandom number - needed?
    
    # Initialise new group and initial fitness
    group, fitness = initialise(population_size, target)

    while GOAL == False:
 
        if method == 'Hillclimb':
            group, fitness = hillclimb(group, fitness, population_size, target)
        elif method == 'GA':
            group, fitness = ga_no_cross(group, fitness, population_size, target)
        elif method == 'GA_with_crossover':
            group, fitness = ga_with_cross(group, fitness, population_size, target)
        else:
            break
            
        #generation += 1  # Generation counter - needed?
    
    return EVALUATION


# In[20]:


# Loop for each method and odds - call evolve function  - returns a 2D matrix

def run_average(population_size, target, runs, step, average_no, odds_array, methods):
        
    global ODDS
   
    run_averages = [[0 for x in range(len(odds_array))] for y in range(len(methods))]  # Make array of correct dimensions
    
    print('Population size:', population_size)
    
    # Loop through methods and odds
    for i, method in enumerate(methods):
        print(' Method:', method)
        for j, odds in enumerate(odds_array):
            
            ODDS = odds*len(target)
            print('  Mutation odds:', odds)
            # Average each evolve run at same: odds, pop size, method
            run_averages[i][j] = round(sum(evolve(method, population_size, target) for k in range(average_no))/ average_no)
            
    return run_averages


# In[21]:


# Collect 3D array of data varying population size, number of samples per average and odds

methods = ['GA', 'GA_with_crossover','Hillclimb'] # Methods of finding target
average_no = 5 # Number of identical tests to average
runs = 5
step = 100
odds_array = [0.5, 1, 2] # More or less likely relative to 1/target length
population_size = 500
target = 'methinks it is like a weasel' # Target can be adjusted here
print('Averages:', average_no, ' Runs:', runs, ' Step:', step,  ' Target:', target)

# Vary the population size and collect in an array
data = [run_average(population_size + i, target, runs, step, average_no, odds_array, methods) for i in range(0, runs*step, step)]


# In[391]:


# Make frame to check data
population_labels = [str(population_size + i) for i in range(0, runs*step, step)]
df = pd.DataFrame(data, index = population_labels, columns = methods)
pd.set_option('display.colheader_justify', 'center')
df.index.name = 'Population size'
print(df)


# In[ ]:


# Plot generations/evaluations vs. population size for each method in seperate graph - different lines are odds
fig, axs = plt.subplots(3,1, figsize=(15, 14), sharey=False)

for i, ax in enumerate(axs.flat):
    ax.plot(population_labels, [data[:][j][i] for j in range(runs)])
    ax.set_title(str(methods[i]))

plt.setp(axs[0], xlabel='Population size') #set common x axis label
plt.setp(axs[:], ylabel='Generations to find solution') #set identical y axis labels
ax.legend(['Odds: 0.5', 'Odds: 1','Odds: 2']) # set legend for odds
plt.show()

