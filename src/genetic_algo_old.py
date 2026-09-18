#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:26:10 2024

@author: drminor
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from water_gym import WaterGym
from water_gym import Actions, pumps, pumpq, Level
#from perform_evaluate import plot_performance


# Define parameters for the genetic algorithm
population_size = 20
num_generations = 10
mutation_rate = 0.1
crossover_rate = 0.8
num_parents = 2

# Define bounds for variables
lower_bound = -5
upper_bound = 5

# [self.rains[self.clock], 
#         cur_inflow, 
#         cur_outflow, 
#         cur_vol, 
#         cur_level],\
#         cur_reward, done,

# Define the objective function to be maximized



def objective_function(act, state, action_list):
    
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    w4 = 0.0
    
    # vol_reward:
    # 펌프량을 반영하데, 현재 수위를 고려  하여 수위가 높을 때 많이 퍼내면 높은 보상
    outflow_act = np.array(pumpq)[Actions[act]].sum() # 펌프량
    vol_reward = (outflow_act/np.array(pumpq).sum()) * (state[-1]/Level[-1])
    
    #print(state)
    
    # act_reward:
    # 변경 수를 반영하여 변경 범프 개수의 음수를 패널티로 부과
    first = action_list[-1]
    second = act
    no_changes = (np.array(first) & np.array(second)).sum() # 변경되지 않은 펌프의 개수
    act_reward = no_changes / len(Actions)
    
    # energy_reward:
    # 최대 펌핑시의 에너지에 대한 현재 펌핑의 에너지 소모량의 비를 이용
    # 소모량이 작을 수록 보상은 1에 가까워지고 최대 소비량에 가까워지면 0에 근접
    energy_reward = 1.0 - outflow_act / np.array(pumpq).sum()
    
    # excess_reward
    # 현재 유수지 수량보다 더 많은 물을 퍼내고자 할 때
    diff = state[3] - outflow_act
    excess_pump = 1.0 if diff < 0 else 0.0
    excess_pump_penalty = -10.0
    
    reward = w1 * vol_reward\
            + w2 * act_reward\
            + w3 * energy_reward\
            + w4 * excess_pump * excess_pump_penalty
    
    return reward


# Initialize population
def initialize_population(population_size):
    population = []
    i = 0
    while True :
        temp_p = []
        for j in range(pumps):
            temp_p.append(bool(np.random.randint(0, 2, size=1)))
        if temp_p in Actions:
            population.append(temp_p)
            i += 1
            if i >= population_size:
                break
    
    return np.array(population)


# Calculate fitness for each individual in the population
def calculate_fitness(population, state, action_list): #outflow, vol, level:
    # print(population)
    # print(population[0])
    # print(action_transform(population[0]))
    fitness_values = [objective_function(action_transform(individual), state, action_list) \
                      for individual in population]
        
    return np.array(fitness_values)


# Select parents based on tournament selection
def select_parents(population, fitness_values, num_parents):
    """
    size 개의 해를 랜덤하게 선택한 후 이중 fitness가 가장 큰 해를 num_parents개 만큼 반환

    Parameters
    ----------
    population : TYPE
        DESCRIPTION.
    fitness_values : TYPE
        DESCRIPTION.
    num_parents : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    selected_parents = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(len(population), size=3, replace=False) 
        tournament_fitness = fitness_values[tournament_indices]
        selected_parents.append(population[tournament_indices[np.argmax(tournament_fitness)]])
    
    return np.array(selected_parents)


# Perform crossover to create offspring
def crossover(parents, crossover_rate):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parents[0])-1)
        offspring1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
        offspring2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
        if (list(offspring1) in Actions) and (list(offspring2) in Actions):
            return offspring1, offspring2
        
    return parents[0], parents[1]


# Perform mutation
def mutate(offspring, mutation_rate):
    original_offspring = copy.deepcopy(offspring)
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            if offspring[i] == True:
                offspring[i] = False
            else:
                offspring[i] = True
    if list(offspring) in Actions:            
        return offspring
    else:
        return original_offspring


# Genetic algorithm main function
def genetic_algorithm(state, action_list):
    population = initialize_population(population_size)
    
    for generation in range(num_generations):
        fitness_values = calculate_fitness(population, state, action_list)
        
        # Select parents
        parents = select_parents(population, fitness_values, num_parents)
        
        new_population = []
        for i in range(0, population_size, 2):
            # Perform crossover
            offspring1, offspring2 = crossover(parents, crossover_rate) 
            # Perform mutation
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            
            
            new_population.extend([offspring1, offspring2])
            parents = select_parents(population, fitness_values, num_parents)
        
        population = np.array(new_population)
       
    # Find the best individual
    fitness_values = calculate_fitness(population, state, action_list)
    best_index = np.argmax(fitness_values)
    best_action = population[best_index]
    # print(best_index)
    # print(fitness_values[:5])
    # print(population[:5])
    best_action = action_transform(best_action)
    #print(best_action)
    best_fitness = fitness_values[best_index]
    
    return best_action


# Transform the best action
def action_transform(action):
    """
    유전자 알고리즘의 결과 해를 Actions의 해로 변경한 후 해당 action index 반환

    Parameters
    ----------
    action : TYPE
        DESCRIPTION.

    Returns
    -------
    best_action : TYPE
        DESCRIPTION.

    """
    # mask100 = np.array([True, True, True, False, False])
    # mask170 = np.array([False, False, False, True, True])
    # n100 = np.sum(action & mask100)
    # n170 = np.sum(action & mask170)
    # best_action = np.array([False]*5)
    # best_action[:n100] = True
    # best_action[3:3+n170] = True
    # #print(best_action, end=': ')
    # for i in range(len(Actions)):
    #     if np.sum(np.array(Actions[i]) ^ best_action) == 0:
    #         best_action = i
    #         break
    # #print(best_action, end=' - ')
    
    best_action = Actions.index(list(action))
    
    return best_action


# Run the genetic algorithm
def run_genetic_algo(test_inp, minutes):    
    gym = WaterGym(test_inp, minutes=minutes)
    init_state_, reward, done, info = gym.reset()
    
    action_list = [0] # to calculate the number of on/off switchs 
    state_list = [init_state_]
    reward_list = [reward]
    info_list = [info]
    
    # g_times = []
    # g_inflows = []
    # g_outflows = []
    # g_rainfalls = []
    # g_before_elevations = []
    # g_after_elevations = []
    # g_before_volumes =[]
    # g_after_volumes = []
    # g_changes = []
    # g_actions = []
    
    action = genetic_algorithm(init_state_, action_list)
    action_list.append(action)
    
    while (1):
        state_, reward, done, info = gym.step(action)
        
        # for performance evaluations
        state_list.append(state_)
        reward_list.append(reward)
        info_list.append(info)
        
        if done == True:
            break
    
        action = genetic_algorithm(state_, action_list)
        #print(action)
        action_list.append(action)
        
        
    return action_list, state_list, reward_list, info_list   


def plot_performance(ftitle, state_list, actions, 
                     kind=['rain', 'inflow', 'outflow', 'vol', 'level']):
    rain_list = []
    inflow_list = []
    outflow_list = []
    vol_list = []
    level_list = []
    
    
    for state in state_list:
        rain, inflow, outflow, vol, level = state
        rain_list.append(rain)
        inflow_list.append(inflow)
        outflow_list.append(outflow)
        vol_list.append(vol)
        level_list.append(level)
    
    kind_dic = {'rain': rain_list, 'inflow': inflow_list, 
                'outflow': outflow_list, 'vol': vol_list, 'level': level_list}
    color = ['black', 'blue', 'red', 'green', 'yellow']
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    fig.suptitle(ftitle, y=0.92, fontsize=20)
    for i in range(len(kind)-1):
        ax[0].plot(range(len(kind_dic[kind[i]])), 
                kind_dic[kind[i]], 
                color=color[i],
                label=list(kind_dic.keys())[i])
    ax[0].legend()
    
    ax[1].plot(range(len(actions)),
                 kind_dic['rain'],
                 color=color[0],
                 label='rain/m')
    ax[1].plot(range(len(actions)),
                 kind_dic['level'],
                 color=color[1],
                 label='level')
    ax[1].scatter(range(len(actions)),
                 actions,
                 color=color[2],
                 label='actions')
    ax[1].legend()
    
    fig.show()


if __name__ == '__main__':
    test_list = ['data/gasan/30year/30yr_0720m_h954.inp'] #'0060m_h150.inp']
    action_list, state_list, reward_list, info_list = run_genetic_algo(test_list[0], minutes=2)
    plt.figure()
    fig_title =  test_list[0].split('/')[-1]
    plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])
