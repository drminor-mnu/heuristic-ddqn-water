#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:26:10 2024

genetic_algo.py  

@author: drminor
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from water_gym import WaterGym
from water_gym import (
    Actions0, Actions1, pumps, pumpq, Level, Volume, switching_stability,
)
#from perform_evaluate import plot_performance


# Define parameters for the genetic algorithm
population_size = 100
num_generations = 100
mutation_rate = 0.01
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

class GeneticAlgorithm():
    def __init__(self, **params):
        self.minutes = params.get('minutes', 2)
        self.weights = params.get('weights', [1, 1, 1, 0])
        self.w1, self.w2, self.w3, self.w4 = self.weights
        self.act_type = params.get('act_type', 0)
        if self.act_type == 0:
            self.Actions = Actions0
        else:
            self.Actions = Actions1

    @staticmethod
    def _volume_to_level(volume):
        """Convert volume to level using the same interpolation as WaterGym."""
        if volume < 0:
            return Level[0]
        for index in range(len(Volume) - 1):
            if volume < Volume[index + 1]:
                break
        else:
            index += 1
        if index < len(Volume) - 1:
            return Level[index] + (
                (Level[index + 1] - Level[index])
                / (Volume[index + 1] - Volume[index])
                * (volume - Volume[index])
            )
        return Level[index]

    def objective_function(self, act, state, action_list, next_inflow=0.0):
        """Predict the reward that WaterGym will return for the next step."""
        pump_rate = np.array(pumpq)[self.Actions[act]].sum()
        attempted_outflow = pump_rate * self.minutes
        available_volume = state[3] + next_inflow
        excess_pump = 1.0 if attempted_outflow > available_volume else 0.0
        actual_outflow = min(attempted_outflow, available_volume)
        next_volume = max(available_volume - attempted_outflow, 0.0)
        next_level = self._volume_to_level(next_volume)

        # Water-level reward -- manuscript Eq. (13): R_level = (Q(a_t)/Q_max) . (h_t/h_max),
        # where h_t/h_max is the *current* normalized water level (state[-1]), NOT the
        # level after pumping. Using next_level (level after the water is pumped out) is a
        # regression: it makes vol_reward shrink the more effectively the pump drains the
        # basin, so the 1-step objective stops rewarding pump initiation and GA/PSO lock
        # onto action 0 (see docs/E00 section 2-3, results/genetic_*_20260724.csv which
        # matched the manuscript used the current-level form).  Reverted 2026-08-28.
        # NOTE: the leading factor here is actual_outflow/Q_max (capped, x minutes), not
        # the raw Q(a_t)/Q_max of Eq. (13) -- left unchanged per instruction; separate item.
        vol_reward = (actual_outflow/np.array(pumpq).sum()) * (state[-1]/Level[-1])
        #vol_reward = 1. - (state[-1]-Level[-1])/(Level[-1]-Level[0])
        
        # act_reward:
        # 변경 수를 반영하여 변경 범프 개수의 음수를 패널티로 부과
        first = action_list[-1]
        second = act
        #print(first, second)
        act_reward = switching_stability(
            self.Actions[first], self.Actions[second]
        )
        #print(no_changes, act_reward)
        # energy_reward:
        # 최대 펌핑시의 에너지에 대한 현재 펌핑의 에너지 소모량의 비를 이용
        # 소모량이 작을 수록 보상은 1에 가까워지고 최대 소비량에 가까워지면 0에 근접
        energy_reward = 1.0 - pump_rate / np.array(pumpq).sum()

        excess_pump_penalty = -1.0
        
        reward = self.w1 * vol_reward\
                + self.w2 * act_reward\
                + self.w3 * energy_reward\
                + self.w4 * excess_pump * excess_pump_penalty
        
        return reward
    
    
    # Initialize population
    def initialize_population(self, population_size):
        population = []
        i = 0
        while True :
            temp_p = []
            for j in range(pumps):
                temp_p.append(bool(np.random.randint(0, 2, size=1)))
            if temp_p in self.Actions:
                population.append(temp_p)
                i += 1
                if i >= population_size:
                    break
        
        return np.array(population)
    
    
    # Calculate fitness for each individual in the population
    def calculate_fitness(self, population, state, action_list, next_inflow=0.0): #outflow, vol, level:
        # A population contains many duplicates because the action space is small.
        # Cache deterministic objective values within this call so the GA search
        # and random-number stream remain unchanged while avoiding repeated work.
        action_lookup = {
            tuple(bool(value) for value in action): index
            for index, action in enumerate(self.Actions)
        }
        cached_fitness = {}
        fitness_values = []
        for individual in population:
            key = tuple(bool(value) for value in individual)
            action = action_lookup[key]
            if action not in cached_fitness:
                cached_fitness[action] = self.objective_function(
                    action, state, action_list, next_inflow
                )
            fitness_values.append(cached_fitness[action])
        return np.array(fitness_values)
    
    
    # Select parents based on tournament selection
    def select_parents(self, population, fitness_values, num_parents):
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
    def crossover(self, parents, crossover_rate):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parents[0])-1)
            offspring1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
            offspring2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
            if (list(offspring1) in self.Actions) and (list(offspring2) in self.Actions):
                return offspring1, offspring2
            
        return parents[0], parents[1]
    
    
    # Perform mutation
    def mutate(self, offspring, mutation_rate):
        original_offspring = copy.deepcopy(offspring)
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                if offspring[i] == True:
                    offspring[i] = False
                else:
                    offspring[i] = True
        if list(offspring) in self.Actions:            
            return offspring
        else:
            return original_offspring
    
    
    # Genetic algorithm main function
    def genetic_algorithm(self, state, action_list, next_inflow=0.0):
    
        # 종료 조건 파라미터
        patience = 10
        min_improvement = 1e-6
        best_fitness_so_far = -np.inf
        no_improve_count = 0
            
        population = self.initialize_population(population_size)
        
        for generation in range(num_generations):
            fitness_values = self.calculate_fitness(population, state, action_list, next_inflow)
                    
            # 현재 세대 최고 fitness
            current_best_fitness = np.max(fitness_values)
            # print(f"Early stopping at generation {generation}, {current_best_fitness}")
    
            # 개선 여부 확인
            if current_best_fitness > best_fitness_so_far + min_improvement:
                best_fitness_so_far = current_best_fitness
                no_improve_count = 0
            else:
                no_improve_count += 1
    
            # 종료 조건: 일정 세대 도안 개선 없으면 종료
            if no_improve_count >= patience:
                #print(f"Early stopping at generation {generation}, {current_best_fitness}")
                break
            
            # Select parents
            parents = self.select_parents(population, fitness_values, num_parents)
            
            new_population = []
            for i in range(0, population_size, 2):
                # Perform crossover
                offspring1, offspring2 = self.crossover(parents, crossover_rate) 
                # Perform mutation
                offspring1 = self.mutate(offspring1, mutation_rate)
                offspring2 = self.mutate(offspring2, mutation_rate)
                
                
                new_population.extend([offspring1, offspring2])
                parents = self.select_parents(population, fitness_values, num_parents)
            
            population = np.array(new_population)
           
        # Find the best individual
        fitness_values = self.calculate_fitness(population, state, action_list, next_inflow)
        best_index = np.argmax(fitness_values)
        best_action = population[best_index]
        # print(best_index)
        # print(fitness_values[:5])
        # print(population[:5])
        best_action = self.action_transform(best_action)
        #print(best_action)
        best_fitness = fitness_values[best_index]
        
        return best_action
    
    
    # Transform the best action
    def action_transform(self, action):
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
        
        best_action = self.Actions.index(list(action))
        
        return best_action
    
    
    # Run the genetic algorithm
    def run_genetic_algo(self, test_inp):    
        gym = WaterGym(test_inp, minutes=self.minutes,
                       weights=self.weights, act_type=self.act_type)
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
        
        action = self.genetic_algorithm(
            init_state_, action_list, next_inflow=gym.outfalls[gym.clock + 1]
        )
        action_list.append(action)
        
        while (1):
            state_, reward, done, info = gym.step(action)
            
            # for performance evaluations
            state_list.append(state_)
            reward_list.append(reward)
            info_list.append(info)
            
            if done == True:
                break
        
            action = self.genetic_algorithm(
                state_, action_list, next_inflow=gym.outfalls[gym.clock + 1]
            )
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
    params = {'minutes':2, 'epochs':1, 'weights': [1.0, 1.0, 1.0, 1.0], 'act_type':0}
    ga = GeneticAlgorithm(**params)
    action_list, state_list, reward_list, info_list = ga.run_genetic_algo(test_list[0])
    plt.figure()
    fig_title =  test_list[0].split('/')[-1]
    plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])
