#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:06:13 2024



@author: drminor
"""

import re 
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import sys

from water_gym import Actions0, Actions1, pumpq
import water_gym
import dqn_from_demon_v1
import genetic_algo
from genetic_algo import GeneticAlgorithm
import man_policy

from plot_style import set_plot_style

set_plot_style()


random.seed(time.time())

def stratified_split_data(years, durations, trate=0.8, vrate=0.0, stratify=True):
    """
    data/gasan/10year/10yr_0010m_h055.inp
    year_freq=[10, 20, 30, 50, 80, 100],
    duration=[0010, 0060, 0120, 0180, 0240, 0360, 0540, 0720, 1080, 1440]
    
    Parameters
    ----------
    years : string list
        years = ['10', '20', '30', '50', '80', '100'],
    minutes : string list
        minutes = ['0010', '0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']

    Returns
    -------
    inp_list : TYPE
        DESCRIPTION.
    train_inps : TYPE
        DESCRIPTION.
    valid_inps : TYPE
        DESCRIPTION.
    test_inps : TYPE
        DESCRIPTION.

    """
    with open('../data/selected_inp.txt', 'r') as fd:
        inp_list = fd.readlines()
        inp_list = [f.strip() for f in inp_list] 
        inp_list = [f for f in inp_list if (re.findall(r'\d+', f.split('/')[-1])[0] in years)  
                    and (re.findall(r'\d+', f.split('/')[-1])[1] in durations)]
    
    inp_list.sort()
    train_inps = []
    valid_inps = []
    test_inps = []
    if stratify:
        for y in years:
            for m in durations:
                data = [f for f in inp_list 
                            if re.findall(r'\d+', f.split('/')[-1])[0] == y
                            and re.findall(r'\d+', f.split('/')[-1])[1] == m]
                random.shuffle(data)
                nsample = len(data) # the number of total sample
                ntrate = int(nsample * trate) # the number of training sample
                nvrate = int(nsample * vrate) # the number of validation sample
                
                train_inps.extend(data[:ntrate])
                valid_inps.extend(data[ntrate:ntrate+nvrate])
                test_inps.extend(data[ntrate+nvrate:])
    random.shuffle(train_inps)
    random.shuffle(valid_inps)
    random.shuffle(test_inps)
    
    return inp_list, train_inps, valid_inps, test_inps


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
        

def cal_perform(state_list, actions):
    """
    최고 수위
    평균 수위
    펌프 조합 변경수
    펌프 용량 미만의 저수 빗물 펌프 시도 횟수
    state_list => [rains, cur_inflow, cur_outflow, cur_vol, cur_level]
    
    Returns
    -------
    None.

    """
   
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


def count_changes(actions, act_type):
    """
    펌프 변경 수 계산

    Parameters
    ----------
    actions : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    Actions = Actions0 if (act_type == 0) else Actions1
    
    changes = 0
    for i in range(len(actions)-1):
        first = Actions[actions[i]]
        second = Actions[actions[i+1]]
        changes += (np.array(first) ^ np.array(second)).sum()
    
    return changes


def count_pumps(actions, act_type)->tuple:
    """
    사용된 2가지 펌프의 갯수 반환
    
    pumpq = (100, 100, 100, 170, 170) # unit: cmm(cubic meter per minute), outflow rate of pump
    pumps = 5

    Actions = [[False, False, False, False, False],
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True]] 

    Actions1 = [[False, False, False, False, False],
                [True, False, False, False, False],
                [False, False, False, True, False],
                [True, True, False, False, False],
                [True, False, False, True, False],
                [True, True, True, False, False],
                [False, False, False, True, True],
                [True, True, False, True, False],
                [True, False, False, True, True],
                [True, True, True, True, False],
                [True, True, False, True, True],
                [True, True, True, True, True]] 
    Parameters
    ----------
    actions : TYPE
        DESCRIPTION.

    Returns
    -------
    num_pumps100 : int
        the number of turnning on pump 100 model
    num_pumps170 : int 
        the number of turnning on pump 170 model
    npumps : np.array
        the number of tunning on each pump
    """
    
    Actions = Actions0 if (act_type == 0) else Actions1
    
    npumps = np.zeros(len(Actions[0]), dtype=int)
    act_list = list(range(len(Actions)))
    for action in actions:
        if action in act_list:
            npumps += np.array(Actions[action])
        else:
            print("perform_evaluation 233: not in action list")
            raise SystemExit
            
    num_pump100 = sum(npumps[:3])
    num_pump170 = sum(npumps[3:])
            
    return num_pump100, num_pump170, npumps


def plot_reward(reward1, reward2):
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(30,12)
    ax[0].plot(reward1, label='Regular DDQN')
    ax[1].plot(reward2, label='GA guided DDQN')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()


def evaluate_dpn_gru(train_inps, test_inps, 
                     years, durations,
                     train:bool=True,
                     model_path:str=None,
                     **params):
    """
    The performance evalutaion of Deep Q-Network with a specific DL agent

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    train_inps :
    
    test_inps :
        
    minutes : int, optional
        DESCRIPTION. The default is 2.
    epochs : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    
    minutes = params.get('minutes', 2)
    weights = params.get('weights', [1.,1.,1.,0.])
    elapsed = 0.0 # for training time
    act_type = params.get('act_type', 0)
    #losses = None
    
    Actions = Actions0 if (act_type == 0) else Actions1
    
    if train:
        print(f'trains: {len(train_inps)}, tests: {len(test_inps)}') 
        start = time.perf_counter()
        model, basic_losses, basic_rewards, \
            = dqn_from_demon_v1.train(train_inps, model_path=model_path, **params)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Elapsed time for ddqn guided by GA: {elapsed} seconds')
        print(f"training is completed -> {model_path}")
    else:
        print(f'tests: {len(test_inps)}')
        print(f'model -> {model_path}')
        model = None
        
        
    # for seek the average highest level
    highest_levels = np.zeros(shape=(len(years), len(durations)))
    counts = np.full((len(years), len(durations)), 1e-10) # sample count
    
    # count flooding cases
    counts_flood = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pump changes
    pump_changes = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pumps used
    counts_pump100 = np.zeros(shape=(len(years), len(durations)))
    counts_pump170 = np.zeros(shape=(len(years), len(durations)))
    cnt_pumps = np.zeros(shape=(len(years), len(durations), len(Actions[0])))
    
    # count the over dumping
    counts_overpump = np.zeros(shape=(len(years), len(durations)))
    
    dqn_test_rewards = []
    for i, test_inp in enumerate(test_inps):
        print(f'\rGRU Test #{i}', end='', flush=True)
        _, actions, states, rewards, infos \
            = dqn_from_demon_v1.test_model(model, test_inp, minutes=minutes, 
                                           model_path=model_path, 
                                           weights=weights, 
                                           act_type=act_type)
        
        # Reward calculation of the i-th test data
        reward = np.array(rewards).sum() # total reward of a test scenario
        dqn_test_rewards.append(reward)
        
        # find ids and increase the sample count by one
        id_year = years.index(re.findall(r'\d+', test_inp.split('/')[-1])[0])
        id_du = durations.index(re.findall(r'\d+', test_inp.split('/')[-1])[1])
        counts[id_year][id_du] += 1
        
        # compute the number of pump changes
        pump_changes[id_year][id_du] += count_changes(actions, act_type)
        
        # compute the number of pumps
        pump100, pump170, npump = count_pumps(actions, act_type)
        counts_pump100[id_year][id_du] += pump100
        counts_pump170[id_year][id_du] += pump170
        cnt_pumps[id_year][id_du] += npump
        
        # find the hightest level for a scenario
        highest = -1
        for state in states:
            if highest < state[-1]:
                highest = state[-1]
        highest_levels[id_year][id_du] += highest
        
        # if the highest level is greater than or equal 10M then flooing is occurred
        if highest >= water_gym.Level[-1]:
            counts_flood[id_year][id_du] += 1
        
        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        counts_overpump[id_year][id_du] = np.sum(ainfos[:, -1])
        
    avg_highest_levels = highest_levels / counts
    avg_pump_changes = pump_changes / counts
    avg_pump100 = counts_pump100 / counts
    avg_pump170 = counts_pump170 / counts
    avg_overpump = counts_overpump / counts
    avg_cnt_pumps = cnt_pumps/counts[..., np.newaxis]
    
    #str_w = '_'.join(['w'+str(w) for w in weights])
    str_w = model_path.split('/')[-1].replace(".pkl", '')
    save_results(avg_highest_levels, avg_pump_changes, avg_pump100, avg_pump170, \
                 avg_cnt_pumps, avg_overpump, counts_flood, counts, file=f'{str_w}')
    # print(avg_highest_levels)
    # print(avg_pump_changes)
    # print(avg_pump100)
    # print(avg_pump170)
    # print(avg_cnt_pumps)
    # print(avg_overpump)
    # print(counts_flood)
    
    return basic_losses, basic_rewards, dqn_test_rewards, elapsed


def evaluate_dpn_guided_GA(train_inps, test_inps, 
                     years, durations,
                     train:bool=True,
                     model_path:str=None,
                     **params):
    """
    The performance evalutaion of Deep Q-Network with a specific DL agent

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    train_inps :
    
    test_inps :
        
    minutes : int, optional
        DESCRIPTION. The default is 2.
    epochs : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    minutes = params.get('minutes', 2)
    weights = params.get('weights', [1.,1.,1.,0.])
    elapsed = 0.0 # for training time
    act_type = params.get('act_type', 0)
    # losses = None
    Actions = Actions0 if (act_type == 0) else Actions1
    
    if train:
        print(f'trains: {len(train_inps)}, tests: {len(test_inps)}') 
        start = time.perf_counter()
        model, ga_losses, ga_rewards, \
            = dqn_from_demon_v1.train_guided_byGA(train_inps, model_path, **params)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Elapsed time for ddqn guided by GA: {elapsed} seconds')
        print(f"training is completed -> {model_path}")
    else:
        print(f'tests: {len(test_inps)}')
        print(f'model -> {model_path}')
        model = None
        
        
    # for seek the average highest level
    highest_levels = np.zeros(shape=(len(years), len(durations)))
    counts = np.full((len(years), len(durations)), 1e-10) # sample count
    
    # count flooding cases
    counts_flood = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pump changes
    pump_changes = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pumps used
    counts_pump100 = np.zeros(shape=(len(years), len(durations)))
    counts_pump170 = np.zeros(shape=(len(years), len(durations)))
    cnt_pumps = np.zeros(shape=(len(years), len(durations), len(Actions[0])))
    
    # count the over dumping
    counts_overpump = np.zeros(shape=(len(years), len(durations)))
    
    ga_test_rewards = []
    for i, test_inp in enumerate(test_inps):
        print(f'\rDQN guided by GA Test #{i}', end='', flush=True)
        _, actions, states, rewards, infos \
            = dqn_from_demon_v1.test_model(model, test_inp, minutes=minutes, 
                                           model_path=model_path, weights=weights,
                                           act_type=act_type)
        
        # Reward calculation of the i-th test data
        reward = np.array(rewards).sum() # total reward of a test scenario
        ga_test_rewards.append(reward)
        
        # find ids and increase the sample count by one
        id_year = years.index(re.findall(r'\d+', test_inp.split('/')[-1])[0])
        id_du = durations.index(re.findall(r'\d+', test_inp.split('/')[-1])[1])
        counts[id_year][id_du] += 1
        
        # compute the number of pump changes
        pump_changes[id_year][id_du] += count_changes(actions, act_type)
        
        # compute the number of pumps
        pump100, pump170, npump = count_pumps(actions, act_type)
        counts_pump100[id_year][id_du] += pump100
        counts_pump170[id_year][id_du] += pump170
        cnt_pumps[id_year][id_du] += npump
        
        # find the hightest level for a scenario
        highest = -1
        for state in states:
            if highest < state[-1]:
                highest = state[-1]
        highest_levels[id_year][id_du] += highest
        
        # if the highest level is greater than or equal 10M then flooing is occurred
        if highest >= water_gym.Level[-1]:
            counts_flood[id_year][id_du] += 1
        
        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        counts_overpump[id_year][id_du] = np.sum(ainfos[:, -1])
        
    avg_highest_levels = highest_levels / counts
    avg_pump_changes = pump_changes / counts
    avg_pump100 = counts_pump100 / counts
    avg_pump170 = counts_pump170 / counts
    avg_overpump = counts_overpump / counts
    avg_cnt_pumps = cnt_pumps/counts[..., np.newaxis]

    
    #str_w = '_'.join(['w'+str(w) for w in weights])
    str_w = model_path.split('/')[-1].replace(".pkl", '')
    save_results(avg_highest_levels, 
                 avg_pump_changes, avg_pump100, avg_pump170, 
                 avg_cnt_pumps, avg_overpump, counts_flood, counts, file=f'{str_w}')
    # print(avg_highest_levels)
    # print(avg_pump_changes)
    # print(avg_pump100)
    # print(avg_pump170)
    # print(avg_cnt_pumps)
    # print(avg_overpump)
    # print(counts_flood)
    
    return ga_losses, ga_rewards, ga_test_rewards, elapsed


def evaluate_genetic_algo(test_inps, 
                     years, durations,
                     **params):
    """
    The performance evalutaion of Deep Q-Network with a specific DL agent

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    train_inps :
    
    test_inps :
        
    minutes : int, optional
        DESCRIPTION. The default is 2.
    epochs : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    
    minutes = params.get("minutes", 2)
    weights = params.get("weights", [1,1,1,0])
    act_type = params.get("act_type", 0)
    
    Actions = Actions0 if (act_type == 0) else Actions1
    
    print(f'tests: {len(test_inps)}')  
        
    # for seek the average highest level
    highest_levels = np.zeros(shape=(len(years), len(durations)))
    counts = np.full((len(years), len(durations)), 1e-10)
    
    # count flooding cases
    counts_flood = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pump changes
    pump_changes = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pumps used
    counts_pump100 = np.zeros(shape=(len(years), len(durations)))
    counts_pump170 = np.zeros(shape=(len(years), len(durations)))
    cnt_pumps = np.zeros(shape=(len(years), len(durations), len(Actions[0])))
    
    # count the over dumping
    counts_overpump = np.zeros(shape=(len(years), len(durations)))
    
    for i, test_inp in enumerate(test_inps):
        print(f'\rGenetic Test #{i}', end='', flush=True)
        GA = genetic_algo.GeneticAlgorithm(**params)
        actions, states, rewards, infos = GA.run_genetic_algo(test_inp)
            
        # find ids and increase the sample count by one
        id_year = years.index(re.findall(r'\d+', test_inp.split('/')[-1])[0])
        id_du = durations.index(re.findall(r'\d+', test_inp.split('/')[-1])[1])
        counts[id_year][id_du] += 1
        
        # compute the number of pump changes
        pump_changes[id_year][id_du] += count_changes(actions, act_type)
        
        # compute the number of pumps
        pump100, pump170, npump = count_pumps(actions, act_type)
        counts_pump100[id_year][id_du] += pump100
        counts_pump170[id_year][id_du] += pump170
        cnt_pumps[id_year][id_du] += npump
        
        # find the hightest level for a scenario
        highest = -1
        for state in states:
            if highest < state[-1]:
                highest = state[-1]
        highest_levels[id_year][id_du] += highest
        
        # if the highest level is greater than or equal 10M then flooing is occurred
        if highest >= water_gym.Level[-1]:
            counts_flood[id_year][id_du] += 1
        
        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        counts_overpump[id_year][id_du] = np.sum(ainfos[:, -1])
    
    print()
    avg_highest_levels = highest_levels / counts
    avg_pump_changes = pump_changes / counts
    avg_pump100 = counts_pump100 / counts
    avg_pump170 = counts_pump170 / counts
    avg_overpump = counts_overpump / counts
    avg_cnt_pumps = cnt_pumps/counts[..., np.newaxis]
    
    str_w = '_'.join(['w'+str(w) for w in weights])
    save_results(avg_highest_levels, 
                 avg_pump_changes, avg_pump100, avg_pump170, 
                 avg_cnt_pumps, avg_overpump, counts_flood, counts, file=f'genetic_{str_w}')
    print(avg_highest_levels)
    print(avg_pump_changes)
    print(avg_pump100)
    print(avg_pump170)
    print(avg_cnt_pumps)
    print(avg_overpump)
    print(counts_flood)
    

def evaluate_man_policy(test_inps, 
                     years, durations,
                     **params):
    """
    The performance evalutaion of Deep Q-Network with a specific DL agent

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    train_inps :
    
    test_inps :
        
    minutes : int, optional
        DESCRIPTION. The default is 2.
    epochs : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    
    minutes = params.get('minutes', 2)
    weights = params.get('weights', [1,1,1,0])
    act_type = params.get('act_type', 0)
    
    Actions = Actions0 if (act_type == 0) else Actions1
    
    print(f'tests: {len(test_inps)}')  
        
    # for seek the average highest level
    highest_levels = np.zeros(shape=(len(years), len(durations)))
    counts = np.full((len(years), len(durations)), 1e-10)
    
    # count flooding cases
    counts_flood = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pump changes
    pump_changes = np.zeros(shape=(len(years), len(durations)))
    
    # count the number of pumps used
    counts_pump100 = np.zeros(shape=(len(years), len(durations)))
    counts_pump170 = np.zeros(shape=(len(years), len(durations)))
    cnt_pumps = np.zeros(shape=(len(years), len(durations), len(Actions[0])))
    
    # count the over dumping
    counts_overpump = np.zeros(shape=(len(years), len(durations)))
    
    for i, test_inp in enumerate(test_inps):
        print(f'\rMan Test #{i}', end='', flush=True)
        actions, states, rewards, infos = man_policy.operation(test_inp, minutes)
            
        # find ids and increase the sample count by one
        id_year = years.index(re.findall(r'\d+', test_inp.split('/')[-1])[0])
        id_du = durations.index(re.findall(r'\d+', test_inp.split('/')[-1])[1])
        counts[id_year][id_du] += 1
        
        # compute the number of pump changes
        pump_changes[id_year][id_du] += count_changes(actions, act_type)
        
        # compute the number of pumps
        pump100, pump170, npump = count_pumps(actions, act_type)
        counts_pump100[id_year][id_du] += pump100
        counts_pump170[id_year][id_du] += pump170
        cnt_pumps[id_year][id_du] += npump
        
        # find the hightest level for a scenario
        highest = -1
        for state in states:
            if highest < state[-1]:
                highest = state[-1]
        highest_levels[id_year][id_du] += highest
        
        # if the highest level is greater than or equal 10M then flooing is occurred
        if highest >= water_gym.Level[-1]:
            counts_flood[id_year][id_du] += 1
        
        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        counts_overpump[id_year][id_du] = np.sum(ainfos[:, -1])
    
    print()
    avg_highest_levels = highest_levels / counts
    avg_pump_changes = pump_changes / counts
    avg_pump100 = counts_pump100 / counts
    avg_pump170 = counts_pump170 / counts
    avg_overpump = counts_overpump / counts
    avg_cnt_pumps = cnt_pumps/counts[..., np.newaxis]
    
    str_w = '_'.join(['w'+str(w) for w in weights])
    save_results(avg_highest_levels, 
                 avg_pump_changes, avg_pump100, avg_pump170, 
                 avg_cnt_pumps, avg_overpump, counts_flood, counts, file=f'man_policy_{str_w}')
    print(avg_highest_levels)
    print(avg_pump_changes)
    print(avg_pump100)
    print(avg_pump170)
    print(avg_cnt_pumps)
    print(avg_overpump)
    print(counts_flood)


def save_results(level, change, pump100, pump170, npumps,
                 overpump, flood, count, file:str='results'):
    """
    전체 테스트 데이터((year, duration) 별)를 대상으로 최고수위평균, 펌프변경수평균, 펌프100사용수평규,
    펌프170사용수평균, 과다펌핑평균, 홍수발생수, 시나리오수 반환

    Parameters
    ----------
    level : TYPE
        DESCRIPTION.
    change : TYPE
        DESCRIPTION.
    pump100 : TYPE
        DESCRIPTION.
    pump170 : TYPE
        DESCRIPTION.
    overpump : TYPE
        DESCRIPTION.
    flood : TYPE
        DESCRIPTION.
    count : TYPE
        DESCRIPTION.
    file : str, optional
        DESCRIPTION. The default is 'results'.

    Returns
    -------
    None.

    """
    fname = f'../results/{file}.csv'
    fun = lambda x: f'{x:.3f}' # long float type number to 0.ddd format string
    vec_fun = np.vectorize(fun)
    
    with open(fname, 'w') as fd:
        # the number of samples for each experiment combination(year, duration)
        for row in vec_fun(count):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        # the water level
        for row in vec_fun(level):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        # the number of on/off switchs
        for row in vec_fun(change):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        # the number of usages of pump100
        for row in vec_fun(pump100):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        # the  number of usages of pump170
        for row in vec_fun(pump170):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        # the number of each pumps in pumps
        npumps = vec_fun(npumps)
        for y in npumps:
            for f in y:
                value = ','.join(f)
                fd.write(value+'\n')
            fd.write('\n')
        fd.write('\n')
        # the number of dry runs
        for row in vec_fun(overpump):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        # the number of flood occurances 
        for row in vec_fun(flood):
            value = ','.join(row)
            fd.write(value+'\n')
        

if __name__ == '__main__':
    ### Model parameters ###
    # actions = Actions
    # nstates = 3 # how to many state data use (the number of previous states: 2 + current state: 1)
    # num_layers = 3 # the number of layers of GRU model
    # hidden_dim = 32 # ther number of nodes for a layer
    
    """
    state <= [self.rains[self.clock], 
            cur_inflow, 
            cur_outflow, 
            cur_vol, 
            cur_level]
    """
    
    years = ['10', '20', '30', '50', '80', '100']
    durations = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    trate = 0.2              # 0.05 -> 5% training data 
    vrate = 1.0 - (0.1 + trate) # 0.2 -> 20% test data
    inp, train_inps, valid_inps, test_inps = stratified_split_data(years, 
                                                                    durations, 
                                                                    trate=trate, 
                                                                    vrate=vrate, 
                                                                    stratify=True)
    print(f'trains: {len(train_inps)}, tests: {len(test_inps)}')  
    # weights: w1 -> level(volume), w2 -> on/off, w3 -> energy, w4 -> dry run
    # ga_rate:  if (epsilon > ga_rate * initial_epsilon) in dqn_from_demon_v1.py
    # params = {'minutes':2, 'epochs':1, 'weights': [1.0, 1.0, 0.0, 0.0], 'ga_rate':0.1}
    params_reg = {'epochs': 1, 'minutes': 2, 'weights': [0.25, 0.4, 0.25, 0.1], 
              'ga_start': 0.6, 'ga_end': 0.0,   # epsilon schedule for random exploration when GA is not used
              'eps_start': 0.3, 'eps_end': 0.05, 
              'gamma': 0.1, 'batch_size': 20, 'learning_rate': 1e-3, 'sync_freq': 200, # based on update count
              'act_type':0} # act_type => 0: limited actions, 1: full actions
    
    # model_path = f"/trained_models/dqn_gru_regular_w{params_reg['weights'][0]}_w{params_reg['weights'][1]}_w{params_reg['weights'][2]}_w{params_reg['weights'][3]}_min2_tr{trate}_acttype{params_reg['act_type']}.pkl"
    # model_path = model_path.replace('.', '_', model_path.count('.') - 1)
    # model_path = ".." + model_path
    # basic_losses, basic_rewards, basic_test_rewards, basic_elapsed \
    #     = evaluate_dpn_gru(train_inps, test_inps, 
    #                       years, durations, 
    #                       train=True,
    #                       model_path=model_path,
    #                       **params_reg)
    
    # params_gen = {'epochs': 1, 'minutes': 2, 'weights': [0.25, 0.4, 0.25, 0.1], 
    #       'ga_start': 0.6, 'ga_end': 0.0,   # epsilon schedule for random exploration when GA is not used
    #       'eps_start': 0.3, 'eps_end': 0.05, 
    #       'gamma': 0.1, 'batch_size': 20, 'learning_rate': 1e-3, 'sync_freq': 200, # based on update count
    #       'act_type':0} # act_type 1: using full actions
    
    # model_path = f"/trained_models/dqn_guided_GA_w{params_gen['weights'][0]}_w{params_gen['weights'][1]}_w{params_gen['weights'][2]}_w{params_gen['weights'][3]}_ga{params_gen['ga_start']}_min2_tr{trate}_acttype{params_gen['act_type']}.pkl"
    # model_path = model_path.replace('.', '_', model_path.count('.') - 1)
    # model_path = ".." + model_path
    # ga_losses, ga_rewards, ga_test_rewards, ga_elapsed \
    #     = evaluate_dpn_guided_GA(train_inps, test_inps, 
    #                       years, durations,
    #                       train=True,
    #                       model_path=model_path,
    #                       **params_gen)

    params = {'minutes':2, 'epochs':1, 'weights': [0.25, 0.4, 0.25, 0.1], 'act_type':0}
    ga = GeneticAlgorithm(**params)
    action_list, state_list, reward_list, info_list = ga.run_genetic_algo(test_list[0])
    
    # #plot loss and reward of training data set
    # if basic_losses is not None and ga_losses is not None:
    #     dqn_from_demon_v1.plot_losses(basic_losses, ga_losses)
    #     dqn_from_demon_v1.plot_rewards(basic_rewards, ga_rewards)
    
    # #Plot reward of test data set
    # if basic_test_rewards is not None and ga_test_rewards is not None:
    #     dqn_from_demon_v1.plot_rewards(basic_test_rewards, ga_test_rewards)
    # print(f"Elapsed Time -> DDQN: {basic_elapsed} seconds, GA_DDQN: {ga_elapsed} seconds")
    # print(f"Test Rewards -> DDQN: {np.array(basic_test_rewards).sum()}, GA_DDQN: {np.array(ga_test_rewards).sum()}")

    



#     start_time = time.perf_counter()
#     evaluate_genetic_algo(test_inps, years, durations, **params)
#     end_time = time.perf_counter()
#     print(f"elapsed time for genetic algorithm: {end_time-start_time} seconds")
#     evaluate_man_policy(test_inps, years, durations, **params)
    
    
    
#     model, losses = dqn_gru_torch.train(train_inps, valid_inps, epochs=epochs, minutes=minutes)
    
#     # for seek the average highest level
#     highest_levels = np.zeros(shape=(len(years), len(durations)))
#     counts = np.full((len(years), len(durations)), 1e-10)
    
#     # count flooding cases
#     counts_flood = np.zeros(shape=(len(years), len(durations)))
    
#     # count the number of pump changes
#     pump_changes = np.zeros(shape=(len(years), len(durations)))
    
#     for test_inp in test_inps[:50]:
#         _, actions, states, rewards, infos \
#             = dqn_gru_torch.test_model(None, test_inp, minutes=minutes)
        
#         # find ids and increase the sample count by one
#         id_year = years.index(re.findall(r'\d+', test_inp.split('/')[-1])[0])
#         id_du = durations.index(re.findall(r'\d+', test_inp.split('/')[-1])[1])
#         counts[id_year][id_du] += 1
        
#         # compute the number of pump changes
#         pump_changes[id_year][id_du] += count_changes(actions)
        
#         # find the hightest level for a scenario
#         highest = -1
#         for state in states:
#             if highest < state[-1]:
#                 highest = state[-1]
#         highest_levels[id_year][id_du] += highest
        
#         # if the highest level is greater than 10M then flooing is occurred
#         if highest > water_gym.Level[-1]:
#             counts_flood [id_year][id_du] += 1
        
#     avg_highest_levels = highest_levels / counts
#     avg_pump_changes = pump_changes / counts
    
#     print(avg_highest_levels)
#     print(avg_pump_changes)
#     print(counts_flood)
    
# %%
# 개별 그림        
# plt.figure()
# fig_title =  test_inp.split('/')[-1]
# perform_evaluate.plot_performance(fig_title, states, actions, ['rain', 'inflow', 'outflow', 'vol', 'level'])
