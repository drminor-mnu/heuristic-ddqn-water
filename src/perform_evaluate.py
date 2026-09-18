#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:06:13 2024



@author: drminor
"""

import re 
import csv
from itertools import zip_longest
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import sys

from water_gym import Actions0, Actions1, pumpq
import water_gym
import dqn_from_demon_v1
from genetic_algo import GeneticAlgorithm
from pso import ParticleSwarmOptimization
from local_search import LocalSearch
import man_policy

from plot_style import set_plot_style
import data_paths
from atomic_io import atomic_write

set_plot_style()


#random.seed(time.time())

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
                     seed=None,
                     model_label=None,
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
    scenario_rows = []
    for i, test_inp in enumerate(test_inps):
        print(f'\rGRU Test #{i}', end='', flush=True)
        scenario_start = time.perf_counter()
        _, actions, states, rewards, infos \
            = dqn_from_demon_v1.test_model(model, test_inp, minutes=minutes,
                                           model_path=model_path,
                                           weights=weights,
                                           act_type=act_type)
        scenario_total_time_s = time.perf_counter() - scenario_start

        # Reward calculation of the i-th test data
        reward = np.array(rewards).sum() # total reward of a test scenario
        dqn_test_rewards.append(reward)

        # find ids and increase the sample count by one
        file_year = re.findall(r'\d+', test_inp.split('/')[-1])[0]
        file_du = re.findall(r'\d+', test_inp.split('/')[-1])[1]
        id_year = years.index(file_year)
        id_du = durations.index(file_du)
        counts[id_year][id_du] += 1

        # compute the number of pump changes
        n_switches = count_changes(actions, act_type)
        pump_changes[id_year][id_du] += n_switches

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
        overflow_flag = highest >= water_gym.Level[-1]
        if overflow_flag:
            counts_flood[id_year][id_du] += 1

        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        n_dryrun_proxy = np.sum(ainfos[:, -1])
        counts_overpump[id_year][id_du] += n_dryrun_proxy

        scenario_rows.append({
            'scenario_id': Path(test_inp).stem,
            'return_period': file_year,
            'duration_min': file_du,
            'quartile': '',
            'seed': seed if seed is not None else '',
            'model': model_label if model_label is not None else '',
            'max_level_m': highest,
            'n_switches': n_switches,
            'n_intervals_100': pump100,
            'n_intervals_170': pump170,
            'n_dryrun_proxy': int(n_dryrun_proxy),
            'cum_reward': reward,
            'overflow_flag': int(overflow_flag),
            'inference_time_s': '',
            'swmm_time_s': '',
            'total_time_s': scenario_total_time_s,
        })

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
    save_scenario_metrics(scenario_rows, file=str_w)
    # print(avg_highest_levels)
    # print(avg_pump_changes)
    # print(avg_pump100)
    # print(avg_pump170)
    # print(avg_cnt_pumps)
    # print(avg_overpump)
    # print(counts_flood)

    return basic_losses, basic_rewards, dqn_test_rewards, elapsed


def _evaluate_dpn_guided(train_inps, test_inps,
                         years, durations,
                         trainer,
                         guide_label,
                         train: bool = True,
                         model_path: str = None,
                         seed=None,
                         model_label=None,
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
        model, guided_losses, guided_rewards, \
            = trainer(train_inps, model_path, **params)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Elapsed time for ddqn guided by {guide_label}: {elapsed} seconds')
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
    
    guided_test_rewards = []
    scenario_rows = []
    for i, test_inp in enumerate(test_inps):
        print(f'\rDQN guided by {guide_label} Test #{i}', end='', flush=True)
        scenario_start = time.perf_counter()
        _, actions, states, rewards, infos \
            = dqn_from_demon_v1.test_model(model, test_inp, minutes=minutes,
                                           model_path=model_path, weights=weights,
                                           act_type=act_type)
        scenario_total_time_s = time.perf_counter() - scenario_start

        # Reward calculation of the i-th test data
        reward = np.array(rewards).sum() # total reward of a test scenario
        guided_test_rewards.append(reward)

        # find ids and increase the sample count by one
        file_year = re.findall(r'\d+', test_inp.split('/')[-1])[0]
        file_du = re.findall(r'\d+', test_inp.split('/')[-1])[1]
        id_year = years.index(file_year)
        id_du = durations.index(file_du)
        counts[id_year][id_du] += 1

        # compute the number of pump changes
        n_switches = count_changes(actions, act_type)
        pump_changes[id_year][id_du] += n_switches

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
        overflow_flag = highest >= water_gym.Level[-1]
        if overflow_flag:
            counts_flood[id_year][id_du] += 1

        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        n_dryrun_proxy = np.sum(ainfos[:, -1])
        counts_overpump[id_year][id_du] += n_dryrun_proxy

        scenario_rows.append({
            'scenario_id': Path(test_inp).stem,
            'return_period': file_year,
            'duration_min': file_du,
            'quartile': '',
            'seed': seed if seed is not None else '',
            'model': model_label if model_label is not None else '',
            'max_level_m': highest,
            'n_switches': n_switches,
            'n_intervals_100': pump100,
            'n_intervals_170': pump170,
            'n_dryrun_proxy': int(n_dryrun_proxy),
            'cum_reward': reward,
            'overflow_flag': int(overflow_flag),
            'inference_time_s': '',
            'swmm_time_s': '',
            'total_time_s': scenario_total_time_s,
        })

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
    save_scenario_metrics(scenario_rows, file=str_w)
    # print(avg_highest_levels)
    # print(avg_pump_changes)
    # print(avg_pump100)
    # print(avg_pump170)
    # print(avg_cnt_pumps)
    # print(avg_overpump)
    # print(counts_flood)

    return guided_losses, guided_rewards, guided_test_rewards, elapsed


def evaluate_dpn_guided_GA(train_inps, test_inps, years, durations,
                           train: bool = True, model_path: str = None,
                           **params):
    """Evaluate DDQN-GRU trained with GA-guided replay transitions."""
    return _evaluate_dpn_guided(
        train_inps, test_inps, years, durations,
        trainer=dqn_from_demon_v1.train_guided_byGA,
        guide_label='GA', train=train, model_path=model_path, **params,
    )


def evaluate_dpn_guided_PSO(train_inps, test_inps, years, durations,
                            train: bool = True, model_path: str = None,
                            **params):
    """Evaluate DDQN-GRU trained with PSO-guided replay transitions."""
    return _evaluate_dpn_guided(
        train_inps, test_inps, years, durations,
        trainer=dqn_from_demon_v1.train_guided_byPSO,
        guide_label='PSO', train=train, model_path=model_path, **params,
    )


def evaluate_dpn_guided_ENUM(train_inps, test_inps, years, durations,
                             train: bool = True, model_path: str = None,
                             **params):
    """Evaluate DDQN-GRU trained with ENUM-guided replay transitions (E4-a).

    Same evaluation path as the GA/PSO variants; only the guide optimizer
    differs (exact 1-step argmax over the 6 feasible actions).
    """
    return _evaluate_dpn_guided(
        train_inps, test_inps, years, durations,
        trainer=dqn_from_demon_v1.train_guided_byENUM,
        guide_label='ENUM', train=train, model_path=model_path, **params,
    )


def evaluate_genetic_algo(test_inps,
                     years, durations,
                     seed=None,
                     model_label=None,
                     result_file=None,
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
    
    scenario_rows = []
    for i, test_inp in enumerate(test_inps):
        print(f'\rGenetic Test #{i}', end='', flush=True)
        GA = GeneticAlgorithm(**params)
        scenario_start = time.perf_counter()
        actions, states, rewards, infos = GA.run_genetic_algo(test_inp)
        scenario_total_time_s = time.perf_counter() - scenario_start

        # find ids and increase the sample count by one
        file_year = re.findall(r'\d+', test_inp.split('/')[-1])[0]
        file_du = re.findall(r'\d+', test_inp.split('/')[-1])[1]
        id_year = years.index(file_year)
        id_du = durations.index(file_du)
        counts[id_year][id_du] += 1

        # compute the number of pump changes
        n_switches = count_changes(actions, act_type)
        pump_changes[id_year][id_du] += n_switches

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
        overflow_flag = highest >= water_gym.Level[-1]
        if overflow_flag:
            counts_flood[id_year][id_du] += 1

        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        n_dryrun_proxy = np.sum(ainfos[:, -1])
        counts_overpump[id_year][id_du] += n_dryrun_proxy

        scenario_rows.append({
            'scenario_id': Path(test_inp).stem,
            'return_period': file_year,
            'duration_min': file_du,
            'quartile': '',
            'seed': seed if seed is not None else '',
            'model': model_label if model_label is not None else '',
            'max_level_m': highest,
            'n_switches': n_switches,
            'n_intervals_100': pump100,
            'n_intervals_170': pump170,
            'n_dryrun_proxy': int(n_dryrun_proxy),
            'cum_reward': np.array(rewards).sum(),
            'overflow_flag': int(overflow_flag),
            'inference_time_s': '',
            'swmm_time_s': '',
            'total_time_s': scenario_total_time_s,
        })

    print()
    avg_highest_levels = highest_levels / counts
    avg_pump_changes = pump_changes / counts
    avg_pump100 = counts_pump100 / counts
    avg_pump170 = counts_pump170 / counts
    avg_overpump = counts_overpump / counts
    avg_cnt_pumps = cnt_pumps/counts[..., np.newaxis]

    str_w = '_'.join(['w'+str(w) for w in weights])
    output_name = result_file or f'genetic_{str_w}'
    save_results(avg_highest_levels,
                 avg_pump_changes, avg_pump100, avg_pump170,
                 avg_cnt_pumps, avg_overpump, counts_flood, counts, file=output_name)
    save_scenario_metrics(scenario_rows, file=output_name)
    # print(avg_highest_levels)
    # print(avg_pump_changes)
    # print(avg_pump100)
    # print(avg_pump170)
    # print(avg_cnt_pumps)
    # print(avg_overpump)
    # print(counts_flood)


def _evaluate_heuristic(test_inps, years, durations, optimizer_class,
                        optimize_label, run_method, result_prefix,
                        result_file=None, seed=None, model_label=None,
                        **params):
    """Evaluate a heuristic controller with the same metrics as the GA."""
    weights = params.get("weights", [1, 1, 1, 0])
    act_type = params.get("act_type", 0)
    actions_space = Actions0 if act_type == 0 else Actions1

    print(f'tests: {len(test_inps)}')

    shape = (len(years), len(durations))
    highest_levels = np.zeros(shape=shape)
    counts = np.full(shape, 1e-10)
    counts_flood = np.zeros(shape=shape)
    pump_changes = np.zeros(shape=shape)
    counts_pump100 = np.zeros(shape=shape)
    counts_pump170 = np.zeros(shape=shape)
    cnt_pumps = np.zeros(shape=shape + (len(actions_space[0]),))
    counts_overpump = np.zeros(shape=shape)

    scenario_rows = []
    for i, test_inp in enumerate(test_inps):
        print(f'\r{optimize_label} Test #{i}', end='', flush=True)
        optimizer = optimizer_class(**params)
        scenario_start = time.perf_counter()
        actions, states, rewards, infos = getattr(optimizer, run_method)(test_inp)
        scenario_total_time_s = time.perf_counter() - scenario_start

        file_numbers = re.findall(r'\d+', test_inp.split('/')[-1])
        id_year = years.index(file_numbers[0])
        id_du = durations.index(file_numbers[1])
        counts[id_year][id_du] += 1

        n_switches = count_changes(actions, act_type)
        pump_changes[id_year][id_du] += n_switches
        pump100, pump170, npump = count_pumps(actions, act_type)
        counts_pump100[id_year][id_du] += pump100
        counts_pump170[id_year][id_du] += pump170
        cnt_pumps[id_year][id_du] += npump

        highest = max(state[-1] for state in states)
        highest_levels[id_year][id_du] += highest
        overflow_flag = highest >= water_gym.Level[-1]
        if overflow_flag:
            counts_flood[id_year][id_du] += 1

        ainfos = np.asarray(infos, dtype=np.float32)
        n_dryrun_proxy = np.sum(ainfos[:, -1])
        counts_overpump[id_year][id_du] += n_dryrun_proxy

        scenario_rows.append({
            'scenario_id': Path(test_inp).stem,
            'return_period': file_numbers[0],
            'duration_min': file_numbers[1],
            'quartile': '',
            'seed': seed if seed is not None else '',
            'model': model_label if model_label is not None else '',
            'max_level_m': highest,
            'n_switches': n_switches,
            'n_intervals_100': pump100,
            'n_intervals_170': pump170,
            'n_dryrun_proxy': int(n_dryrun_proxy),
            'cum_reward': np.array(rewards).sum(),
            'overflow_flag': int(overflow_flag),
            'inference_time_s': '',
            'swmm_time_s': '',
            'total_time_s': scenario_total_time_s,
        })

    print()
    avg_highest_levels = highest_levels / counts
    avg_pump_changes = pump_changes / counts
    avg_pump100 = counts_pump100 / counts
    avg_pump170 = counts_pump170 / counts
    avg_overpump = counts_overpump / counts
    avg_cnt_pumps = cnt_pumps / counts[..., np.newaxis]

    str_w = '_'.join(['w' + str(weight) for weight in weights])
    output_name = result_file or f'{result_prefix}_{str_w}'
    save_results(avg_highest_levels,
                 avg_pump_changes, avg_pump100, avg_pump170,
                 avg_cnt_pumps, avg_overpump, counts_flood, counts,
                 file=output_name)
    save_scenario_metrics(scenario_rows, file=output_name)


def evaluate_pso(test_inps, years, durations, **params):
    """Evaluate PSO with the same inputs and outputs as evaluate_genetic_algo."""
    return _evaluate_heuristic(
        test_inps, years, durations,
        optimizer_class=ParticleSwarmOptimization,
        optimize_label='PSO',
        run_method='run_pso',
        result_prefix='pso',
        **params,
    )


def evaluate_local_search(test_inps, years, durations, **params):
    """Evaluate local search with the same inputs and outputs as the GA."""
    return _evaluate_heuristic(
        test_inps, years, durations,
        optimizer_class=LocalSearch,
        optimize_label='Local Search',
        run_method='run_local_search',
        result_prefix='local_search',
        **params,
    )
    

def evaluate_man_policy(test_inps,
                     years, durations,
                     seed=None,
                     model_label=None,
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
    
    scenario_rows = []
    for i, test_inp in enumerate(test_inps):
        print(f'\rMan Test #{i}', end='', flush=True)
        scenario_start = time.perf_counter()
        actions, states, rewards, infos = man_policy.operation(test_inp, minutes)
        scenario_total_time_s = time.perf_counter() - scenario_start

        # find ids and increase the sample count by one
        file_year = re.findall(r'\d+', test_inp.split('/')[-1])[0]
        file_du = re.findall(r'\d+', test_inp.split('/')[-1])[1]
        id_year = years.index(file_year)
        id_du = durations.index(file_du)
        counts[id_year][id_du] += 1

        # compute the number of pump changes
        n_switches = count_changes(actions, act_type)
        pump_changes[id_year][id_du] += n_switches

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
        overflow_flag = highest >= water_gym.Level[-1]
        if overflow_flag:
            counts_flood[id_year][id_du] += 1

        # count the overpumping
        ainfos = np.array(infos, dtype=np.float32)
        n_dryrun_proxy = np.sum(ainfos[:, -1])
        counts_overpump[id_year][id_du] += n_dryrun_proxy

        scenario_rows.append({
            'scenario_id': Path(test_inp).stem,
            'return_period': file_year,
            'duration_min': file_du,
            'quartile': '',
            'seed': seed if seed is not None else '',
            'model': model_label if model_label is not None else '',
            'max_level_m': highest,
            'n_switches': n_switches,
            'n_intervals_100': pump100,
            'n_intervals_170': pump170,
            'n_dryrun_proxy': int(n_dryrun_proxy),
            'cum_reward': np.array(rewards).sum(),
            'overflow_flag': int(overflow_flag),
            'inference_time_s': '',
            'swmm_time_s': '',
            'total_time_s': scenario_total_time_s,
        })

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
    save_scenario_metrics(scenario_rows, file=f'man_policy_{str_w}')
    # print(avg_highest_levels)
    # print(avg_pump_changes)
    # print(avg_pump100)
    # print(avg_pump170)
    # print(avg_cnt_pumps)
    # print(avg_overpump)
    # print(counts_flood)


def save_results(level, change, pump100, pump170, npumps,
                 overpump, flood, count, file:str='results'):
    """
    전체 테스트 데이터((year, duration) 별)를 대상으로 최고수위평균, 펌프변경수평균, 펌프100사용수평규,
    펌프170사용수평균, 과다펌핑평균, 홍수발생수, 시나리오수 반환

    `atomic_write()`(temp 파일 작성 → fsync → os.replace → 상위 디렉터리 fsync)로 쓰므로
    E3 장시간 실행 중 중단돼도 최종 경로에 불완전한 파일이 남지 않는다.

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

    def _write(fd):
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
        # the number of each pumps in pumps (maybe 6가지 actions 의미)
        vec_npumps = vec_fun(npumps)
        for y in vec_npumps:
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

    atomic_write(fname, _write)


def save_scenario_metrics(rows, file: str = 'scenario_metrics'):
    """
    시나리오 단위 raw 결과 저장 (REVISION_EXPERIMENT_PLAN.md 2.2절 test_metrics.csv 규격).

    save_results()는 (year x duration) 집계만 남기므로 R4-10이 요구한 표준편차·
    신뢰구간이나 E3의 시나리오 단위 paired 검정을 낼 수 없다. 이 함수는 그 대신
    시나리오 1건당 1행을 raw로 저장한다. save_results()를 대체하지 않고 별도로
    호출한다.

    컬럼: scenario_id, return_period, duration_min, quartile, seed, model,
    max_level_m, n_switches, n_intervals_100, n_intervals_170, n_dryrun_proxy,
    cum_reward, overflow_flag, inference_time_s, swmm_time_s, total_time_s

    - quartile: 강우 시나리오 파일명의 분위(Huff quartile) 인덱스. 확정 불가
      (docs/CODEBASE_MAP.md "확인 불가 항목 1" 참조) — 항상 빈 값.
    - seed, model: 호출자가 넘기는 값. 기본값(None)이면 빈 값으로 기록되어
      기존 호출부를 깨지 않는다.
    - n_dryrun_proxy: 시나리오 내 과다펌핑(dry-running proxy) 발생 스텝 수.
      water_gym.py의 step()이 매 스텝 산출하는 이진 플래그(over_pump, 0 또는 1)의
      합이며, 비율이나 연속값이 아니다 (docs/CODEBASE_MAP.md 확인 완료 항목 참조).
    - total_time_s는 SWMM 시뮬레이션(WaterGym.reset)을 포함한다.
      inference_time_s와 swmm_time_s로의 분해는 E10에서 수행한다.
      이번 단계에서는 total_time_s만 채우고 나머지 둘은 빈 값으로 둔다.
    - `atomic_write()`로 쓰므로 E3 장시간 실행 중 중단돼도 최종 경로에
      불완전한 파일이 남지 않는다.

    Parameters
    ----------
    rows : list of dict
        각 dict는 위 15개 컬럼 키를 가져야 한다.
    file : str, optional
        출력 파일명(확장자 제외). '../results/{file}_scenario_metrics.csv'로 저장된다.

    Returns
    -------
    str
        저장된 파일 경로.
    """
    fieldnames = [
        'scenario_id', 'return_period', 'duration_min', 'quartile',
        'seed', 'model', 'max_level_m', 'n_switches',
        'n_intervals_100', 'n_intervals_170', 'n_dryrun_proxy',
        'cum_reward', 'overflow_flag',
        'inference_time_s', 'swmm_time_s', 'total_time_s',
    ]
    fname = f'../results/{file}_scenario_metrics.csv'

    def _write(fd):
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return atomic_write(fname, _write, newline='')


def save_reward_data(basic_rewards, ga_rewards,
                     basic_test_rewards, ga_test_rewards,
                     filename='reward_comparison.csv'):
    """Save four reward sequences as columns in a single CSV file."""
    result_dir = Path(__file__).resolve().parent.parent / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / filename

    def values(data):
        if data is None:
            return []
        return np.asarray(data, dtype=float).reshape(-1).tolist()

    columns = [
        values(basic_rewards),
        values(ga_rewards),
        values(basic_test_rewards),
        values(ga_test_rewards),
    ]

    with output_path.open('w', newline='', encoding='utf-8') as fd:
        writer = csv.writer(fd)
        writer.writerow([
            'regular_train_reward',
            'ga_guided_train_reward',
            'regular_test_reward',
            'ga_guided_test_reward',
        ])
        writer.writerows(zip_longest(*columns, fillvalue=''))

    return output_path
        

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
    trate = 0.6              # 0.2 -> 20% training data 
    vrate = 1.0 - (0.4 + trate) # 0.1 -> 10% test data
    train_inps, valid_inps, test_inps = data_paths.load_fixed_split()
    meta = data_paths.get_split_meta()
    print(f"split={meta['split_file']} seed={meta['split_seed']}")
    print(f'trains: {len(train_inps)}, valids: {len(valid_inps)}, tests: {len(test_inps)}')
    # wights: w1 -> level(volume), w2 -> on/off, w3 -> energy, w4 -> dry run
    # ga_rate:  if (epsilon > ga_rate * initial_epsilon) in dqn_from_demon_v1.py
    # params = {'minutes':2, 'epochs':1, 'weights': [1.0, 1.0, 0.0, 0.0], 'ga_rate':0.1}
    params_reg = {'epochs': 1, 'minutes': 2, 'weights': [1.0, 1.0, 0.0, 0.0], 
              'ga_start': 0.6, 'ga_end': 0.0,   # epsilon schedule for random exploration when GA is not used
              'eps_start': 0.3, 'eps_end': 0.05, 
              'gamma': 0.1, 'batch_size': 20, 'learning_rate': 1e-3, 'sync_freq': 200, # based on update count
              'act_type':0} # act_type => 0: limited actions, 1: full actions
    
    model_path = f"/trained_models/dqn_gru_regular_w{params_reg['weights'][0]}_w{params_reg['weights'][1]}_w{params_reg['weights'][2]}_w{params_reg['weights'][3]}_min2_tr{trate}_acttype{params_reg['act_type']}.pkl"
    model_path = model_path.replace('.', '_', model_path.count('.') - 1)
    model_path = ".." + model_path
    basic_losses, basic_rewards, basic_test_rewards, basic_elapsed \
        = evaluate_dpn_gru(train_inps, test_inps, 
                          years, durations, 
                          train=True,
                          model_path=model_path,
                          **params_reg)
    
    params_gen = {'epochs': 1, 'minutes': 2, 'weights': [0.25, 0.4, 0.25, 0.1], 
          'ga_start': 0.6, 'ga_end': 0.0,   # epsilon schedule for random exploration when GA is not used
          'eps_start': 0.3, 'eps_end': 0.05, 
          'gamma': 0.1, 'batch_size': 20, 'learning_rate': 1e-3, 'sync_freq': 200, # based on update count
          'act_type':0} # act_type 1: using full actions
    
    model_path = f"/trained_models/dqn_guided_GA_w{params_gen['weights'][0]}_w{params_gen['weights'][1]}_w{params_gen['weights'][2]}_w{params_gen['weights'][3]}_ga{params_gen['ga_start']}_min2_tr{trate}_acttype{params_gen['act_type']}.pkl"
    model_path = model_path.replace('.', '_', model_path.count('.') - 1)
    model_path = ".." + model_path
    ga_losses, ga_rewards, ga_test_rewards, ga_elapsed \
        = evaluate_dpn_guided_GA(train_inps, test_inps, 
                          years, durations,
                          train=True,
                          model_path=model_path,
                          **params_gen)

    # reward_csv = save_reward_data(
    #     basic_rewards,
    #     ga_rewards,
    #     basic_test_rewards,
    #     ga_test_rewards,
    #     filename='reward_4_percent.csv'
    # )
    # print(f"Reward data saved to: {reward_csv}")

    #plot loss and reward of training data set
    if basic_losses is not None and ga_losses is not None:
        dqn_from_demon_v1.plot_losses(basic_losses, ga_losses)
        dqn_from_demon_v1.plot_rewards(basic_rewards, ga_rewards)
    
    #Plot reward of test data set
    if basic_test_rewards is not None and ga_test_rewards is not None:
        dqn_from_demon_v1.plot_rewards(basic_test_rewards, ga_test_rewards)
    print(f"Elapsed Time -> DDQN: {basic_elapsed} seconds, GA_DDQN: {ga_elapsed} seconds")
    print(f"Test Rewards -> DDQN: {np.array(basic_test_rewards).sum()}, GA_DDQN: {np.array(ga_test_rewards).sum()}")

    
    # PARAMS = {
    # 'epochs': 1,
    # 'minutes': 2,
    # 'weights': [0.25, 0.4, 0.25, 0.1],
    # 'ga_start': 0.6,
    # 'ga_end': 0.0,
    # 'eps_start': 0.3,
    # 'eps_end': 0.05,
    # 'gamma': 0.1,
    # 'batch_size': 20,
    # 'learning_rate': 0.001,
    # 'sync_freq': 200,
    # 'act_type': 0,
    # }

    # EXPERIMENT_NAME = 'dqn_guided_PSO_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0'
    # root = Path(__file__).resolve().parent.parent
    # model_path = root / 'trained_models' / f'{EXPERIMENT_NAME}.pkl'
    # result_path = root / 'results' / f'{EXPERIMENT_NAME}.csv'

    # pso_losses, pso_rewards, pso_test_rewards, pso_elapsed = evaluate_dpn_guided_PSO(
    #     train_inps, test_inps, years, durations,
    #     train=True, model_path=str(model_path), **PARAMS,
    # )
    # if not result_path.is_file() or result_path.stat().st_size == 0:
    #     raise RuntimeError(f'Expected result was not created: {result_path}')
    # print(f'Result saved to: {result_path}')
    # print(f"Elapsed Time -> PSO_DDQN: {pso_elapsed} seconds")
    # print(f"Test Rewards -> PSO_DDQN: {np.array(pso_test_rewards).sum()}")

    
    """
    Evaluate the performance of the genetic algorithm, PSO, and local search on the test inputs.
    The parameters for the evaluation can be adjusted in the params dictionary.
    """
    # params = {'minutes':2, 'epochs':1, 'weights': [0.25, 0.4, 0.25, 0.1], 'act_type':0}
    # start_time = time.perf_counter()
    # evaluate_genetic_algo(test_inps, years, durations, **params)
    # end_time = time.perf_counter()
    # print(f"elapsed time for genetic algorithm: {end_time-start_time} seconds")

    # start_time = time.perf_counter()
    # evaluate_pso(test_inps, years, durations, **params)
    # end_time = time.perf_counter()
    # print(f"elapsed time for PSO: {end_time-start_time} seconds")

    # start_time = time.perf_counter()
    # evaluate_local_search(test_inps, years, durations, **params)
    # end_time = time.perf_counter()
    # print(f"elapsed time for local search: {end_time-start_time} seconds")
    

    """
    Evaluate the performance of the manual policy on the test inputs.
    The parameters for the evaluation can be adjusted in the params dictionary."""
    # start_time = time.perf_counter()
    # evaluate_man_policy(test_inps, years, durations, **params)
    # end_time = time.perf_counter()
    # print(f"elapsed time for manual policy: {end_time-start_time} seconds")
    
    
    
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
