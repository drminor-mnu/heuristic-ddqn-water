#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:06:13 2024

@author: drminor
"""

import matplotlib.pyplot as plt
import random
import numpy as np

import dqn_model
import dqn_gru_torch
import man_policy
import perform_evaluate
import cost_model_v2
import graph_draw
from water_gym import WaterGym, Actions
import utils

random.seed(24)

"""
state_, reward, done, info , <= 
        [self.rains[self.clock], 
        cur_inflow, 
        cur_outflow, 
        cur_vol, 
        cur_level],\
        cur_reward, done, info
        
info <= (vol_reward, act_reward, energy_reward, excess_pump * excess_pump_penalty)
"""

#years = ['10', '20', '30', '50', '80', '100'],
#minutes = ['0010', '0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']

model_path = '../trained_models/dqn_gru_pumpmodel_w1_w1_w1_min2.pkl'
print(model_path)
# %%
# Individual case graph for energy consumption and cost

years = ['50']
durations = ['0120']

# Parameters for experiments
params = {"minutes":2, "init_cycles":15000}

# data acquisition
inp, train_inps, valid_inps, test_inps = perform_evaluate.stratified_split_data(years, durations, trate=0.7, vrate=0.1, stratify=True)
print(f'trains: {len(train_inps)}, tests: {len(test_inps)}')
choice = random.randint(0, len(test_inps))
test_inp = test_inps[choice]
test_inp = 'data/gasan/50year/50yr_0120m_h290.inp'

# Test for dqn_gru
model, dactions, dstates, drewards, dinfos = dqn_gru_torch.test_model(None, test_inp, minutes=params["minutes"], model_path=model_path)
fig_title =  test_inp.split('/')[-1]
dqn_pumps = cost_model_v2.execute_scenario(dactions, dinfos, minutes=params["minutes"], init_cycles=params["init_cycles"])
dqn_results = cost_model_v2.analysis_result(dqn_pumps)

graph_draw.plot_fig_pumps(dqn_pumps, dqn_results, "energy_consume_cost_day_1_dqn.png")
# graph_draw.plot_individual_case(states, actions)
# graph_draw.plot_performance(fig_title, states, actions, ['rain', 'inflow', 'outflow', 'vol', 'level'])

# Test for man policy
mactions, mstates, mrewards, minfos = man_policy.operation(test_inp, minutes=params["minutes"])
fig_title =  test_inp.split('/')[-1]
man_pumps = cost_model_v2.execute_scenario(mactions, minfos, minutes=params["minutes"], init_cycles=params["init_cycles"])
man_results = cost_model_v2.analysis_result(man_pumps)

graph_draw.plot_fig_pumps(man_pumps, man_results, "energy_consume_cost_day_1_man.png")
# # graph_draw.plot_individual_case(states, actions)
# graph_draw.plot_performance(fig_title, states, actions, ['rain', 'inflow', 'outflow', 'vol', 'level'])

dlevel = []
for s in dstates:
    dlevel.append(s[-1])

mlevel = []
for s in mstates:
    mlevel.append(s[-1])
graph_draw.plot_individual_level(dlevel, mlevel, "level_dqn_man.png")


# %%
# Scenario Experiments (using 36 inp files)

params = {"minutes":2, "init_cycles":0}
inps = utils.inp_select_for_scenarios()
random.shuffle(inps)

# Test for DQN
dqn_actions = []
dqn_infos = []
lengths = [] # 시나리오에 속하는 각 inp 파일의 actions의 길이 -> inp(or day) 별로 상황 변화 확인을 위해
for inp in inps:
    model, actions, states, rewards, infos = dqn_gru_torch.test_model(None, inp, minutes=params["minutes"], model_path=model_path)
    lengths.append(len(actions))
    dqn_actions.extend(actions)
    dqn_infos.extend(infos)
lengths = list(np.array(lengths).cumsum())
dqn_dstruct, dqn_pumps, dqn_results = cost_model_v2.execute_scenario_fordays(dqn_actions, 
                      dqn_infos, lengths,  
                      params["minutes"], params["init_cycles"])


# Test for MAN
man_actions = []
man_infos = []
lengths = [] # 시나리오에 속하는 각 inp 파일의 actions의 길이 -> inp(or day) 별 상황 변화 확인을 위해
for inp in inps:
    actions, states, rewards, infos = man_policy.operation(inp, minutes=params["minutes"])
    lengths.append(len(actions))
    man_actions.extend(actions)
    man_infos.extend(infos)
lengths = list(np.array(lengths).cumsum())
man_dstruct, man_pumps, man_results = cost_model_v2.execute_scenario_fordays(man_actions, 
                      man_infos, lengths,  
                      params["minutes"], params["init_cycles"])

graph_draw.plot_fig_221(dqn_dstruct, "energy_consume_cost_day_36_dqn.png")
graph_draw.plot_fig_221(man_dstruct, "energy_consume_cost_day_36_man.png")
graph_draw.plot_fig_222(dqn_dstruct, "onoff_compose_day_36_down_0_dqn.png")
graph_draw.plot_fig_222(man_dstruct, "onoff_compose_day_36_down_0_man.png")
graph_draw.plot_fig_223(dqn_dstruct, man_dstruct, "maintenance_day_36_down_0_dqn_man.png")
graph_draw.plot_fig_224(dqn_dstruct, man_dstruct, "total_day_36_down_0_dqn_man.png")
# # %%

