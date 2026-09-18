#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:07:37 2024

@author: drminor
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import csv

import dqn_gru_torch
import man_policy
import dqn_from_demon_v1
import perform_evaluate
import cost_model_v2

from plot_style import set_plot_style

set_plot_style()


#x_ticks=['10', '60', '120', '180', '240', '360', '540', '720', '1080', '1440']

def sci_formatter(x, pos):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coef = x / 10**exp
    # 5 x 10^3 형태 (mathtext)
    return r"${:.0f}\times10^{{{}}}$".format(coef, exp)

def plot_individual_case(state_list, action_list):
    """
    state_list 
    [self.rains[self.clock], 
    cur_inflow, 
    cur_outflow, 
    cur_vol, 
    cur_level]

    Parameters
    ----------
    state_list : TYPE
        DESCRIPTION.
    action_list : TYPE
        DESCRIPTION.
    ['rain', 'inflow', 'outflow', 'vol', 'level'] : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
plot_individual_case(state_list, action_list)
    """
    rain = []
    inflow = []
    outflow = []
    volume = []
    level = []
    action = np.array(action_list) + 1
    
    for state in state_list:
        rain.append(state[0])
        inflow.append(state[1])
        outflow.append(state[2])
        volume.append(state[3])
        level.append(state[4])
    
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 12)
    
    ax[0].set_xlabel('Time($min$)')
    ax[0].set_ylabel('Volume($m^3$)')
    ax[0].plot(inflow, color='r', marker='D', label= 'Inflow into the reservoir')
    ax[0].plot(outflow, color='b', marker='o', label= 'Outflow from the reservoir')
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Time($min$)')
    ax[1].set_ylabel('Elevation($m$)')
    line1, = ax[1].plot(level, color='b', marker='o', label= 'Reservoir Elevation')
    #ax[1].legend(loc='best')
    
    axa = ax[1].twinx()
    line2, = axa.plot(action, linestyle='None', color='g', marker='D', label='Pump Operation')
    axa.set_ylabel('Action Type')
    
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax[1].legend(lines, labels, loc="upper right")
    
    plt.tight_layout()
    plt.show()


def plot_performance(ftitle, state_list, actions, 
                     kind=['rain', 'inflow', 'outflow', 'vol', 'level']):
    """
    state_list와 actions

    Parameters
    ----------
    ftitle : TYPE
        DESCRIPTION.
    state_list : TYPE
        DESCRIPTION.
    actions : TYPE
        DESCRIPTION.
    kind : TYPE, optional
        DESCRIPTION. The default is ['rain', 'inflow', 'outflow', 'vol', 'level'].

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


def plot_fig_pumps(pumps:list, results:list, ftitle:str="Sample Title"):
    """
    5가지 펌프의 개별 에너지 소모량과 비용 + 총합 plot 
    (maintenance cost 제외)

    Parameters
    ----------
    pumps : list
        개별 펌프의 에너지 소모량과 비용
    results : list
        전체 펌프의 에너지 소모량과 비용 총합

    Returns
    -------
    None.

    """    
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 14)
    
    fig.suptitle(ftitle, y=0.99)   
    #markers = ['D', 's', 'o', '+', 'p', '*']
    markers = ['*', '*', '*', '*', '*', '*']
    #lines = [(0,(1,5)), ':', (1,(5,5)), '-.', '--', '-']
    lines = [(0,(1,5)), (3,(1,5)), (5,(1,5)), '-.', '--', '-']
    labels = [r'$P_1$100', r'$P_2$100', r'$P_3$100', r'$P_1$170', r'$P_2$170', r'$\bf{Total}$']
    colors = ['b', 'g', 'k', 'orange', (0.6, 0, 0, 0.6), 'r']
    ax[0].set_xlabel('Time($min$)')
    ax[0].set_ylabel('Energy Consumption($kw/h$)')
    for i, p in enumerate(pumps):
        ax[0].plot(p.energy_consumption_cum['consume'], c=colors[i], marker=markers[i], lw=2, ls=lines[i], label= labels[i])
    ax[0].plot(results['total']['energy_consume'], c=colors[-1], marker=markers[-1], lw=2, ls=lines[-1], label= labels[-1])
    x_ticks = [30*i for i in range(len(p.energy_consumption_cum['consume']))]
    ax[0].set_xticklabels(x_ticks)
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Time($min$)')
    ax[1].set_ylabel('Energy Cost($)')
    for i, p in enumerate(pumps):
        ax[1].plot(p.energy_cost_cum['consume'], c=colors[i], marker=markers[i], lw=2, ls=lines[i], label= labels[i])
    ax[1].plot(results['total']['cost_consume'], c=colors[-1], marker=markers[-1], lw=2, ls=lines[-1], label= labels[-1])
    ax[1].set_xticklabels(x_ticks)
    ax[1].legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    fpath = f"../results/{ftitle}"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    

def plot_individual_level(dlevel:list, mlevel:list, ftitle):
    """
    inp 개별 사례에 대한 2가지 펌프운용 적용시의 수위(level) 변화    

    Parameters
    ----------
    dlevel : list
        dqn 운용에 따른 수위
    mlevel : list
        manual 운용에 따른 수위

    Returns
    -------
    None.

    """    
    fig, ax = plt.subplots(1,1, sharex=False)
    fig.set_size_inches(15, 14)
    
    fig.suptitle(ftitle, y=0.99)   
    markers = ['s', 'o']
    lines = ['-.', '-']
    labels = [r'DQN', r'Manual']
    colors = ['b', 'g']
    ax.set_xlabel('Time($min$)')
    ax.set_ylabel('Reservoir level($m$)')
    
    ax.plot(dlevel, c=colors[0], marker=markers[0], ls=lines[0], label= labels[0])
    ax.plot(mlevel, c=colors[-1], marker=markers[-1], ls=lines[-1], lw=5, label= labels[-1])
    x_ticks = [30*i for i in range(len(dlevel))]
    ax.set_xticklabels(x_ticks)
    ax.legend(loc='best')
    
    
    plt.tight_layout()
    plt.show()
    
    fpath = f"../results/{ftitle}"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")

def plot_fig_221(dstruct:dict, ftitle:str):   
    """
    36일간의 펌프 유형별 에너지 소모량과 비용 + 총합 plot    

    Parameters
    ----------
    dstruct : dict
        개별 펌프의 에너지 소모량과 비용
        전체 펌프의 에너지 소모량과 비용 총합

    Returns
    -------
    None.

    """    
    
    ds = dstruct
    
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 14)
    
    fig.suptitle(ftitle, y=0.99)   
    markers = ['+', 's', 'o']
    lines = ['-.', '--', '-']
    labels = [r'P100', r'P170', r'$\bf{Total}$']
    colors = ['g', 'b', 'r']
    ax[0].set_xlabel('Days')
    ax[0].set_ylabel('Energy Consumption($kw/h$)')
    for i, key in enumerate(ds.keys()):
        ax[0].plot(ds[key]["energy_consume"], c=colors[i], marker=markers[i], ls=lines[i], label= labels[i])
    #ax[0].set_xticklabels([d for d in range(len(ds[key]["energy_consume"]))])
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel(r'Energy Cost($\$$)')
    for i, key in enumerate(ds.keys()):
        ax[1].plot(ds[key]['cost_consume'], c=colors[i], marker=markers[i], ls=lines[i], label= labels[i])
    #ax[1].set_xticklabels(x_ticks)
    ax[1].legend(loc='best')
    
    ax[0].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    ax[1].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    
    plt.tight_layout()
    plt.show()
    
    fpath = f"../results/{ftitle}"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    

def plot_fig_222(dstruct:dict, ftitle:str):
    """
    On/off cost composition displays for DQN or MAN

    Parameters
    ----------
    dqn_ds : dict
        DESCRIPTION.
    man_ds : dict
        DESCRIPTION.
    tfile : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ds = dstruct
    costs = ["cost_start_extra", "cost_wear", "cost_repair",
             "cost_down", "cost_onoff"]
    
    fig, ax = plt.subplots(1,3, sharex=False)
    fig.set_size_inches(45, 14)
    
    fig.suptitle(ftitle, y=0.99) 
    
    markers = ['+', 's', 'o', 'D', 'p']
    lines = [(0,(1,5)), (3,(1,5)), '-.', '--', '-']
    labels_p100 = [r'P100 start extra cost', r'P100 wear cost', r'P100 repair cost',
                   r'P100 down cost', r'P100 on/off cost']
    labels_p170 = [r'P170 start extra cost', r'P170 wear cost', r'P170 repair cost',
                   r'P170 down cost', r'P170 on/off cost']
    labels_total = [r'$\bf{Total}$ start extra cost', r'$\bf{Total}$ wear cost', r'$\bf{Total}$ repair cost',
                   r'$\bf{Total}$ down cost', r'$\bf{Total}$ on/off cost']               
    colors = ['orange', (0.6, 0, 0, 0.6), 'g', 'b', 'r']
    
    ax[0].set_xlabel('Days')
    ax[0].set_ylabel('Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[0].plot(ds['p100'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels_p100[i])
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel(r'Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[1].plot(ds['p170'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels_p170[i])
    ax[1].legend(loc='best')
    
    ax[2].set_xlabel('Days')
    ax[2].set_ylabel(r'Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[2].plot(ds['total'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels_total[i])
    ax[2].legend(loc='best')
    
    #ax[0].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    #ax[1].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    #ax[2].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    
    plt.tight_layout()
    plt.show()
    
    fpath = f"../results/{ftitle}"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")


def plot_fig_223(dqn_ds:dict, man_ds:dict, ftitle):
    """
    Maintenance cost consisted of on/off cost + dry run cost for DQN(A) and MAN(B)

    Parameters
    ----------
    dqn_ds : dict
        DESCRIPTION.
    man_ds : dict
        DESCRIPTION.
    ftitel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    """
    dryrun 발생에 따른 비용 plot

    Returns
    -------
    None.

    """
    
    costs = ["cost_onoff", "cost_dryrun", "cost_maintenance"]
    
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 14)
    
    fig.suptitle(ftitle, y=0.99) 
    
    markers = ['D', 'D', '+']
    lines = ['-.', '--', '-']
    labels = [r'$\bf{Total}$ On/off cost', r'$\bf{Total}$ Dry run cost', r'$\bf{Total}$ Maintenance cost']
    colors = ['g', 'b', 'r']
    
    ax[0].set_xlabel('Days')
    ax[0].set_ylabel('Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[0].plot(dqn_ds['total'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels[i])
        print('11', dqn_ds['total'][cost][-5:-1])
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel(r'Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[1].plot(man_ds['total'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels[i])
        print('12', man_ds['total'][cost][-5:-1])
    ax[1].legend(loc='best')
    
    
    ax[0].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    #ax[1].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    #ax[2].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    
    plt.tight_layout()
    plt.show()
    
    fpath = f"../results/{ftitle}"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")


def plot_fig_224(dqn_ds:dict, man_ds:dict, ftitle):
    pass    
    # "cost_consume":[0.0]
    # "cost_maintenance":[0.0]
    # "cost_total
    """
    Total cost consisted of consume cost + maintenance cost for DQN(A) and MAN(B)

    Parameters
    ----------
    dqn_ds : dict
        DESCRIPTION.
    man_ds : dict
        DESCRIPTION.
    ftitel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    costs = ["cost_consume", "cost_maintenance", "cost_total"]
    
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 14)
    
    fig.suptitle(ftitle, y=0.99) 
    
    markers = ['D', 'D', '+']
    lines = ['-.', '--', '-']
    labels = [r'$\bf{Total}$ Energy cost', r'$\bf{Total}$ Maintenance cost', r'$\bf{Total}$ Total cost']
    colors = ['g', 'b', 'r']
    
    ax[0].set_xlabel('Days')
    ax[0].set_ylabel('Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[0].plot(dqn_ds['total'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels[i])
        print('21', dqn_ds['total'][cost][-5:-1])
    ax[0].legend(loc='best')
    
    
    
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel(r'Cost($\$$)')
    for i, cost in enumerate(costs):
        ax[1].plot(man_ds['total'][cost], c=colors[i], marker=markers[i], ls=lines[i], label= labels[i])
        print('22', man_ds['total'][cost][-5:-1])
    ax[1].legend(loc='best')
    
    
    ax[0].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    ax[1].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    #ax[2].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    
    plt.tight_layout()
    plt.show()
    
    fpath = f"../results/{ftitle}"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")


def plot():
    """
    Water paper -> Figure 5

    Returns
    -------
    None.

    """
    x_ticks=['10', '60', '120', '180', '240', '360', '540', '720', '1080', '1440']
    elev = 9
    change = 16
    homedir = Path(__file__).parents[1]/'results'
    print(homedir)
    
    fname = ['dqn_gru_w1_n3.csv', 'dqn_gru_w1w2_n3.csv', 'man_policy.csv']
    
    ele_data = []
    cha_data = []
    for fn in fname:
        with open(homedir/fn) as fd:
            lines = list(csv.reader(fd))
            ele_data.append([float(i) for i in lines[elev]])
            cha_data.append([float(i) for i in lines[change]])
      
    
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 14)
    
    
    markers = ['o', 's', 'p']
    lines = ['-', '-.', '--']
    labels = ['DDQN($w_e$=1,$w_p$=0)', 'DDQN($w_e$=2,$w_p$=1)', 'Rule base']
    
    ax[0].set_xlabel('Duration($min$)')
    ax[0].set_ylabel('Elevation($m$)')
    for i, d in enumerate(ele_data):
        ax[0].plot(d, marker=markers[i], linestyle=lines[i], label= labels[i])
    ax[0].set_xticklabels(x_ticks)
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Duration($min$)')
    ax[1].set_ylabel('Number of changes')
    for i, d in enumerate(cha_data):
        ax[1].plot(d, marker=markers[i], linestyle=lines[i], label= labels[i])
    ax[1].set_xticklabels(x_ticks)
    ax[1].legend(loc='lower right')
       

def plot1():   
    """
    
    Water paper -> Figure 6.
    
    Parameters
    ----------
    state_list : TYPE
        DESCRIPTION.
    action_list : TYPE
        DESCRIPTION.
    ['rain', 'inflow', 'outflow', 'vol', 'level'] : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    x_ticks=['10', '60', '120', '180', '240', '360', '540', '720', '1080', '1440']
    elev = 9
    change = 16
    homedir = Path(__file__).parents[1]/'results'
    print(homedir)
    
    fname_w1 = ['dqn_gru_w1_n1.csv', 
                'dqn_gru_w1_n2.csv', 
                'dqn_gru_w1_n3.csv', 
                'dqn_gru_w1_n4.csv', 
                'dqn_gru_w1_n5.csv']
    fname_w1w2 = ['dqn_gru_w1w2_n1.csv', 
                  'dqn_gru_w1w2_n2.csv',
                  'dqn_gru_w1w2_n3.csv', 
                  'dqn_gru_w1w2_n4.csv',
                  'dqn_gru_w1w2_n5.csv']
    
    ele_data = []
    cha_data = []
    for fn in fname_w1:
        with open(homedir/fn) as fd:
            lines = list(csv.reader(fd))
            ele_data.append([float(i) for i in lines[elev]])
            cha_data.append([float(i) for i in lines[change]])
    
    ele_data1 = []
    cha_data1 = []
    for fn in fname_w1w2:
        with open(homedir/fn) as fd:
            lines = list(csv.reader(fd))
            ele_data1.append([float(i) for i in lines[elev]])
            cha_data1.append([float(i) for i in lines[change]])
    
    fig, ax = plt.subplots(2,2, sharex=False)
    fig.set_size_inches(25, 22)
   
    markers = ['+', 's', 'o', 'D', 'p']
    lines = [(0,(5,5)), '-.', '-', ':', '--' ]
    
    ax[0, 0].set_xlabel('Duration($min$)')
    ax[0,0].set_ylabel('Elevation($m$)')
    for i, d in enumerate(ele_data):
        ax[0,0].plot(d, marker=markers[i], linestyle=lines[i], label= f'δ: {i+1}')
    ax[0,0].set_xticklabels(x_ticks)
    ax[0,0].legend(loc='best')
    
    ax[1,0].set_xlabel('Duration($min$)')
    ax[1,0].set_ylabel('Number of changes')
    for i, d in enumerate(cha_data):
        ax[1,0].plot(d, marker=markers[i], linestyle=lines[i], label= f'δ: {i+1}')
    ax[1,0].set_xticklabels(x_ticks)
    ax[1,0].legend(loc='best')
    
    ax[0,1].set_xlabel('Duration($min$)')
    ax[0, 1].set_ylabel('Elevation($m$)')
    for i, d in enumerate(ele_data1):
        ax[0,1].plot(d, marker=markers[i], linestyle=lines[i], label= f'δ: {i+1}')
    ax[0,1].set_xticklabels(x_ticks)
    ax[0,1].legend(loc='best')
    
    ax[1,1].set_xlabel('Duration($min$)')
    ax[1,1].set_ylabel('Number of changes')
    for i, d in enumerate(cha_data1):
        ax[1,1].plot(d, marker=markers[i], linestyle=lines[i], label= f'δ: {i+1}')
    ax[1,1].set_xticklabels(x_ticks)
    ax[1,1].legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    test_inp = "data/gasan/30year/30yr_0060m_h182.inp"
    #test_inp = "data/gasan/100year/100yr_0010m_h001.inp"
    regularDqn_model ='../trained_models/dqn_gru_regular_w0_5_w0_5_w0_0_w0_0_min2_tr0_1_acttype0.pkl'
    gaDqn_model = "../trained_models/dqn_guided_GA_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_1_acttype0.pkl"
    model, actions, states, rewards, infos = dqn_from_demon_v1.test_model(None, \
            test_inp, minutes=2, model_path=gaDqn_model)
    fig_title =  test_inp.split('/')[-1]
    
    plot_individual_case(states, actions)
    
    # years = ['50']
    # durations = ['0120']

    # # Parameters for experiments
    # params = {"minutes":2, "init_cycles":15000}
    
    # # data acquisition
    # inp, train_inps, valid_inps, test_inps = perform_evaluate.stratified_split_data(years, durations, trate=0.7, vrate=0.1, stratify=True)
    # print(f'trains: {len(train_inps)}, tests: {len(test_inps)}')
    # test_inp = test_inps[0]

    # # Test for dqn_gru
    # model_path ='../trained_models/dqn_gru_pumpmodel_w2_w1.pkl'
    # model, actions, states, rewards, infos = dqn_gru_torch.test_model(None, test_inp, minutes=params["minutes"], model_path=model_path)
    # fig_title =  test_inp.split('/')[-1]
    # dqn_pumps = cost_model_v2.execute_scenario(actions, infos, minutes=params["minutes"], init_cycles=params["init_cycles"])
    # dqn_results = cost_model_v2.analysis_result(dqn_pumps)
    
    # #plt.figure()
    # plot_fig_pumps(dqn_pumps, dqn_results)
    