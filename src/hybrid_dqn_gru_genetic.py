#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:44:13 2024

This file was replaced by dqn_from_domon.py

@author: drminor
"""


import torch
from torchviz import make_dot
from water_gym import WaterGym, Actions
from dqn_models import DqnGRU
from genetic_algo import *

import re
import numpy as np
import random
import math
import copy
import time
from collections import deque
from matplotlib import pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device}\n")
actions = Actions
nstates = 5 # how many states are use (the number of previous states: 2 + current state: 1)
num_layers = 3 # the number of layers of GRU model
hidden_dim = 32 # ther number of nodes for a layer
minutes = 2 # the interval minutes for simulation


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
    # random.shuffle(test_inps)
    
    return inp_list, train_inps, valid_inps, test_inps


def data_split(trate:float=0.7, vrate:float=0.1):
    """
    Data splition with training data rate, validation data rate, test data rate
    test data rate = 1.0 - (training data rate + validation data rate)
    
    Parameters
    ----------
    trate : TYPE, optional
        Training data rate. The default is 0.7.
    vrate : TYPE, optional
        Validation data rate. The default is 0.1.

    Returns
    -------
    train_inps : list
        train data set
    valid_inps : list
        validation data set
    test_inps : list
        test data set

    """
    with open('../data/selected_inp.txt', 'r') as fd: # 마이너스 강우량이 보정된 inp 파일 리스트 데이터
        inp_list = fd.readlines()
        inp_list = [f.strip() for f in inp_list]
    
    random.shuffle(inp_list)
    nsample = len(inp_list) # the number of total sample
    ntrate = int(nsample * trate) # the number of training sample
    nvrate = int(nsample * vrate) # the number of validation sample
    
    train_inps = inp_list[:ntrate]
    valid_inps = inp_list[ntrate:ntrate+nvrate]
    test_inps = inp_list[ntrate+nvrate:]
    
    return inp_list, train_inps, valid_inps, test_inps
    

def train(train_inps, valid_inps, hybrid:bool=True, epochs:int=1, minutes:int=1, valid:bool=False):
    """
    model training procedure
    
    Parameters
    ----------
    train_inps: list
        Training data set.
    valid_inps : list
        Validation data set.
    minutes : int, optional
        펌프 조합 의사결정 시간단위. 기본 값 1분
    
    Returns
    -------
    policy_model
        trained model
    train_epoch_losses: np.array
        losses computed each epoch against training data set
    valid_epoch_losses: np.array
        losses computed each epoch against validation data set
    """

    save_filename = '../trained_models/water_dqn_gru_hybrid.pkl' \
            if hybrid else '../trained_models/water_dqn_gru.pkl'
    train_inps = copy.deepcopy(train_inps)
    valid_inps = copy.deepcopy(valid_inps)
    # input_dim: the dimension of features, output_dim: the length of actions    
    policy_model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                          output_dim=len(actions), 
                          num_layers=num_layers).to(device)
    
    target_model = copy.deepcopy(policy_model).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    
    gamma = 0.1
    
    epsilon = 1.0
    train_epoch_losses = []
    valid_epoch_losses = []
    mem_size = 100
    batch_size = 20
    replay = deque(maxlen=mem_size)
    sync_freq = 10 # policy model과 target model 사이의 파라미터 일치 간격
    best_val = math.inf
    
    for epoch in range(epochs):
        print(f'\nEpoch # {epoch}/{epochs}')
        num_samples = len(train_inps)
        losses = []
        for i in range(num_samples): 
            print(f'\r{i+1}/{num_samples}', end='', flush=True)
            gym = WaterGym(train_inps[i], minutes)
            state_, reward, done, info = gym.reset()
            
            input_q = deque([], nstates) # for build input vector
            # build intial input 
            for k in range(nstates-1): 
                input_q.append(np.array([0.0 for _ in range(len(state_))]))
            state_n = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
            input_q.append(state_n)
            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            
            j = 0
            status = 1
            prev_action = 0
            while(status): 
                j+=1
                state_prev = torch.Tensor.cpu(state)
                q_val, hidden_state = policy_model(state.unsqueeze(0))
                q_val_ = torch.Tensor.cpu(q_val).data.numpy()
                if (random.random() < epsilon):
                    # generate solution using genetic algorithm and insert it replay memory
                    # While DQNGRU uses 5(input_dim) sequences in detemining an actions
                    # genetic algorithm uses only the last state 
                    if hybrid:
                        # !! notice: Input state must be an unnormalized one for the genetic algorithm
                        unnorm_state = state_
                        action = genetic_algorithm(unnorm_state, prev_action)
                        #print(f'hybrid:{unnorm_state} {action}')
                    else: # Random exploration
                        action = np.random.randint(0,len(actions))
                else:
                    action = np.argmax(q_val_)
                    #print(f'armax: {action}')
                prev_action = action
                
                #print(action)
                
                state_, reward, done, info = gym.step(action)
                state_n = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
                input_q.append(state_n)
                state = torch.tensor(np.stack(input_q),
                                     dtype=torch.float32, device=device)
                
                exp =  (state_prev, action, reward, torch.Tensor.cpu(state), done)
                replay.append(exp) 
                
                if len(replay) > batch_size:
                    minibatch = random.sample(replay, batch_size)
                    state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                    action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                    reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                    state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                    done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)
                       
                    Q1_gru, hidden_state = policy_model(state1_batch) 
                    with torch.no_grad():
                        Q2_gru, hidden_state = target_model(state2_batch)
                    
                    acts_ = torch.argmax(Q1_gru, dim=1).unsqueeze(dim=1)
                    tqvals = Q2_gru.gather(dim=1, index=acts_).squeeze()
                    Y = reward_batch + gamma * (1-done_batch) * tqvals
                    X = Q1_gru.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = loss_fn(X, Y.detach())
                    #print(i, loss.item())
                 
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    
                if j % sync_freq == 0:
                    target_model.load_state_dict(policy_model.state_dict())
                if done == True:
                    status = 0
            if epsilon > 0.2:
                epsilon -= (1/num_samples)
        train_epoch_losses.append(np.asarray(losses).mean())
        if valid: # validation process
            print("\nvalidation processing")
            valid_loss = compute_valid_loss(policy_model, target_model, valid_inps, 
                                            gamma, epsilon, batch_size, mem_size, minutes)
            valid_epoch_losses.append(valid_loss)
            if valid_loss < best_val:
                torch.save(policy_model.state_dict(), save_filename)
    print() # for output line adjustment                 
      
    if not valid:
        torch.save(policy_model.state_dict(), save_filename)
    
    
    return policy_model, np.array(train_epoch_losses), np.array(valid_epoch_losses) 


def compute_valid_loss(policy_net, target_net, valid_inps:list, 
                       gamma, epsilon, batch_size, mem_size, minutes:int):
    
    valid_epoch_losses = []
    batch_size = 20
    replay = deque(maxlen=mem_size)
    
    q_net = copy.deepcopy(policy_net).to(device)
    q_net.load_state_dict(policy_net.state_dict())
    q_net.eval()
    
    t_net = copy.deepcopy(target_net).to(device)
    t_net.load_state_dict(target_net.state_dict())
    
    num_samples = len(valid_inps)
    losses = []
    for i in range(num_samples): 
        print(f'\r{i+1}/{num_samples} validation', end='', flush=True)
        gym = WaterGym(valid_inps[i], minutes)
        state_, reward, done, info = gym.reset()
        
        input_q = deque([], nstates) # for build input vector
        # build intial input 
        for k in range(nstates-1): 
            input_q.append(np.array([0.0 for _ in range(len(state_))]))
        state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
        input_q.append(state_)
        state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
    
        j = 0
        status = 1
        while(status): 
            j+=1
            state_prev = torch.Tensor.cpu(state)
            q_val, hidden_state = q_net(state.unsqueeze(0))
            q_val_ = torch.Tensor.cpu(q_val).data.numpy()
            action = np.argmax(q_val_)
            #print(action)
            
            state_, reward, done, info = gym.step(action)
            state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state = torch.tensor(np.stack(input_q),
                                 dtype=torch.float32, device=device)
            
            exp =  (state_prev, action, reward, torch.Tensor.cpu(state), done)
            replay.append(exp) 
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)
                   
                Q1_gru, hidden_state = q_net(state1_batch) 
                with torch.no_grad():
                    Q2_gru, hidden_state = t_net(state2_batch)
                
                acts_ = torch.argmax(Q1_gru, dim=1).unsqueeze(dim=1)
                tqvals = Q2_gru.gather(dim=1, index=acts_).squeeze()
                Y = reward_batch + gamma * (1-done_batch) * tqvals
                X = Q1_gru.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = torch.nn.functional.mse_loss(X, Y.detach())
                losses.append(loss.item())
            if done == True:
                status = 0
                
    return np.asarray(losses).mean()


def plot_losses(losses):
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Loss",fontsize=22)


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

def test_model(model, test_inp, hybrid:bool=True, minutes:int=1):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_filename = '../trained_models/water_dqn_gru_hybrid.pkl' \
            if hybrid else '../trained_models/water_dqn_gru.pkl'
    if model == None:
        model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                       output_dim=len(actions), num_layers=num_layers).to(device)
        model.load_state_dict(torch.load(save_filename))
        model.eval()
    
    gym = WaterGym(test_inp, minutes)
    init_state_, reward, done, info = gym.reset()
    input_q = deque([], nstates) # for build input vector
    for k in range(nstates-1): 
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = np.array(init_state_) / (np.array(init_state_).sum() + 0.00001) # input normalization
    input_q.append(state_)
    state = torch.tensor(np.stack(input_q),
                         dtype=torch.float32, 
                         device=device)
    
    action_list = [0]
    state_list = [init_state_]
    reward_list = [reward]
    info_list = [info]
    while(1):
        #print(state.unsqueeze(0).unsqueeze(0).shape)
        q_val, hidden_state = model(state.unsqueeze(0))
        q_val_ = torch.Tensor.cpu(q_val).data.numpy()
        #print(q_val_)
        action = np.argmax(q_val_)
        action_list.append(action)
        
        state_, reward, done, info = gym.step(action)
        
        # for performance evaluations
        state_list.append(state_) 
        reward_list.append(reward)
        info_list.append(info)
        
        state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
        input_q.append(state_)
        state = torch.tensor(np.stack(input_q),
                             dtype=torch.float32,
                             device=device)
        
        if done == True:
            break
        
    return model, action_list, state_list, reward_list, info_list


def plot_performance(ftitle:str, state_list, actions, 
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
    
    if not ftitle:
        ftitle = "Test"
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
        

def plot_individual_case(state_list, action_list):
    """
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

    """
    rain = []
    inflow = []
    outflow = []
    volume = []
    level = []
    
    for state in state_list:
        rain.append(state[0])
        inflow.append(state[1])
        outflow.append(state[2])
        volume.append(state[3])
        level.append(state[4])
    
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 23)
    
    label_size = 25
    tick_size = 20
    legend_size = 25
    
    ax[0].set_xlabel('Time($min$)', fontsize=label_size, fontweight='bold')
    ax[0].set_ylabel('Volume($m^3$)', fontsize=label_size, fontweight='bold')
    ax[0].plot(inflow, color='r', marker='D', label= 'inflow into the reservoir')
    ax[0].plot(outflow, color='b', marker='o', label= 'outflow from the reservoir')
    ax[0].tick_params(labelsize=tick_size)
    ax[0].legend(loc='best', fontsize=legend_size)
    
    ax[1].set_xlabel('Time($min$)', fontsize=label_size, fontweight='bold')
    ax[1].set_ylabel('Elevation($m$)', fontsize=label_size, fontweight='bold')
    ax[1].plot(level, color='b', marker='o', label= 'reservoir elevation')
    ax[1].legend(loc='best', fontsize=legend_size)
    
    plt.tight_layout()
    plt.show()


def plot_reward(reward_list, info_list):
    pass
    

if __name__ == '__main__':
    
    years = ['10']#, '20']#, '30', '50', '80', '100']
    durations = ['0010', '0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    minutes = 2  
    epochs = 1

    inp, train_inps, valid_inps, test_inps = stratified_split_data(years, 
                                                        durations, trate=0.05, 
                                                         vrate=0.01, stratify=True)
    # train_start = time.time()
    train_inps = ['data/gasan/10year/10yr_1080m_h1097.inp']
    model, train_losses, valid_losses = train(train_inps, valid_inps, 
                                              hybrid=True, epochs=epochs, 
                                              minutes=minutes, valid=False)
    # train_end = time.time()
    # print(f"Elapsed time to train: {train_end - train_start} seconds\n")
    
    ## plot_losses(losses) # the number of epochs are 1, so it doesn't have meaning
    
    # test_start = time.time()
    # test_list = ['data/gasan/30year/30yr_0720m_h954.inp']
    # tmodel, action_list, state_list, reward_list, info_list = test_model(model, test_list[0], minutes=minutes)
    # test_end = time.time()
    # print(f"Elapsed time to test: {test_end - test_start} seconds\n")
    
    
    ## plt.figure()
    ## plot_individual_case(state_list, action_list)
    ## plt.figure()
    ## fig_title =  test_list[0].split('/')[-1]
    ## plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])
