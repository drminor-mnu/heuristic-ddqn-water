#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:39:18 2023

@author: drminor
"""

import torch
from torchviz import make_dot
from water_gym import WaterGym, Actions
from dqn_model import DqnCNN

import numpy as np
import random
import copy
from collections import deque
from matplotlib import pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nstates = 3
actions = Actions

l1 = 15
l2 = 150
l3 = 100
l4 = len(actions)

# model = torch.nn.Sequential(
#     torch.nn.Linear(l1, l2),
#     torch.nn.ReLU(),
#     torch.nn.Linear(l2, l3),
#     torch.nn.ReLU(),
#     torch.nn.Linear(l3,l4)
# )


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
    with open('../data/selected_inp.txt', 'r') as fd:
        inp_list = fd.readlines()
        inp_list = [f.strip() for f in inp_list]
    
    random.shuffle(inp_list)
    nsample = len(inp_list) # the number of total sample
    ntrate = int(nsample * 0.7) # the number of training sample
    nvrate = int(nsample * 0.1) # the number of validation sample
    
    train_inps = inp_list[:ntrate]
    valid_inps = inp_list[ntrate:ntrate+nvrate]
    test_inps = inp_list[ntrate+nvrate:]
    
    return inp_list, train_inps, valid_inps, test_inps
    

def train(train_inps, valid_inps, epochs:int=1, minutes:int=1):
    """
    

    Parameters
    ----------
    train_inps : TYPE
        DESCRIPTION.
    valid_inps : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.
    losses : TYPE
        DESCRIPTION.

    """
    model = DqnCNN(len(actions)).to(device)
    
    model2 = copy.deepcopy(model).to(device)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    gamma = 0.9
    
    epochs = len(train_inps) 
    epsilon = 1.0
    losses = []
    mem_size = 100
    batch_size = 20
    replay = deque(maxlen=mem_size)
    sync_freq = 20 #A
    j=0
    nstates = 3 # how to many state data use (the number of previous states: 2 + current state: 1)
    
    for i in range(epochs): 
        print(i)
        gym = WaterGym(train_inps[i])
        state_, reward, done, info = gym.reset()
        
        input_q = deque([], nstates) # for build input vector
        for k in range(nstates-1): 
            input_q.append(np.array([0.0 for _ in range(len(state_))]))
        state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
        input_q.append(state_)
        state = torch.tensor(np.stack(input_q).reshape(-1),
                             dtype=torch.float32, device=device)
        
        status = 1
        while(status == 1): 
            j+=1
            state_prev = torch.Tensor.cpu(state)
            q_val = model(state.unsqueeze(0).unsqueeze(0))
            q_val_ = torch.Tensor.cpu(q_val).data.numpy()
            if (random.random() < epsilon):
                action = np.random.randint(0,len(actions))
            else:
                action = np.argmax(q_val_)
            #print(action)
            
            state_, reward, done, info = gym.step(action)
            state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state = torch.tensor(np.stack(input_q).reshape(-1),
                                 dtype=torch.float32, device=device)
            
            exp =  (state_prev, action, reward, torch.Tensor.cpu(state), done)
            replay.append(exp) #H
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)
                   
                Q1 = model(state1_batch.unsqueeze(1)) 
                with torch.no_grad():
                    Q2 = model2(state2_batch.unsqueeze(1))
                
                Y = reward_batch + gamma * (1-done_batch) * torch.max(Q2, dim=1)[0]
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                #print(i, loss.item())
             
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
            if done == True:
                status = 0
        if epsilon > 0.1:
            epsilon -= (1/epochs)
                
    losses = np.array(losses)
    
    torch.save(model.state_dict(), '../trained_models/water_dqn_cnn_dual.pkl')
    
    return model, losses


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

def test_model(model, test_inp, minutes:int=1):
    
    if model == None:
        model = DqnCNN(len(actions)).to(device)
        model.load_state_dict(torch.load('../trained_models/water_dqn_cnn_dual.pkl'))
        model.eval()
    
    gym = WaterGym(test_inp)
    init_state_, reward, done, info = gym.reset()
    input_q = deque([], nstates) # for build input vector
    for k in range(nstates-1): 
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = np.array(init_state_) / (np.array(init_state_).sum() + 0.00001) # input normalization
    input_q.append(state_)
    state = torch.tensor(np.stack(input_q).reshape(-1),
                         dtype=torch.float32, 
                         device=device)
    
    action_list = [0]
    state_list = [init_state_]
    reward_list = [reward]
    info_list = [info]
    while(1):
        print(state.unsqueeze(0).unsqueeze(0).shape)
        q_val = model(state.unsqueeze(0).unsqueeze(0))
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
        state = torch.tensor(np.stack(input_q).reshape(-1),
                             dtype=torch.float32,
                             device=device)
        
        if done == True:
            break
        
    return model, action_list, state_list, reward_list, info_list


def plot_performance(state_list, actions, 
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
        

def plot_reward(reward_list, info_list):
    pass
    

if __name__ == '__main__':
    total, train_list, valid_list, test_list = data_split(trate=0.5)
    
    # model, losses = train(train_list, valid_list)
    # plot_losses(losses)
    test_list = ['data/gasan/30year/30yr_0060m_h150.inp']
    tmodel, action_list, state_list, reward_list, info_list = test_model(None, test_list[0])
    plt.figure()
    plot_performance(state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])