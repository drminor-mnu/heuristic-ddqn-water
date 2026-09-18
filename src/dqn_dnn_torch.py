#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:39:18 2023

@author: drminor
"""

import torch
from torchviz import make_dot
from water_gym import WaterGym, Actions
from dqn_model import DqnDNN, DqnCNN
#from perform_evaluate import plot_performance

import numpy as np
from scipy.special import softmax
import random
import copy
from collections import deque
from matplotlib import pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nstates = 3
actions = Actions

#l1 = 15
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


def data_split(trate:float=0.7, vrate:float=0.0):
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
    ntrate = int(nsample * trate) # the number of training sample
    nvrate = int(nsample * vrate) # the number of validation sample
    
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
    model = DqnDNN(nstates*5, l2, l3, l4).to(device)
    
    model2 = copy.deepcopy(model).to(device)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    gamma = 0.0
    
    epochs = len(train_inps) 
    epsilon = 1.0
    losses = []
    mem_size = 100
    batch_size = 20
    replay = deque(maxlen=mem_size)
    sync_freq = 20 #A
    j = 0
    # nstates = 3 # how many states of data use (the number of previous states: 2 + current state: 1)
    
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
            q_val = model(state)
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
                
                # print(state1_batch.shape)
                
                Q1 = model(state1_batch) 
                with torch.no_grad():
                    Q2 = model2(state2_batch)
                
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
    
    torch.save(model.state_dict(), '../trained_models/water_dqn_dnn_dual.pkl')
    
    return model, losses


def train_iter(train_inps, valid_inps, niter=3, nstates=3):
    """
    전체 데이터 셋을 대상으로 여러번 추가 학습

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
    # model = DqnDNN(nstates*5, l2, l3, l4).to(device)
    model = DqnCNN(len(actions)).to(device)
    
    model2 = copy.deepcopy(model).to(device)
    model2.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    gamma = 1.0
    
    epochs = len(train_inps) 
    #epsilon = 1.0
    losses = []
    mem_size = 100
    batch_size = 20
    replay = deque(maxlen=mem_size)
    sync_freq = 20 #A
    j=0
    # nstates = 3 # how many states of data use (the number of previous states: 2 + current state: 1)
    
    for n_i in range(niter):
        epsilon = 1.0
        for i in range(epochs): 
            print(f'Iter: {n_i}, {i}')
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
                q_val = model(state)
                q_val_ = torch.Tensor.cpu(q_val).data.numpy()
                if (random.random() < epsilon):
                    action = np.random.randint(0,len(actions))
                else:
                    # epsilon greedy 
                    # action = np.argmax(q_val_)
                    
                    # epsilon greedy + softmax distribution
                    action = np.random.choice(len(actions), 1, p=softmax(q_val_))[0]
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
                    
                    # print(state1_batch.shape)
                    
                    Q1 = model(state1_batch) 
                    with torch.no_grad():
                        Q2 = model2(state2_batch)
                    
                    Y = reward_batch + gamma * (1-done_batch) * torch.max(Q2, dim=1)[0]
                    X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = loss_fn(X, Y.detach())
                    #print(i, loss.item())
                 
                    optimizer.zero_grad()
                    loss.backward()
                    #losses.append(loss.item())
                    optimizer.step()
                    
                    if j % sync_freq == 0:
                        model2.load_state_dict(model.state_dict())
                if done == True:
                    status = 0
            if epsilon > 0.1:
                epsilon -= (1/epochs)
            losses.append(loss.item())
            
    losses = np.array(losses)
    
    torch.save(model.state_dict(), '../trained_models/water_dqn_dnn_dual.pkl')
    
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
        model = DqnDNN(nstates*5, l2, l3, l4).to(device)
        model.load_state_dict(torch.load('../trained_models/water_dqn_dnn_dual.pkl'))
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
        q_val = model(state)
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


if __name__ == '__main__':
    total, train_list, valid_list, test_list = data_split()
    
    model, losses = train_iter(train_list[:500], valid_list, 2)
    plot_losses(losses)
    test_list = ['data/gasan/30year/30yr_0060m_h150.inp']
    tmodel, action_list, state_list, reward_list, info_list = test_model(model, test_list[0])
    plt.figure()
    fig_title =  test_list[0].split('/')[-1]
#    plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])