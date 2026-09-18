#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:44:13 2024

@author: drminor

improved contents of v2 
1. validation loss 
2. reward performance indicator per episode
included 
"""


import torch
from torchviz import make_dot
from water_gym import WaterGym, Actions
from dqn_model import DqnGRU
from plot_style import set_plot_style

import re
import numpy as np
import random
import copy
from collections import deque
from matplotlib import pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_plot_style()

actions = Actions
nstates = 5 # how to many state data use (the number of previous states: 2 + current state: 1)
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
    inp_list : TYPimport genetic_algo
E
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
    

def train(train_inps, valid_inps, model_path:str=None, **params):
    """
    DQN + GRU ���� ����

    Parameters
    ----------
    train_inps: TYPE
        Training data rate. The default is 0.7.
    valid_inps : TYPE, optional
        Validation data rate. The default is 0.1.
    epochs : int, optional
        epochs of training
    minutes : int, optional
        펌프 조합 의사결정 시간단위. 기본 값 1분
    model_path : str, optional
        the path of model for saving       

    Returns
    -------
    policy_model : torch.nn.Module
        trained policy network
    train_epoch_losses : np.ndarray
        train average TD-loss per epoch
    valid_epoch_losses : np.ndarray
        validation average TD-loss per epoch
    train_epoch_rewards : np.ndarray
        train total episodes average reward per epoch
    valid_epoch_rewards : np.ndarray
        validation total episodes average reward per epoch
    """

    # params 
    epochs  = params.get('epochs', 1)
    minutes = params.get('minutes', 1)
    weights = params.get('weights', [1, 1, 1, 0])
    
    train_inps = copy.deepcopy(train_inps)    
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
    train_epoch_rewards = []   # <<< NEW
    valid_epoch_rewards = []   # <<< NEW

    mem_size = 100
    batch_size = 20
    replay = deque(maxlen=mem_size)
    sync_freq = 10 #A
    
    for epoch in range(epochs):
        print(f'Epoch # {epoch+1}/{epochs}')
        num_samples = len(train_inps)
        losses = []
        train_episode_returns = []   # <<< NEW: train total reward save for each episode

        for i in range(num_samples): 
            print(f'\rTrain {i+1}/{num_samples}', end='', flush=True)
            gym = WaterGym(train_inps[i], minutes, weights)
            state_, reward, done, info = gym.reset()
            
            input_q = deque([], nstates) # for build input vector
            # build initial input 
            for k in range(nstates-1): 
                input_q.append(np.array([0.0 for _ in range(len(state_))]))
            state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            
            j = 0
            status = 1
            total_reward = 0.0      # <<< NEW: total reward for this episode
            while status: 
                j += 1
                state_prev = torch.Tensor.cpu(state)
                q_val, hidden_state = policy_model(state.unsqueeze(0))
                q_val_ = torch.Tensor.cpu(q_val).data.numpy()
                if (random.random() < epsilon):
                    action = np.random.randint(0,len(actions))
                else:
                    action = np.argmax(q_val_)
                
                state_, reward, done, info = gym.step(action)
                total_reward += reward   # <<< NEW: reward accumulation

                state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
                input_q.append(state_)
                state = torch.tensor(np.stack(input_q),
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
                       
                    Q1_gru, hidden_state = policy_model(state1_batch) 
                    with torch.no_grad():
                        Q2_gru, hidden_state = target_model(state2_batch)
                    
                    # DDQN target
                    acts_ = torch.argmax(Q1_gru, dim=1).unsqueeze(dim=1)
                    tqvals = Q2_gru.gather(dim=1, index=acts_).squeeze()
                    Y = reward_batch + gamma * (1-done_batch) * tqvals
                    X = Q1_gru.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = loss_fn(X, Y.detach())
                 
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    
                if j % sync_freq == 0:
                    target_model.load_state_dict(policy_model.state_dict())
                if done:
                    status = 0

            # 이 에피소드의 총 reward 저정
            train_episode_returns.append(total_reward)

            # epsilon decay per episode
            if epsilon > 0.2:
                epsilon -= (1/num_samples)

        # ---- train loss & reward (epoch) ----
        train_epoch_losses.append(np.asarray(losses).mean() if len(losses) > 0 else np.nan)
        train_epoch_rewards.append(np.asarray(train_episode_returns).mean() if len(train_episode_returns) > 0 else np.nan)
        print(f" Loss: {train_epoch_losses[-1]}, Reward: {train_epoch_rewards[-1]}")
        
        # ===========================
        #   VALIDATION PHASE
        # ===========================
        if valid_inps is not None and len(valid_inps) > 0:
            val_losses = []
            val_episode_returns = []   # <<< NEW: validation 에피소드별 총 reward
        
            with torch.no_grad():
                num_valid = len(valid_inps)
                for vi in range(num_valid):
                    print(f'\rValid {vi+1}/{num_valid}', end='', flush=True)

                    gym_v = WaterGym(valid_inps[vi], minutes, weights)
                    state_, reward, done, info = gym_v.reset()

                    input_q_v = deque([], nstates)
                    for _ in range(nstates - 1):
                        input_q_v.append(np.zeros_like(state_, dtype=float))

                    state_ = np.array(state_, dtype=float)
                    state_ = state_ / (state_.sum() + 1e-5)
                    input_q_v.append(state_)
                    state_v = torch.tensor(np.stack(input_q_v), dtype=torch.float32, device=device)

                    total_reward_v = 0.0
                    status_v = 1

                    while status_v:
                        # greedy ����
                        q_val_v, hidden_state_v = policy_model(state_v.unsqueeze(0))
                        q_val_v_np = q_val_v.detach().cpu().numpy()
                        action_v = int(np.argmax(q_val_v_np))

                        # ���� ����
                        next_state_raw_v, reward, done, info = gym_v.step(action_v)
                        total_reward_v += reward

                        next_state_raw_v = np.array(next_state_raw_v, dtype=float)
                        next_state_raw_v = next_state_raw_v / (next_state_raw_v.sum() + 1e-5)
                        input_q_v.append(next_state_raw_v)
                        state_v_next = torch.tensor(np.stack(input_q_v), dtype=torch.float32, device=device)

                        # --- TD-loss ���� (replay ���� ����) ---
                        q1, _ = policy_model(state_v.unsqueeze(0))       # ���� ���� Q(s,��)
                        q2, _ = target_model(state_v_next.unsqueeze(0))  # ���� ���� Q_target(s',��)

                        # train ���� ������ DDQN ������ ���� (argmax�� Q1 ����)
                        next_acts = torch.argmax(q1, dim=1, keepdim=True)     # (1,1)
                        target_q_next = q2.gather(1, next_acts).squeeze(1)[0] # scalar

                        done_tensor = torch.tensor(float(done), device=device)
                        reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32)

                        y_v = reward_tensor + gamma * (1.0 - done_tensor) * target_q_next
                        pred_q_v = q1[0, action_v]

                        loss_v = loss_fn(pred_q_v, y_v)
                        val_losses.append(loss_v.item())
                        # --------------------------------------

                        state_v = state_v_next

                        if done:
                            status_v = 0

                    val_episode_returns.append(total_reward_v)

            valid_epoch_losses.append(np.asarray(val_losses).mean() if len(val_losses) > 0 else np.nan)
            valid_epoch_rewards.append(np.asarray(val_episode_returns).mean() if len(val_episode_returns) > 0 else np.nan)
        else:
            valid_epoch_losses.append(np.nan)
            valid_epoch_rewards.append(np.nan)
            
        print(f" Loss: {valid_epoch_losses[-1]}, Reward: {valid_epoch_rewards[-1]}")
        print()  # for output line adjustment                  

    print() # for output line adjustment                  
    
    train_epoch_losses = np.array(train_epoch_losses)
    valid_epoch_losses = np.array(valid_epoch_losses)
    train_epoch_rewards = np.array(train_epoch_rewards)
    valid_epoch_rewards = np.array(valid_epoch_rewards)

    if model_path is None:
        model_path = '../trained_models/dqn_gru_default.pkl'
    torch.save(policy_model.state_dict(), model_path)
    
    return policy_model, train_epoch_losses, valid_epoch_losses, train_epoch_rewards, valid_epoch_rewards



def plot_losses(train_epoch_losses, valid_epoch_losses):
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 7)
    ax.plot(train_epoch_losses, label='train loss')
    ax.plot(valid_epoch_losses, label='valid loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()




def plot_reward(train_epoch_rewards, valid_epoch_rewards):
    fig, ax = plt.subplots()
    fig.set_size_inches(30,7)
    ax.plot(train_epoch_rewards, label='train reward')
    ax.plot(valid_epoch_rewards, label='valid reward')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Reward")
    
    plt.tight_layout()
    plt.show()
    


def test_model(model, test_inp, minutes:int=1, model_path:str=None):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model == None:
        if model_path == None:
            path = '../trained_models/dqn_gru_pumpmodel_w2_w1.pkl'
        else:
            path = model_path
        model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                       output_dim=len(actions), num_layers=num_layers).to(device)
        model.load_state_dict(torch.load(path))
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

if __name__ == '__main__':
    
    years = ['10', '20', '30', '50', '80', '100']
    durations = ['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    exper_params = {'minutes':2, 'epochs':20, 'weights': [1.0, 1.0, 1.0, 0.0]}

    inp, train_inps, valid_inps, test_inps = stratified_split_data(years, durations, trate=0.7, vrate=0.2, stratify=True)
    model_path = '../trained_models/dqn_gru_pumpmodel_w1_w1_w1_min2_epo20.pkl'
    model, train_epoch_losses, valid_epoch_losses, train_epoch_rewards, valid_epoch_rewards = train(train_inps, valid_inps, model_path=model_path, **exper_params)
    print(f"Completed training: {model_path.split('/')[-1]}")
    plot_losses(train_epoch_losses, valid_epoch_losses)
    plot_reward(train_epoch_rewards, valid_epoch_rewards)
    
    
    # plot_losses(losses) # the number of epochs are 1, so it doesn't have meaning
    test_list = ['data/gasan/50year/50yr_0120m_h371.inp']
    tmodel, action_list, state_list, reward_list, info_list = test_model(model, test_list[0], minutes=minutes)
    
    plt.figure()
    plot_individual_case(state_list, action_list)
    plt.figure()
    fig_title =  test_list[0].split('/')[-1]
    plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])
