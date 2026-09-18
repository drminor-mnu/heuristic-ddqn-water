#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:44:13 2024

@author: drminor

improved contents of v2 
1. validation loss 
2. reward performance indicator per episode
3. DDQN 구현 오류 수정
included
"""

# -----------------------------------------------------------------------
# UNUSED LEGACY CODE NOTICE (2026-09-04, repo-wide grep-verified):
#   demonstraion_replay()   lines 176-229
#   pre_train()             lines 230-271
#   train_guided_byGA_old() lines 272-451
# are not called anywhere in this repository. They correspond to an
# earlier two-phase (offline preload + online guidance) training
# design described in the original manuscript's Algorithm 3, which
# was superseded by the single-phase online-guidance design actually
# used (train_guided_byGA/byPSO/byENUM -> _train_guided_by_optimizer).
# Kept for reference rather than deleted, as a record of the
# discrepancy between the described and the implemented procedure.
# -----------------------------------------------------------------------


import torch
from torchviz import make_dot
from water_gym import WaterGym, Actions0, Actions1
from dqn_model import DqnGRU
from plot_style import set_plot_style
from genetic_algo import *
from pso import ParticleSwarmOptimization
from enum_baseline import EnumSearch

import re
import numpy as np
import random
import copy
from collections import deque
from matplotlib import pylab as plt
import time

from plot_style import set_plot_style

set_plot_style()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device}\n")

set_plot_style()

#actions = Actions
nstates = 5 # the number of consecutive states as input sequence for GRU (the number of previous states: 4 + current state: 1)
num_layers = 3 # the number of layers of GRU model
hidden_dim = 32 # ther number of nodes for a layer

# --- input-state standardization (E3 fix, 2026-08-28) -----------------------
# Was:  x / (x.sum() + 1e-5)  -- divides 5 physically incommensurate variables
# (rain, inflow, outflow, volume, level) by their sum. No physical meaning, and
# it collapses the level component to ~0 exactly when volume is large (the flood
# danger zone), so the network can barely see "level is high". water_gym.py
# already defines maxval/minval for proper standardization but they were unused.
# This restores the intended min-max standardization. State order:
#   [rain, inflow, outflow, volume, level]
_NORM_MIN = np.array([0.0, 0.0, 0.0, 0.0, 4.7])
_NORM_MAX = np.array([5.55, 1750.0, 1750.0, 32522.0, 10.0])

def _input_norm(vec):
    a = np.asarray(vec, dtype=np.float32)
    return ((a - _NORM_MIN) / (_NORM_MAX - _NORM_MIN)).astype(np.float32)
# ---------------------------------------------------------------------------
minutes = 2 # the interval minutes for simulation
MemSize = 3000 # the size of replay buffer
 

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


def data_split(years, durations, trate:float=0.7, vrate:float=0.1):
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
    

def demonstraion_replay(replay:deque, demo_inps:list, **params):
    """UNUSED -- retained for reference only. This function is not called
    anywhere in the repository. See docs/response/LEGACY_CODE.md.

    ToU (Time of Use)
    A replay buffer for pre-training is generated from the given demonstration data (inps)
    using a Genetic Algorithm.

    Parameters
    ----------
    replay : deque
        replay buffer 
    train_inps : list
        DESCRIPTION.
    n_episodes : int, optional
        DESCRIPTION. The default is 20.
    **params : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    minutes = params.get('minutes', 2)
    n_episodes = len(demo_inps)

    # (state_prev, action, reward, state_post, done)
    for inp in demon_inps[:n_episodes]:
        GA = GeneticAlgorithm(**params)
        actions, states, rewards, infos = GA.run_genetic_algo(inp)
        input_q = deque([], nstates) # for build input vector
        # build initial input 
        for k in range(nstates-1): 
            input_q.append(np.array([0.0 for _ in range(len(states[0]))]))
        done = False
        for i in range(len(actions)-1):
            state_ = np.array(states[i]) / (np.array(states[i]).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state_prev = torch.tensor(np.stack(input_q), dtype=torch.float32)
            
            state_ = np.array(states[i+1]) / (np.array(states[i+1]).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state_post = torch.tensor(np.stack(input_q), dtype=torch.float32)
            
            "genetic algorithm의 경우 state[0] -> action[1] -> state[1] 순서로 진행됨"
            if i >= len(actions)-2:
                done = True
            exp =  (state_prev, actions[i+1], rewards[i+1], state_post, done) 
            replay.append(exp)
    


def pre_train(train_inps, replay, rate:float=0.1, **params):
    """UNUSED -- retained for reference only. This function is not called
    anywhere in the repository. See docs/response/LEGACY_CODE.md.

    A subset of the training data is selected to generate demonstration data,
    which is then used to pre-train the network.

    Parameters
    ----------
    train_inps : TYPE
        DESCRIPTION.
    rate : float, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    None.

    """
    ## Selecting the demonstration dataset corresponing to the rate, r 
    train_inps.sort()
    years = list(set([re.findall(r'\d+', f.split('/')[-1])[0] for f in train_inps]))
    durations = list(set([re.findall(r'\d+', f.split('/')[-1])[1] for f in train_inps]))
    demo_inps = []
    for y in years:
        for m in durations:
            data = [f for f in train_inps 
                        if re.findall(r'\d+', f.split('/')[-1])[0] == y
                        and re.findall(r'\d+', f.split('/')[-1])[1] == m]
            random.shuffle(data)
            nsample = len(data) # the number of total sample
            n_rate = int(nsample * rate) # the number of training sample
            
            demo_inps.extend(data[:n_rate])
            
    random.shuffle(demo_inps)
    
    demonstraion_replay(replay, demo_inps, **params)
    
    return demo_inps
    

def train_guided_byGA_old(train_inps, model_path:str=None, **params):
    """UNUSED -- retained for reference only. This function is not called
    anywhere in the repository. See docs/response/LEGACY_CODE.md.

    DQN + GRU with demonstration from genetic algorithm

    Parameters
    ----------
    train_inps: TYPE
        Training data rate. The default is 0.7.
    model_path : str, optional
        the path of model for saving       

    Returns
    -------
    policy_model : torch.nn.Module
        trained policy network
    train_epoch_losses : np.ndarray
        train average TD-loss per epoch
    train_epoch_rewards : np.ndarray
        train total episodes average reward per epoch
    """

    # params 
    epochs  = params.get('epochs', 1)
    minutes = params.get('minutes', 1)
    weights = params.get('weights', [1, 1, 1, 0])
    ga_rate = params.get('ga_rate', 0.1)
    
    org_train_inps = copy.deepcopy(train_inps)    
    policy_model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                          output_dim=len(actions), 
                          num_layers=num_layers).to(device)
    
    target_model = copy.deepcopy(policy_model).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    
    gamma = 0.1
    initial_epsilon = 1.0
    epsilon = initial_epsilon
    batch_size = 20
    replay = deque(maxlen=MemSize)
    sync_freq = 10 # 주기를 좀더 크게 해보자.
    guided_train_count = []
    
    update_count = 0
    for epoch in range(epochs): # 실제 epoch는 1번으로 의미 없음. 각 에피소드를 대상으로 loss 변화 측정 및 분석
        print(f'Epoch # {epoch+1}/{epochs}')
        train_inps = org_train_inps
        random.shuffle(train_inps)
        num_samples = len(train_inps)
        
        train_losses = [] # for only one epoch, 각 에피소드마다 로스 저장, 만약 epoch 별로 계산하려면 수정 필요
        train_rewards = [] # for only one epoch, 각 에피소드마다 보상 저장   
        
        for i in range(num_samples):
            print(f'\rTrain {i+1}/{num_samples}', end='', flush=True)
            
            # Instance of Genetic algorithm which would be used as a action selection to guide learning 
            action_list = [0]
            GA = GeneticAlgorithm(**params)
            
            gym = WaterGym(train_inps[i], minutes, weights)
            state_, reward, done, info = gym.reset()
            ga_state = state_  # GA에서 사용하는 state은 network에서 사용하는 state와 구별
            input_q = deque([], nstates) # for build input vector
            # build initial input 
            for k in range(nstates-1): 
                input_q.append(np.array([0.0 for _ in range(len(state_))]))
            state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            
            status = 1
            losses = [] # i번째 에피소드에서 훈련 업데이트마다 loss 저장, 나중에 평균을 계산해서 해당 에피소드의 loss로 사용
            total_reward = 0.0      # <- total reward for this episode
            guided_count = 0
            
            while status: 
                state_prev = torch.Tensor.cpu(state)
                q_val, hidden_state = policy_model(state.unsqueeze(0))
                q_val_ = torch.Tensor.cpu(q_val).data.numpy()
                
                #########################
                ## GA guided learning  ##
                #########################
                
                if (epsilon > ga_rate * initial_epsilon):
                    # if (random.random() < epsilon):
                    #     action = GA.genetic_algorithm(ga_state, action_list)
                    # else:
                    #     action = np.argmax(q_val_)
                        action = GA.genetic_algorithm(
                            ga_state, action_list,
                            next_inflow=gym.outfalls[gym.clock + 1],
                        )
                        ga_count += 1
                else:
                    if (random.random() < epsilon):
                        action = np.random.randint(0,len(actions))
                    else:
                        action = np.argmax(q_val_)
                    # action = np.argmax(q_val_)
                
                state_, reward, done, info = gym.step(action)
                ga_state = state_
                total_reward += reward   # <- reward accumulation

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
                    
                    # 수정된 DDQN
                    Q1_gru, hidden_state = policy_model(state1_batch)  # Q(s, ��)  �� ����(X)��

                    with torch.no_grad():
                        # (1) 다음 상태에서 action 선택은 online(policy)로
                        Q1_next, _ = policy_model(state2_batch)  # Q(s', .) (online)
                        acts_ = torch.argmax(Q1_next, dim=1, keepdim=True)  # a* = argmax_a Q_online(s',a)
                    
                        # (2) 다음 상태에서 그 action의 가치는 target으로 평가
                        Q2_gru, _ = target_model(state2_batch)   # Q(s', .) (target)
                        tqvals = Q2_gru.gather(dim=1, index=acts_).squeeze(1)
                    
                    Y = reward_batch + gamma * (1 - done_batch) * tqvals
                    X = Q1_gru.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze(1)
                    
                    loss = loss_fn(X, Y)  # Y는 no_grad라 detach 불필요(해도 무방)
                 
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    
                    update_count += 1
                    if update_count % sync_freq == 0:
                        target_model.load_state_dict(policy_model.state_dict())
                if done:
                    status = 0
        
            # epsilon decay per episode 
            # 이 부분의 효과 분석 필요 GA를 어느 정도 사용하느냐? 학습 시간, 보상, 성능 지표등에 미치는 영향 분석
            if epsilon > 0.2:
                epsilon -= (1./num_samples)

            # ---- train loss & reward  ----
            train_losses.append(np.asarray(losses).mean() if len(losses) > 0 else np.nan)
            train_rewards.append(total_reward)
            print(f" Loss: {train_losses[-1]}, Reward: {train_rewards[-1]}")
        
       
    print() # for output line adjustment                  
    
    train_losses = np.array(train_losses)
    train_rewards = np.array(train_rewards)
    
    print(model_path)
    if model_path is None:
        model_path = '../trained_models/ddqn_gru_ga_guided_default.pkl'
    torch.save(policy_model.state_dict(), model_path)
    
    return policy_model, train_losses, train_rewards


def _train_guided_by_optimizer(train_inps, model_path: str = None,
                               optimizer_class=GeneticAlgorithm,
                               optimizer_method='genetic_algorithm',
                               guide_label='GA', **params):
    """
    DDQN + GRU with heuristic-guided exploration.

    Recommended action selection:
    1) Use GA with probability ga_prob
    2) Otherwise use epsilon-greedy on Q-network
    """

    import copy
    import random
    import numpy as np
    import torch
    from collections import deque

    # ---------------------------
    # params
    # ---------------------------
    epochs = params.get('epochs', 1)   # user said epoch=1, but keep for compatibility
    minutes = params.get('minutes', 1)
    weights = params.get('weights', [1, 1, 1, 0])

    # GA usage schedule
    ga_start = params.get('ga_start', 0.9)
    ga_end = params.get('ga_end', 0.1)

    # epsilon schedule for random exploration when GA is not used
    eps_start = params.get('eps_start', 0.3)
    eps_end = params.get('eps_end', 0.05)

    gamma = params.get('gamma', 0.1)
    batch_size = params.get('batch_size', 20)
    learning_rate = params.get('learning_rate', 1e-3)
    sync_freq = params.get('sync_freq', 200)   # based on update count
    min_replay_size = params.get('min_replay_size', batch_size * 5)
    
    act_type = params.get('act_type', 0)
    
    org_train_inps = copy.deepcopy(train_inps)
    
    if act_type == 0:
        actions = Actions0
    else:
        actions = Actions1
    
    # ---------------------------
    # models
    # ---------------------------
    policy_model = DqnGRU(
        input_dim=5,
        hidden_dim=hidden_dim,
        output_dim=len(actions),
        num_layers=num_layers
    ).to(device)

    target_model = copy.deepcopy(policy_model).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

    replay = deque(maxlen=MemSize)

    # statistics
    train_losses = []
    train_rewards = []
    guided_train_count = []

    # count actual optimizer updates
    update_count = 0

    for epoch in range(epochs):
        print(f'Epoch # {epoch + 1}/{epochs}')

        train_inps = copy.deepcopy(org_train_inps)
        random.shuffle(train_inps)
        num_samples = len(train_inps)

        for i in range(num_samples):
            print(f'\rTrain {i + 1}/{num_samples}', end='', flush=True)

            # -------------------------------------
            # episode-wise schedule
            # progress: 0.0 ~ 1.0
            # -------------------------------------
            if num_samples > 1:
                progress = i / (num_samples - 1)
            else:
                progress = 1.0

            ga_prob = ga_start - (ga_start - ga_end) * progress
            epsilon = eps_start - (eps_start - eps_end) * progress

            # safety clamp
            ga_prob = max(0.0, min(1.0, ga_prob))
            epsilon = max(0.0, min(1.0, epsilon))

            # Guided optimizer instance for this episode
            guide_optimizer = optimizer_class(**params)
            action_list = [0]

            gym = WaterGym(train_inps[i], minutes, weights, act_type=act_type)
            state_, reward, done, info = gym.reset()

            ga_state = state_

            # build recurrent input state
            input_q = deque([], nstates)
            for _ in range(nstates - 1):
                input_q.append(np.zeros(len(state_), dtype=np.float32))

            state_arr = np.array(state_, dtype=np.float32)
            state_arr = _input_norm(state_arr)  # E3 fix: min-max standardization
            input_q.append(state_arr)

            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)

            losses = []
            total_reward = 0.0
            guided_count = 0
            step_count = 0

            status = 1
            while status:
                step_count += 1

                state_prev = state.detach().cpu().clone()

                with torch.no_grad():
                    q_val, _ = policy_model(state.unsqueeze(0))
                    q_val_np = q_val.detach().cpu().numpy().squeeze(0)

                # -------------------------------------
                # action selection
                # 1) GA with probability ga_prob
                # 2) otherwise epsilon-greedy
                # -------------------------------------
                if random.random() < ga_prob:
                    action = getattr(guide_optimizer, optimizer_method)(
                        ga_state, action_list,
                        next_inflow=gym.outfalls[gym.clock + 1],
                    )
                    guided_count += 1
                else:
                    if random.random() < epsilon:
                        action = np.random.randint(0, len(actions))
                    else:
                        action = int(np.argmax(q_val_np))

                # Keep the optimizer's previous-action history current. The loop
                # originally never updated action_list, unlike run_genetic_algo /
                # run_pso (which append every step) and unlike manuscript
                # Algorithm 3 line 27 ("Update x_t and L_a"), so the optimizer
                # always saw a static [0] previous action.  (E3 fix, 2026-08-28)
                action_list.append(int(action))

                # environment step
                if action not in list(range(len(actions))):
                    print(f'\nLen: {len(actions)}')
                    print(action)
                next_state_, reward, done, info = gym.step(action)
                ga_state = next_state_
                total_reward += reward

                next_state_arr = np.array(next_state_, dtype=np.float32)
                next_state_arr = _input_norm(next_state_arr)  # E3 fix: min-max standardization
                input_q.append(next_state_arr)

                next_state = torch.tensor(
                    np.stack(input_q),
                    dtype=torch.float32,
                    device=device
                )

                exp = (state_prev, action, reward, next_state.detach().cpu().clone(), done)
                replay.append(exp)

                state = next_state

                # -------------------------------------
                # train only after enough replay data
                # -------------------------------------
                if len(replay) >= min_replay_size:
                    minibatch = random.sample(replay, batch_size)

                    state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch]).to(device)
                    action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch],
                                                dtype=torch.long, device=device)
                    reward_batch = torch.tensor([r for (s1, a, r, s2, d) in minibatch],
                                                dtype=torch.float32, device=device)
                    state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch]).to(device)
                    done_batch = torch.tensor([d for (s1, a, r, s2, d) in minibatch],
                                              dtype=torch.float32, device=device)

                    # Q(s, a)
                    Q1, _ = policy_model(state1_batch)
                    current_q = Q1.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        # DDQN
                        Q_online_next, _ = policy_model(state2_batch)
                        next_actions = torch.argmax(Q_online_next, dim=1, keepdim=True)

                        Q_target_next, _ = target_model(state2_batch)
                        next_q = Q_target_next.gather(1, next_actions).squeeze(1)

                        target_q = reward_batch + gamma * (1.0 - done_batch) * next_q

                    loss = loss_fn(current_q, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

                    # sync by update count
                    update_count += 1
                    if update_count % sync_freq == 0:
                        target_model.load_state_dict(policy_model.state_dict())

                if done:
                    status = 0

            train_losses.append(np.mean(losses) if len(losses) > 0 else np.nan)
            train_rewards.append(total_reward)
            guided_train_count.append(guided_count)

            print(
                f" Loss: {train_losses[-1]:.6f},"
                f" Reward: {train_rewards[-1]:.4f},"
                f" {guide_label}_prob: {ga_prob:.3f},"
                f" Eps: {epsilon:.3f},"
                f" {guide_label}_count: {guided_count},"
                f" Steps: {step_count}"
            )

    print()

    train_losses = np.array(train_losses)
    train_rewards = np.array(train_rewards)

    if model_path is None:
        model_path = '../trained_models/ddqn_gru_ga_guided_default.pkl'

    torch.save(policy_model.state_dict(), model_path)

    return policy_model, train_losses, train_rewards


def train_guided_byGA(train_inps, model_path: str = None, **params):
    """Train DDQN-GRU with GA-selected guided replay transitions."""
    return _train_guided_by_optimizer(
        train_inps,
        model_path,
        optimizer_class=GeneticAlgorithm,
        optimizer_method='genetic_algorithm',
        guide_label='GA',
        **params,
    )


def train_guided_byPSO(train_inps, model_path: str = None, **params):
    """Train DDQN-GRU with PSO-selected guided replay transitions."""
    return _train_guided_by_optimizer(
        train_inps,
        model_path,
        optimizer_class=ParticleSwarmOptimization,
        optimizer_method='particle_swarm_optimization',
        guide_label='PSO',
        **params,
    )


def train_guided_byENUM(train_inps, model_path: str = None, **params):
    """Train DDQN-GRU with exhaustive-enumeration-selected guided transitions.

    E4-a: same generic hook as GA/PSO, only the guide optimizer changes.
    EnumSearch.genetic_algorithm() returns the exact 1-step argmax over the 6
    feasible actions (enum_baseline.py). Additive -- the GA/PSO paths above are
    untouched.
    """
    return _train_guided_by_optimizer(
        train_inps,
        model_path,
        optimizer_class=EnumSearch,
        optimizer_method='genetic_algorithm',
        guide_label='ENUM',
        **params,
    )

 
def train(train_inps, model_path:str=None, **params):
    """
    DQN + GRU with demonstration from genetic algorithm

    Parameters
    ----------
    train_inps: TYPE
        Training data.
    model_path : str, optional
        the path of model for saving       

    Returns
    -------
    policy_model : torch.nn.Module
        trained policy network
    train_epoch_losses : np.ndarray
        train average TD-loss per epoch
    train_epoch_rewards : np.ndarray
        train total episodes average reward per epoch
    """

    # params 
    epochs  = params.get('epochs', 1)
    minutes = params.get('minutes', 1)
    weights = params.get('weights', [1., 1., 1., 0.])
    act_type = params.get('act_type', 0)
    if act_type == 0:
        actions = Actions0
    else:
        actions = Actions1
        
    org_train_inps = copy.deepcopy(train_inps)    
    policy_model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                          output_dim=len(actions), 
                          num_layers=num_layers).to(device)
    
    target_model = copy.deepcopy(policy_model).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    
    gamma = params.get('gamma', 0.1)   # E3 fix: was hardcoded 0.1 (near-myopic); pass 0.7
    epsilon = 1.0
    
    batch_size = 20
    replay = deque(maxlen=MemSize)
    sync_freq = 10 #A
    
    for epoch in range(epochs):
        print(f'Epoch # {epoch+1}/{epochs}')
        train_inps = org_train_inps
        random.shuffle(train_inps)
        num_samples = len(train_inps)
        
        train_losses = [] # for only one epoch, 각 에피소드마다 로스 저장, 만약 epoch 별로 계산하려면 수정 필요
        train_rewards = [] # for only one epoch, 각 에피소드마다 보상 저장   
           
        for i in range(num_samples): 
            print(f'\rTrain {i+1}/{num_samples}', end='', flush=True)
            gym = WaterGym(train_inps[i], minutes, weights, act_type)
            state_, reward, done, info = gym.reset()
            
            input_q = deque([], nstates) # for build input vector
            # build initial input 
            for k in range(nstates-1): 
                input_q.append(np.array([0.0 for _ in range(len(state_))]))
            state_ = _input_norm(state_) # input normalization (E3 fix: min-max standardization)
            input_q.append(state_)
            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            
            j = 0
            status = 1
            total_reward = 0.0      # <- total reward for this episode 
            losses = [] # i번째 에피소드에서 훈련 업데이트마다 loss 저장, 나중에 평균을 계산해서 해당 에피소드의 loss로 사용
            while status: 
                j += 1
                state_prev = torch.Tensor.cpu(state)
                q_val, hidden_state = policy_model(state.unsqueeze(0))
                q_val_ = torch.Tensor.cpu(q_val).data.numpy()
                
                # Epsilon-greedy policy 
                if (random.random() < epsilon):
                    action = np.random.randint(0,len(actions))
                else:
                    action = np.argmax(q_val_)
                
                state_, reward, done, info = gym.step(action)
                total_reward += reward   # <- reward accumulation

                state_ = _input_norm(state_) # input normalization (E3 fix: min-max standardization)
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
                    
                    # 수정된 DDQN
                    Q1_gru, hidden_state = policy_model(state1_batch)  # Q(s, ��)  �� ����(X)��

                    with torch.no_grad():
                        # (1) 다음 상태에서 action 선택은 online(policy)로
                        Q1_next, _ = policy_model(state2_batch)  # Q(s', .) (online)
                        acts_ = torch.argmax(Q1_next, dim=1, keepdim=True)  # a* = argmax_a Q_online(s',a)
                    
                        # (2) 다음 상태에서 그 action의 가치는 target으로 평가
                        Q2_gru, _ = target_model(state2_batch)   # Q(s', .) (target)
                        tqvals = Q2_gru.gather(dim=1, index=acts_).squeeze(1)
                    
                    Y = reward_batch + gamma * (1 - done_batch) * tqvals
                    X = Q1_gru.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze(1)
                    
                    loss = loss_fn(X, Y)  # Y는 no_grad라 detach 불필요(해도 무방)
                 
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    
                if j % sync_freq == 0:
                    target_model.load_state_dict(policy_model.state_dict())
                if done:
                    status = 0
    
            
            # epsilon decay per episode
            if epsilon > 0.2:
                epsilon -= (1./num_samples)

            # ---- train loss & reward ----
            train_losses.append(np.asarray(losses).mean() if len(losses) > 0 else np.nan)
            train_rewards.append(total_reward)
            print(f" Loss: {train_losses[-1]}, Reward: {train_rewards[-1]}")
        
       
    print() # for output line adjustment                  
    
    train_losses = np.array(train_losses)
    train_rewards = np.array(train_rewards)
    
    print(model_path)
    if model_path is None:
        model_path = '../trained_models/ddqn_gru_basic_default.pkl'
    torch.save(policy_model.state_dict(), model_path)
    
    return policy_model, train_losses, train_rewards



def plot_losses(basic_loss, ga_loss):
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 12)
    ax.plot(basic_loss, label='Regular DDQN')
    ax.plot(ga_loss, label='GA-guided DDQN')
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Loss")
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('../results/images/loss_two_goal_comparison.png', dpi=600)
    plt.show()



def plot_rewards(basic_reward, ga_reward):
    fig, ax = plt.subplots(2, 1, sharex=True) 
    fig.set_size_inches(30, 12)

    ax[0].plot(basic_reward, color='blue', label='Regular DDQN')
    ax[0].set_ylabel("Reward")
    ax[0].legend(loc='best')

    ax[1].plot(ga_reward, color='red', label='GA-guided DDQN')
    ax[1].set_xlabel("Scenario")  
    ax[1].set_ylabel("Reward")
    ax[1].legend(loc='best')

    plt.tight_layout()
    plt.savefig('../results/images/reward_two_goal_comparison.png', dpi=600)
    plt.show()


def test_model(model, test_inp, minutes:int=1, model_path:str=None, 
               weights:list=[1.,1.,1.,1.], act_type:int=0):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if act_type == 0:
        actions = Actions0
    else:
        actions = Actions1
        
    if model == None:
        if model_path == None:
            path = '../trained_models/dqn_gru_pumpmodel_w2_w1.pkl'
        else:
            path = model_path
        model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                       output_dim=len(actions), num_layers=num_layers).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
    
    gym = WaterGym(test_inp, minutes, weights, act_type=act_type)
    init_state_, reward, done, info = gym.reset()
    input_q = deque([], nstates) # for build input vector
    for k in range(nstates-1): 
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = _input_norm(init_state_) # input normalization (E3 fix: min-max standardization)
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
        
        state_ = _input_norm(state_) # input normalization (E3 fix: min-max standardization)
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
    
    # years = ['10', '20'] #, '30', '50', '80', '100']
    # durations = ['0060', '0120'] #, '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    # exper_params = {'minutes':2, 'epochs':1, 'weights': [1.0, 1.0, 1.0, 0.0]}
    # inp, train_inps, valid_inps, test_inps = stratified_split_data(years, durations, trate=0.7, vrate=0.0, stratify=True)
    
    # # DDQN Basic
    # model_path = '../trained_models/dqn_gru_regular_w1_0_w1_5_w0_5_w0_min2_tr0_05.pkl'
    # start_time = time.time()
    # basic_model, basic_losses, basic_rewards = train(train_inps, model_path=model_path, **exper_params)
    # end_time = time.time()
    # elapsed_basic = end_time - start_time
    
    # print(f"Completed training: {model_path.split('/')[-1]}")
     
    # test_list = ['data/gasan/50year/50yr_0120m_h371.inp']
    # tmodel, action_list, state_list, reward_list, info_list = test_model(None, test_list[0], minutes=minutes, model_path=model_path)
    
    # plt.figure()
    # plot_individual_case(state_list, action_list)
    # plt.figure()
    # fig_title =  test_list[0].split('/')[-1]
    # plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])
    
    
    # DDQN GA-guided
    model_path = '../trained_models/dqn_gru_regular_w0_5_w0_5_w0_0_w0_0_min2_tr0_03_acttype0.pkl'
    # start_time = time.time()
    # ga_model, ga_losses, ga_rewards = train_guided_byGA(train_inps, model_path=model_path, **exper_params)
    # end_time = time.time()
    # elapsed_gaguided = end_time - start_time
    # print(f"Completed training: {model_path.split('/')[-1]}")
    # plot_losses(basic_losses, ga_losses)
    # plot_rewards(basic_rewards, ga_rewards)
     
    test_list = ['data/gasan/100year/100yr_0010m_h000.inp']
    tmodel, action_list, state_list, reward_list, info_list = test_model(None, test_list[0], minutes=2, model_path=model_path)
    
    # plt.figure()
    # plot_individual_case(state_list, action_list)
    # plt.figure()
    # fig_title =  test_list[0].split('/')[-1]
    # plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])
   
