#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:06:13 2024

@author: drminor

cost_model.py

- 여기에 구현된 함수들은 펌프별로 전력소비량과 전기요금 계산을 위한 것임
- 전체 시나리오가 끝나면 선택했던 전체 행위 리스트을 대상으로 전력소비량과 전기요금 계산
- 강화학습에서 최종적으로 보상을 계산하기 위함

"""

import re
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from dataclasses import dataclass

from water_gym import Actions0, pumpq
import water_gym
import man_policy
import dqn_gru_torch
import utils
#from cost_model_plot import CostModelPlot 

Industrial_Rate = 0.1 #$ per kWh
RDOWN = 0.0 # 펌프 다운시 발생 피해액 $
pump_type = {'p100': (True, False, False, False, False), # 펌프타입을 특정 action(리스트)으로 표현
           'p170': (False, False, False, True, False)}


## dry run cost parameter
@dataclass
class DryFailure:
    name: str
    t_dry: float       # 수위 저하 등으로 건식 상태가 시작된 순간부터 보호-정지 명령이 떨어질 때까지의 시간
    tau_s: float       # scale for time-to-fail hazard model (seconds)
    gamma: float       # shape
    cost_repair: float # $ per failure
    r_down: float      # $ per down time(hour)
    mttr_h: float      # hours of downtime if this failure occurs

# Immediate dry-run risk models
# p100 DryFailure
seal_100   = DryFailure(name="Seal-100",  t_dry=12.0, tau_s=30.0,  
                        gamma=2.5, cost_repair=3000.0, r_down=RDOWN, mttr_h=6.0)
motor_100  = DryFailure(name="Motor-100", t_dry=12.0, tau_s=120.0, 
                        gamma=3.0, cost_repair=15000.0, r_down=RDOWN, mttr_h=24.0)

# p170 DryFailure
seal_170   = DryFailure(name="Seal-170", t_dry=10.0, tau_s=20.0,  
                        gamma=3.0, cost_repair=3500.0, r_down=RDOWN, mttr_h=8.0)
motor_170  = DryFailure(name="Motor-170", t_dry=10.0, tau_s=90.0,  
                        gamma=3.5, cost_repair=18000.0, r_down=RDOWN, mttr_h=24.0)

class Pump:
    # the types of costs, additional cost is available
    cost_terms = ("consume", "start_extra", "wear", "repair", "down", "dryrun") # repair cost == failure cost
    
    def __init__(self, idx:int, ptype:str, minutes:int=2, init_cycles:int=0): 
        """
        
        Parameters
        ----------
        idx : int
            DESCRIPTION.
        ptype : str
            pump type - p100, p170.
        init_cycles : int, optional
            펌프의 노후도를 고려하기 위한 변수, 초기 사이클 수(사용상태). 
            The default is 0.

        Returns
        -------
        None.

        """
        self.idx:int = idx
        self.steps = 0 # the length of actions
        self.ptype:str = ptype
        self.minutes:int = minutes
        self.init_cycles:int = init_cycles
        
        self.on_off_states:list= []          # the on/off state of the pump (bool)
        self.dryrun_state:list=[]            # dry run state for each step of the pump (bool)
        self.ncycles_cum:list= []            # cumulative number of cycles
        self.ndryruns_cum:list= []            # cumulative number of dry runs
        self.total_ncycles:int= 0  # the total number of cycles
        self.total_ndryruns:int= 0            # the total number of dry runs
        
        self.energy_consumption_cum = {} # cumulative form: 에너지량이 불가한 경우는 empty
        self.energy_cost_cum = {}        # cumulative form 
        
        self.total_energy_consumption_per_term = {} # cost_term 별 전력소비량
        self.total_energy_cost_per_term = {}        # cost_term 별 전력비용
        
        self.total_energy_consumption = 0.0   
        self.total_energy_cost = 0.0
        
        for term in self.cost_terms:
            self.energy_consumption_cum[term] = []
            self.energy_cost_cum[term] = []
            self.total_energy_consumption_per_term[term] = []
            self.total_energy_cost_per_term[term] = []


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


def calculate_cycles_pump(pump:Pump):
    """
    주어진 on/off state 리스트로부터 사이클 수, 누적 사이클 수 계산

    Parameters
    ----------
    onoffs : list
        DESCRIPTION.

    Returns
    -------
    
    """
    
    ncycles = 0
    length = len(pump.on_off_states)
    pump.ncycles_cum.append(0)
    
    if length <= 1:
        print("onoff state list is empty or one element")
        return -1
    for i in range(1, length):
        if pump.on_off_states[i-1] and not pump.on_off_states[i]:
            pump.total_ncycles += 1
        pump.ncycles_cum.append(pump.total_ncycles)
    if pump.on_off_states[-1]: # 마지막이 true인 경우 사이클 수 1 추가
        pump.total_ncycles += 1
        pump.ncycles_cum[-1] = pump.total_ncycles
    
    # addition initial cycle number
    for i in range(len(pump.ncycles_cum)):
        pump.ncycles_cum[i] += pump.init_cycles
    pump.total_ncycles += pump.init_cycles
                

def count_dryrun_pump(pump:Pump, infos:list):
    """
    infos로 부터 전체 펌프 시스템의 step별 dry run 유무를 추출하고, 
    현재 pump의 가동 상태와 연계한 run_and_on 리스트 작성
    dry run에 의해 발생한 비용 계산을 위해 필요한 함수

    Parameters
    ----------
    pump : Pump
        DESCRIPTION.
    infos : list
        infors[-1]은 해당 스텝에서의 펌프 시스템 전체의 run over 발생 여부

    Returns
    -------
    dry_and_on : list
        dry run이 발생하고 현 펌프가 on 상태이면 true이고 나머지 상태엔 false.

    """
    
    assert len(pump.on_off_states) > 0, "The pump on/off states must be set first."
    
    steps = len(infos) # the number of action steps
    dry_run = [] # whether over run occurred in the level of the system
    
    for info in infos:
        dry_run.append(info[-1])    
    
    dry_and_on_arr = np.array(pump.on_off_states) * np.array(dry_run)
    pump.dryrun_state = dry_and_on_arr.tolist()
    pump.ndryruns_cum = np.cumsum(dry_and_on_arr).tolist()
    pump.total_ndryruns = pump.ndryruns_cum[-1]
    

def start_extra_cost_onoff(pump:Pump, alpha:float=1.5, t_start:float=5.0, minutes:int=1):
    """
    기동 추가 에너지 소비량 계산 및 비용 계산

    Parameters
    ----------
    actions : list
        DESCRIPTION.
    alpha : float, optional
        기동시 순간 전력 계산에 사용된는 multiplier.
        P_start = alpha * P_run(= P_e_kW)
        The default is 1.5.
    t_start : float, optional
        기동에 걸리는 시간. The default is 5.0.(sec)
    minutes : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------

    """
    
    # 사이클 수에 따른 기동 전력과 전기료 계산
    # P_e_kW(p100): 226.55470568673095, (p170): 385.14299966744255
    _, P_e_kW, _ = action_unit_power_cost_list(pump_type[pump.ptype], minutes=minutes) # 여기서 minutes은 무의미
    
    start_extra_eng_kWh = (alpha - 1) * P_e_kW * (t_start/3600)  # 5초 동안의 기동 전력(kWh)
    for cycles in pump.ncycles_cum:
        cy_se_eng = cycles * start_extra_eng_kWh # 사이클 기동 전력
        pump.energy_consumption_cum['start_extra'].append(cy_se_eng)
        pump.energy_cost_cum['start_extra'].append(cy_se_eng * Industrial_Rate)
    pump.total_energy_consumption_per_term['start_extra'] = pump.total_ncycles * start_extra_eng_kWh 
    pump.total_energy_cost_per_term['start_extra'] = pump.total_energy_consumption_per_term['start_extra'] * Industrial_Rate
    pump.total_energy_consumption += pump.total_energy_consumption_per_term['start_extra']
    pump.total_energy_cost += pump.total_energy_cost_per_term['start_extra']
        

def wear_cost_onoff(pump:Pump):
    """
    1) 개별 펌프 운용에 수반되는 예측 가능한 "자잘한" 반복비(접점 체크밸브 시트 마모, 경미한 소모품, 점검/정비의 
       주기별 균등분할분 등)의 달러 표현 
    2) wear cost 계산에는 에너지 소비가 불필요

    Parameters
    ----------
    pump : Pump
        DESCRIPTION.
    Returns
    -------
    total_wear_cost : float
        시나리오에 대한 total wear cost.
        
    cum_wear_cost : list
        action sequence에 대한 시간 별 누적 wear cost

    """
    wearcost_per_cycle = {'p100': 0.05, 'p170': 0.07} # 사이클 1회당(한 번 켜서 끄는 주기 1회) 발생한다고 
                                   # 간주하는 소모, 마모 비용의 달러 값
                                   # pump100cmm: 0.05$, pump170cmm: 0.07$
    for cycles in pump.ncycles_cum:
        pump.energy_cost_cum['wear'].append(wearcost_per_cycle[pump.ptype] * cycles)
    pump.total_energy_cost_per_term['wear'] = wearcost_per_cycle[pump.ptype] * pump.total_ncycles 
    pump.total_energy_cost += pump.total_energy_cost_per_term['wear']
   

def repair_cost_onoff(pump:Pump, k:float=1.3, prob_model:str="nhpp"):
    """
    1) 하나의 시나리오에서 행해진 펌프의 on-off 스위치 사이클 수에 따른 failure cost(수리비) 계산
    
    Parameters
   수 ----------
    actions : list
        특정 inp 파일(시나리오)에 대해 선택된 펌프조합 action 리스트.
    k : int
        노후화 정도
    prob_model : str, optional
        failure cost 계산에 사용되는 확률분포.
        "linear": linear distribution, "weibull": weibull distribution, "nhpp": Crow-AMSAA
        The default is "weibull".

    Returns
    -------
    exp_fail : float
        전체 사이클 수에 대한 총 기대 고장 수
    exp_fail_cum :list
        누적 사이클 수에 대한 누적 기대 고장 수
    """
    
    # 펌프 모델 파라미터  
    pump_param = {
        "p100": {
            "MTBF": 30000, # 제조사 권장 Mean Time Between Failures(= Mean Cycle Between Failures) : 고장 사이의 평균 사이클 수
            "C_rep": 3000, # 1회 펌프 고장 수리_교체 비용(USD), 즉 펌프 가격
            "weibull": {"lambda": 30000, "k": k} # lambda == MTBF, k: 형상 파라미터(노후화 정도) k > 1 -> 노후화, 마모에 의한 고장 상태
        },
        "p170": {
            "MTBF": 20000,
            "C_rep": 4000,
            "weibull": {"lambda": 20000, "k": k}
        }
    }
    
    # 선형 고장 모델
    def linear_prob(N, p): # N: the number of cycle
        return (N / p["MTBF"]) 
    
    # Weibull 고장 모델
    def weibull_cdf(N, p):
        lam = p["weibull"]["lambda"] 
        k = p["weibull"]["k"]
        Pf = 1 - np.exp(-(N / lam)**k)
        return Pf
    
    # Crow-AMSAA/NHPP 고장 모델
    def crow_amsaa_prob(N, p):
        lam = p["weibull"]["lambda"] 
        k = p["weibull"]["k"]
        m_N = (N / lam) ** k
        return m_N
    
    # for reducing length of variable names
    init_cycles = pump.init_cycles
    t_cycles = pump.total_ncycles
    ncycles_cum = pump.ncycles_cum
    ptype = pump.ptype
    c_rep = pump_param[ptype]["C_rep"]  # the price of pump of ptype
    
    # variables for down cost
    exp_fail = 0.0
    exp_fail_cum = []
    
    if prob_model == "linear":
        prob_f = linear_prob
        # Cumalitive repair cost
        for n in ncycles_cum:
            e = prob_f(n, pump_param[ptype])
            exp_fail_cum.append(e)
            pump.energy_cost_cum['repair'].append(e * c_rep)
        # Total repari cost
        exp_fail = prob_f(t_cycles, pump_param[ptype])
        pump.total_energy_cost_per_term['repair'] = exp_fail * c_rep
    
    elif prob_model == "weibull":
        prob_f = weibull_cdf
        ### 
        ### !! 초기 펌프 사용 기대 횟수를 빼주는게 맞는지 추가 확인 필요 
        ###
        p_init = prob_f(init_cycles, pump_param[ptype]) # failure 기대 횟수 계산을 위해 initial cycles를 이용 
        
        # Cumalitive repair cost
        for n in ncycles_cum:
            e = max(prob_f(n, pump_param[ptype]) - p_init, 0.0)
            exp_fail_cum.append(e)
            pump.energy_cost_cum['repair'].append(e * c_rep)
        
        # total repair cost
        exp_fail = max(prob_f(t_cycles, pump_param[ptype]) - p_init, 0.0)
        pump.total_energy_cost_per_term['repair'] = exp_fail * c_rep
            
    elif prob_model == "nhpp":
        prob_f = crow_amsaa_prob
        p_init = prob_f(init_cycles, pump_param[ptype]) # failure 기대 횟수 계산을 위해 initial cycles를 이용 
        
        # Cumalitive 비용 계산
        for n in ncycles_cum:
            e = max(prob_f(n, pump_param[ptype]) - p_init, 0.0)
            exp_fail_cum.append(e)
            pump.energy_cost_cum['repair'].append(e * c_rep)
        
        # total repair
        exp_fail = max(prob_f(t_cycles, pump_param[ptype]) - p_init, 0.0)
        pump.total_energy_cost_per_term['repair'] = exp_fail * c_rep
        
    # total cost update
    pump.total_energy_cost += pump.total_energy_cost_per_term['repair']
        
    return exp_fail, exp_fail_cum


def downtime_cost_onoff(pump:Pump, exp_fail:float, exp_fail_cum:list):
    """
    펌프 불가용 시간(수리시간)에 의해 추가로 발생하는 다운타임 비용(optional)
    down_cost = r_down($/h: 시간당 손실) * MTTR(h: 평균 수리시간) * exp_failures(주어진 사이클에 대한 기대 고장수)
    MTTR = 6 # Mean Time To Repair h
    r_down = 100000 # 펌프 고장으로 가용하지 않을 때 1시간당 발생하는 경제적 손실 $/h

    Returns
    -------
    None.

    """
    
    MTTR = 6 # Mean Time To Repair h
    r_down = RDOWN # 펌프 고장으로 가용하지 않을 때 1시간당 발생하는 경제적 손실 $/h
    
    pump.total_energy_cost_per_term['down'] = exp_fail * MTTR * r_down
    for e in exp_fail_cum:
        pump.energy_cost_cum['down'].append(e * MTTR * r_down)
    
    # total cost update
    pump.total_energy_cost += pump.total_energy_cost_per_term['down']    


def energy_consume_cost(pump:Pump, minutes:int=1):
    """
    주어진 펌프를 대상으로 시나리오에 따른 전력 소비량과 전력 소비 비용 계산

    Parameters
    ----------
    pump : Pump
        특정 펌프 
    minutes : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    consume_kWh : float
        전력 소비량
    consume_kWh_cum : list
        누적 전력 소비량
    consume_cost : float
        전기료
    consume_cost_cum : list
        누적 전기료
    
    """
    
    consume_kWh = 0.0
    consume_cost = 0.0
    
    # 주어진 펌프의 2 minutes 동안(한 시간 유닛)의 전력량과 비용 계산
    # p100: 7.5518235228910315kWh, 0.7551823522891032$
    # p170: 12.838099988914752kWh, 1.2838099988914753$
    kwh, _, cost = action_unit_power_cost_list(pump_type[pump.ptype], minutes)
    # 누적 계산
    for state in pump.on_off_states:
        if state:
            consume_kWh += kwh
            consume_cost += cost
        pump.energy_consumption_cum['consume'].append(consume_kWh)
        pump.energy_cost_cum['consume'].append(consume_cost)
    # Total 계산
    zero_one = [1 if s else 0 for s in pump.on_off_states]
    num_ons = sum(zero_one)
    pump.total_energy_consumption_per_term['consume'] = num_ons * kwh
    pump.total_energy_cost_per_term['consume'] = num_ons * cost
    pump.total_energy_consumption += pump.total_energy_consumption_per_term['consume']
    pump.total_energy_cost += pump.total_energy_cost_per_term['consume']
    

def action_unit_power_cost(action:int, minutes:int=1):
    """
    
    샤프트 출력 P_s (kW) 계산 후 minutes 단위 동안의 전략소비량 및 전기 요금 계산
    즉, 선택된 펌프 운용의 minutes 분 동안 소비된 전략 소비량 및 전기 요금용
    (energy_consume_cost()에서 사용)
    
    q1, q2 -> Q_m3_per_min : float
        분당 유량(Q_m3_per_min, m^3/min).
    H -> H_m : float
        양정(m: 펌프가 물을 끌어올리는 높이).
    eta_pump : float
        펌프 효율(0~1)
    eta_motor : float
        모터 효율 (IE3-IE4 기준, 약 90-95%)
    eta_drive : float
        구동 드라이브(인버터) 효율 (약 95-98%)
    rho : int, optional
        유체 밀도 -> 물 기준 1000kg/m^3
    g : float, optional
        중력 가속도. 9.81.
    industrial_rate : float
        = 0.1 # 산업용 전기 kWh 당 about 120원 (= 0.1 달러)
        
    Parameters
    ----------
    action : int 
        펌프 조합 - Actions의 인덱스
    minutes : int
        minutes 단위로 운영되는 펌프 조합
    
    Returns
    -------
    E_kwh : float
    E_J : float (J 단위 - 1kWh = 3.6 MJ = 3,600,000J)
        minutes 동안 펌프 조합의 에너지 소비량.
    E_Cost : float
        minutes 동안 펌프 조합의 전기 요금
    """
    
    H = 10.0                # m
    eta_pump = 0.80         # 80% 
    eta_motor = 0.93
    eta_drive = 0.97
    rho = 1000
    g = 9.81 
    industrial_rate = 0.1 # 산업용 전기 kWh 당 about 120원 (= 0.1 달러)
    
    P_h_kW = 0.0
    for i, on_off in enumerate(Actions[action]):
        q_m3_per_h = pumpq[i] * on_off * 60.0 #1) 유량을 m^3/h 단위로 변환
        P_h_kW += rho * g * q_m3_per_h * H / 3.6e6 #2) 수력 출력 P_h (kW)
    
    #3) 샤프트 출력 P_s = P_h / eta
    P_s_kW = P_h_kW / eta_pump
    
    #4) 전기 입력 : 전기 비용 계산을 위해 
    P_e_kW = P_s_kW / (eta_motor * eta_drive)
    
    #5) minutes 동안의 전력소모량을 시간당 전력량(kWh)으로 계산 
    E_kWh = P_e_kW * minutes/60.0
    E_J = E_kWh * 3600000 # J로 계산
    
    #6) 전기 비용
    E_Cost = E_kWh * Industrial_Rate
    
    
    return E_kWh, P_e_kW, E_Cost


def action_unit_power_cost_list(action:list, minutes:int=1):
    """
    
    action을 리스트(1 -> [True, False, False, False, False])로 받는 걸 제외하면 action_unit_power_cost 함수와 동일함
    펌프 유형별 power, cost를 계산하기 위해 사용
    pump 100 type -> [True, False, False, False, False]
    pump 170 type -> [False, False, False, True, False]
        
    Parameters
    ----------
    action : list
        펌프 조합 -> [True, True, True, False, False]
    minutes : int
        minutes 단위로 운영되는 펌프 조합
    
    Returns
    -------
    E_kwh : float
    E_J : float (J 단위 - 1kWh = 3.6 MJ = 3,600,000J)
        minutes 동안 펌프 조합의 에너지 소비량.
    E_Cost : float
        minutes 동안 펌프 조합의 전기 요금
    """
    
    H = 10.0                # m
    eta_pump = 0.80         # 80% 
    eta_motor = 0.93
    eta_drive = 0.97
    rho = 1000
    g = 9.81 
    industrial_rate = 0.1 # 산업용 전기 kWh 당 about 120원 (= 0.1 달러)
    
    P_h_kW = 0.0
    for i, on_off in enumerate(action):
        q_m3_per_h = pumpq[i] * on_off * 60.0 #1) 유량을 m^3/h 단위로 변환
        P_h_kW += rho * g * q_m3_per_h * H / 3.6e6 #2) 수력 출력 P_h (kW)
    
    #3) 샤프트 출력 P_s = P_h / eta
    P_s_kW = P_h_kW / eta_pump
    
    #4) 전기 입력 : 전기 비용 계산을 위해 
    P_e_kW = P_s_kW / (eta_motor * eta_drive)
    
    #5) 전력량 계산 
    E_kWh = P_e_kW * minutes/60.0
    E_J = E_kWh * 3600000 # J로 계산
    
    #6) 전기 비용
    E_Cost = E_kWh * Industrial_Rate
    
    
    return E_kWh, P_e_kW, E_Cost


def prob_fail_dry(t_s, tau_s, gamma):
    # 1 - exp(-(t/tau)^gamma)
    return 1.0 - math.exp(- (max(t_s,0.0) / tau_s) ** gamma )


def dry_run_cost(pump:Pump, p_model:str="nhpp"):
    """
    펌프의 dry run으로 인해 발생한 비용 계산
    seal 실패와 motor 번아웃 발생으로 인한 고장 수리비용과 down time에 발생한 즉시 고장 비용만 계산함
    dry run 발생 보호 및 탐지는 없는 것으로 가정함     
    dry run 무효 에너지나 장기 기대고장 비용 등은 고려하지 않음. (추후 개선 필요)

    Parameters
    ----------
    pump : Pump
        DESCRIPTION.
    minutes : TYPE
        DESCRIPTION.
    p_model : str, optional
        DESCRIPTION. The default is "nhpp".

    Returns
    -------
    None.

    """
    seal, motor = (seal_100, motor_100) if pump.ptype == 'p100' else (seal_170, motor_170)
    dryrun_cost_cum = 0.0
    for state in pump.dryrun_state:
        dryrun_cost = 0.0
        if state:
            seal_prob = prob_fail_dry(seal.t_dry, seal.tau_s, seal.gamma)
            seal_imm_cost = seal_prob * (seal.cost_repair + seal.r_down*seal.mttr_h)
            motor_prob = prob_fail_dry(motor.t_dry, motor.tau_s, motor.gamma)
            motor_imm_cost = motor_prob * (motor.cost_repair + motor.r_down*motor.mttr_h)
            dryrun_cost = seal_imm_cost + motor_imm_cost
        dryrun_cost_cum += dryrun_cost
        pump.energy_cost_cum['dryrun'].append(dryrun_cost_cum)
    pump.total_energy_cost_per_term['dryrun'] = dryrun_cost_cum
    pump.total_energy_cost += dryrun_cost_cum
    
    
def execute_scenario(actions:list, infos:list, minutes:int=2, init_cycles:int=0):
    """
    주어진 시나리오(action list)를 대상으로 각 펌프의 누적/총 전력소비량 및 비용 계산 
    개별 펌프(5개)의 전력 소비량과 비용 사용하고자 할 때
    Parameters
    ----------
    actions : list
        
    minutes: int
        시뮬레이션 전개 시간 단위. The default is 2
    
    Returns
    -------
    pumps: Pump list
    개별 펌프들의 전력 소비량 및 비용들

    """
    
    #########################################################
    ####### Pump 생성 및 on/off 상태 초기화, 
    #######  on/off 사이클 수 계산, dry run 수 계산
    #########################################################
    # pumpq를 이용해 5개 펌프의 인스턴스 리스트 생성
    pumps = [Pump(i, 'p'+str(tp), minutes, init_cycles) for i, tp in enumerate(pumpq)]
    
    # set the steps
    for pump in pumps:
        pump.steps = len(actions)
    
    # on/off state setting from actions
    for action in actions: 
        onoff = Actions[action] # ex) [True, True, False, False, False]
        for i, pump in enumerate(pumps):
            pump.on_off_states.append(onoff[i])
    
    # calculate the number of cycles and cumulative cycles from on_off_states
    for pump in pumps:
        calculate_cycles_pump(pump)
        count_dryrun_pump(pump, infos)
    
    
    #########################################################
    ####### Enery Consumption and Cost Calculations #########
    #########################################################
    
    # Compute start extra cost from the number of cycles including cumalative version
    for pump in pumps:
        start_extra_cost_onoff(pump, minutes=minutes)
        
    # Compute energy consumption and cost associated with pump operation
    for pump in pumps:
        energy_consume_cost(pump, minutes=minutes)
        
    # Compute wear cost for each pump (no energy consumption, only energy cost)
    for pump in pumps:
        wear_cost_onoff(pump)
        
    # Compute repair and downtime cost (no energy consumption, only energy cost)
    for pump in pumps:
        exp_fail, exp_fail_cum = repair_cost_onoff(pump)
        downtime_cost_onoff(pump, exp_fail, exp_fail_cum)
        
    # Compute dry run cost for each pump
    for pump in pumps:
        dry_run_cost(pump)
    
    return pumps


def analysis_result(pumps:list):
    """
    execute_scenario()를 통해 얻은 펌프들의 전력량과 비용을 이용해 펌프 유형별 및 
    전체 전력량과 비용 종합 및 분석
    

    Parameters
    ----------
    pumps : list
        execute_scenario()를 통해 얻은 펌프들의 전력량과 비용.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """
    steps = pumps[0].steps 
    results = {
        "p100": 
            {
             # Energy Consumption & Cost 
             "energy_consume": [0. for _ in range(steps)], # cumulative enery consumption
             "cost_consume": [0. for _ in range(steps)],   # cumulative cost consumption
             
             # On/off derived cost
             "energy_start_extra": [0. for _ in range(steps)],
             "cost_start_extra": [0. for _ in range(steps)],
             "cost_wear": [0. for _ in range(steps)],
             "cost_repair": [0. for _ in range(steps)],
             "cost_down": [0. for _ in range(steps)],
             "cost_onoff": [0. for _ in range(steps)],
             
             # Dryrun cost
             "cost_dryrun": [0. for _ in range(steps)],

             "cost_maintenance": [0. for _ in range(steps)], # cost_onoff + cost_dryrun
             "cost_total": [0. for _ in range(steps)], # cost_consume + cost_maintenance
             
             "ncycles": [],       # for each pump, 3 pumps in this case
             "ndryruns": [],
             "total_cycles": 0,
             "total_energy": 0.0,
             "total_cost": 0.0
             },
        "p170":
             {
              "energy_consume": [0. for _ in range(steps)], # cumulative enery consumption
              "cost_consume": [0. for _ in range(steps)],   # cumulative cost consumption

              "energy_start_extra": [0. for _ in range(steps)],              
              "cost_start_extra": [0. for _ in range(steps)],
              "cost_wear": [0. for _ in range(steps)],
              "cost_repair": [0. for _ in range(steps)],
              "cost_down": [0. for _ in range(steps)],
              "cost_onoff": [0. for _ in range(steps)],
              
              "cost_dryrun": [0. for _ in range(steps)],
              
              "cost_maintenance": [0. for _ in range(steps)],
              "cost_total": [0. for _ in range(steps)], # cost_consume + cost_maintenance
              
              "ncycles": [],       # for each pump, 2 pumps in this case
              "ndryruns": [],
              "total_cycles": 0,
              "total_energy": 0.0,
              "total_cost": 0.0
              },
        "total":
             {
              "energy_consume": [0. for _ in range(steps)], # cumulative enery consumption
              "cost_consume": [0. for _ in range(steps)],   # cumulative cost consumption
              
              "energy_start_extra": [0. for _ in range(steps)],
              "cost_start_extra": [0. for _ in range(steps)],
              "cost_wear": [0. for _ in range(steps)],
              "cost_repair": [0. for _ in range(steps)],
              "cost_down": [0. for _ in range(steps)],
              "cost_onoff": [0. for _ in range(steps)],
              
              "cost_dryrun": [0. for _ in range(steps)],
              
              "cost_maintenance": [0. for _ in range(steps)],
              "cost_total": [0. for _ in range(steps)], # cost_consume + cost_maintenance
              
              "ncycles_per_ptype": [0., 0.],                         # for two types(p100, p170)   
              "ndryruns_per_ptype": [0., 0.],
              "total_cycles": 0,
              "total_dryruns": 0,
              "total_energy": 0.0,
              "total_cost": 0.0
              }
        }
        
    for pump in pumps:
        for i in range(steps):
            results[pump.ptype]["energy_consume"][i] += pump.energy_consumption_cum["consume"][i]
            results[pump.ptype]["energy_start_extra"][i] += pump.energy_consumption_cum["start_extra"][i]
            results[pump.ptype]["cost_consume"][i] += pump.energy_cost_cum["consume"][i]
            
            results[pump.ptype]["cost_start_extra"][i] += pump.energy_cost_cum["start_extra"][i]
            results[pump.ptype]["cost_wear"][i] += pump.energy_cost_cum["wear"][i]
            results[pump.ptype]["cost_repair"][i] += pump.energy_cost_cum["repair"][i]
            results[pump.ptype]["cost_down"][i] += pump.energy_cost_cum["down"][i]
            
            results[pump.ptype]["cost_dryrun"][i] += pump.energy_cost_cum["dryrun"][i]
                                                         
            results["total"]["energy_consume"][i] += pump.energy_consumption_cum["consume"][i]
            results["total"]["energy_start_extra"][i] += pump.energy_consumption_cum["start_extra"][i]
            results["total"]["cost_consume"][i] += pump.energy_cost_cum["consume"][i]
            
            results["total"]["cost_start_extra"][i] += pump.energy_cost_cum["start_extra"][i]
            results["total"]["cost_wear"][i] += pump.energy_cost_cum["wear"][i]
            results["total"]["cost_repair"][i] += pump.energy_cost_cum["repair"][i]
            results["total"]["cost_down"][i] += pump.energy_cost_cum["down"][i]
            results["total"]["cost_dryrun"][i] += pump.energy_cost_cum["dryrun"][i]
            
        results[pump.ptype]["ncycles"].append(pump.total_ncycles)
        results[pump.ptype]["ndryruns"].append(pump.total_ndryruns)
        results[pump.ptype]["total_cycles"] += pump.total_ncycles
        results[pump.ptype]["total_energy"] += pump.total_energy_consumption
        results[pump.ptype]["total_cost"] += pump.total_energy_cost
        
        idx = 0 if pump.ptype == "p100" else 1
        results["total"]["ncycles_per_ptype"][idx] += pump.total_ncycles
        results["total"]["ndryruns_per_ptype"][idx] += pump.total_ndryruns
        results["total"]["total_cycles"] += pump.total_ncycles
        results["total"]["total_dryruns"] += pump.total_ndryruns
        results["total"]["total_energy"] += pump.total_energy_consumption
        results["total"]["total_cost"] += pump.total_energy_cost
    
    for key in list(results.keys())[:-1]: # only for p100 and p170
        for i in range(steps):
            results[key]["cost_onoff"][i] += results[key]["cost_start_extra"][i]\
                                          + results[key]["cost_wear"][i]\
                                          + results[key]["cost_repair"][i]\
                                          + results[key]["cost_down"][i]
            results[key]["cost_maintenance"][i] += results[key]["cost_onoff"][i] + results[key]["cost_dryrun"][i]
            results[key]["cost_total"][i] += results[key]["cost_maintenance"][i] + results[key]["cost_consume"][i]
            
            results["total"]["cost_onoff"][i] += results[key]["cost_onoff"][i]
            results["total"]["cost_maintenance"][i] += results[key]["cost_maintenance"][i]
            results["total"]["cost_total"][i] += results[key]["cost_total"][i]
    
    return results   


def execute_scenario_fordays(actions:list, infos:list, lengths_cum:list, minutes:int=1, init_cycles:int=0):
    assert len(actions) == len(infos), "cost_model_v2.py: Mismatch len(actions_scenario) == len(infors_scenario)"
    
    # data structure for days(36)
    dstruct = {"p100": {"energy_consume":[0.0], 
                        "cost_consume":[0.0],
                        "cost_start_extra":[0.0],
                        "cost_wear":[0.0],
                        "cost_repair":[0.0],
                        "cost_down":[0.0],
                        "cost_onoff":[0.0],       # cost_start_extra + cost_wear + cost_repari + cost_down
                        "cost_dryrun":[0.0],
                        "cost_maintenance":[0.0], # cost_onoff + cost_dryrun
                        "cost_total":[0.0]        # energy_cost + cost_maintenance 
                        }, 
               "p170": {"energy_consume":[0.0], 
                        "cost_consume":[0.0],
                        "cost_start_extra":[0.0],
                        "cost_wear":[0.0],
                        "cost_repair":[0.0],
                        "cost_down":[0.0],
                        "cost_onoff":[0.0],       # cost_start_extra + cost_wear + cost_repari + cost_down
                        "cost_dryrun":[0.0],
                        "cost_maintenance":[0.0], # cost_onoff + cost_dryrun
                        "cost_total":[0.0]        # energy_cost + cost_maintenance 
                        },
               "total": {"energy_consume":[0.0], 
                         "cost_consume":[0.0],
                         "cost_start_extra":[0.0],
                         "cost_wear":[0.0],
                         "cost_repair":[0.0],
                         "cost_down":[0.0],
                         "cost_onoff":[0.0],       # cost_start_extra + cost_wear + cost_repari + cost_down
                         "cost_dryrun":[0.0],
                         "cost_maintenance":[0.0], # cost_onoff + cost_dryrun
                         "cost_total":[0.0]        # energy_cost + cost_maintenance 
                         }
              } 
    
    pumps = execute_scenario(actions, infos, minutes=minutes)
    results = analysis_result(pumps)
    
    terms = ["energy_consume", "cost_consume", "cost_start_extra", "cost_wear",
            "cost_repair", "cost_down", "cost_onoff", "cost_dryrun",
            "cost_maintenance", "cost_total"]
    for key in dstruct.keys():
        for term in terms:
            for n in lengths_cum:
                dstruct[key][term].append(results[key][term][n-1])
            
    return dstruct, pumps, results


def save_results(metrics:dict, flood:int, count:int, file:str='results'):
    """
    전체 테스트 데이터((year, duration) 별)를 대상으로 최고수위평균, 펌프변경수평균, 펌프100사용수평규,
    펌프170사용수평균, 과다펌핑평균, 홍수발생수, 시나리오수 반환

    Parameters
    ----------
    metrics : dict
        DESCRIPTION.
    overpump : int
        DESCRIPTION.
    flood : int
        DESCRIPTION.
    count : int
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
        for row in vec_fun(count):
            value = ','.join(row)
            fd.write(value+'\n')
        fd.write('\n')
        for _, value in metrics.items():
            for row in vec_fun(value):
                value = ','.join(row)
                fd.write(value+'\n')
            fd.write('\n')
        for row in vec_fun(flood):
            value = ','.join(row)
            fd.write(value+'\n')
  


# if __name__ == '__main__':
#     params = {"minutes":2, "init_cycles":15000}
#     inps = utils.inp_select_for_scenarios()

#     # Test for DQN
#     model_path ='../trained_models/dqn_gru_pumpmodel_w2_w1.pkl'
#     actions_scenario = []
#     infos_scenario = []
#     lengths = [] # 시나리오에 속하는 각 inp 파일의 actions의 길이 -> inp(or day) 별 상황 변화 확인을 위해
#     for inp in inps:
#         model, actions, states, rewards, infos = dqn_gru_torch.test_model(None, inp, minutes=params["minutes"], model_path=model_path)
#         lengths.append(len(actions))
#         actions_scenario.extend(actions)
#         infos_scenario.extend(infos)
#     lengths = list(np.array(lengths).cumsum())
#     dqn_dstruct, dqn_pumps, dqn_results = execute_scenario_fordays(actions_scenario, 
#                           infos_scenario, lengths,  
#                           params["minutes"], params["init_cycles"])
    
#     # Test for MAN
#     actions_scenario = []
#     infos_scenario = []
#     lengths = [] # 시나리오에 속하는 각 inp 파일의 actions의 길이 -> inp(or day) 별 상황 변화 확인을 위해
#     for inp in inps:
#         actions, states, rewards, infos = man_policy.operation(inp, minutes=params["minutes"])
#         lengths.append(len(actions))
#         actions_scenario.extend(actions)
#         infos_scenario.extend(infos)
#     lengths = list(np.array(lengths).cumsum())
#     man_dstruct, man_pumps, man_results = execute_scenario_fordays(actions_scenario, 
#                           infos_scenario, lengths,  
#                           params["minutes"], params["init_cycles"])
    
    
# %%   
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
    
    # years = ['100']#['10', '20', '30', '50', '80', '100']
    # durations = ['1440'] #['0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    # minutes = 2 # the interval minutes for simulation
    # epochs = 1
    
    # inp, train_inps, valid_inps, test_inps = stratified_split_data(years, 
    #                                                                 durations, 
    #                                                                 trate=0.1, 
    #                                                                 vrate=0.8, 
    #                                                                 stratify=True)
    # # print(f'trains: {len(train_inps)}, tests: {len(test_inps)}')  
    # test_inp = test_inps[0]
    # print(test_inps[0])
    # # actions, states, rewards, infos = man_policy.operation(test_inp, minutes)
    # tmodel, actions, states, rewards, infos = dqn_gru_torch.test_model(None, test_inp, minutes=minutes)
    
    # pumps = execute_scenario(actions, infos, minutes=minutes, init_cycles=10000)
    # results = analysis_result(pumps)