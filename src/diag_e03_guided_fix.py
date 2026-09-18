"""E3 diagnostic 3-3: does the 4-criterion GUIDED DDQN survive with the fix
settled in steps 1-2 (gamma 0.7 + minmax input normalization)?

Standalone harness. Re-implements dqn_from_demon_v1._train_guided_by_optimizer
faithfully. Configurable via CLI:
  --gamma  discount factor (default 0.7; the original hardcodes/passes 0.1)
  --norm   input normalization: 'minmax' (default) or 'sum' (== original
           state/(state.sum()+1e-5))
Everything else identical to the E3 guided run: ga_prob 0.6->0.0, eps 0.3->0.05,
sync every 200 optimizer updates (original guided loop already does this),
batch 20, lr 1e-3, replay 3000, weights [0.25,0.4,0.25,0.1], 1 epoch.

FIDELITY CHECK: run with --gamma 0.1 --norm sum --tag ctrl -- this should
reproduce diag_e03_guided.py (real evaluate_dpn_guided_* path) results:
GA-guided overflow ~1.00, PSO-guided action0-lock ~1.00.

Small config: 500 train (first 500 of seed-42 fixed split) / 30 test (first 30) /
seeds {1,2}. Compare to Regular-fixed G7_1 (diag_e03_ablation): overflow 0/0, max_level 8.27.

Writes results/summary/E03_guided_fix<_tag>.csv.
"""
from __future__ import annotations
import os
import sys
import csv
import copy
import time
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.chdir(SRC)

OUT_CSV = os.path.join(ROOT, "results", "summary", "E03_guided_fix.csv")

NSTATES = 5
HIDDEN = 32
LAYERS = 3
MINUTES = 2
MEMSIZE = 3000
BATCH = 20
LR = 1e-3
WEIGHTS = [0.25, 0.4, 0.25, 0.1]
GA_START, GA_END = 0.6, 0.0
EPS_START, EPS_END = 0.3, 0.05
SYNC_FREQ = 200          # optimizer-update count (matches E3 guided)
MIN_REPLAY = 100
N_TRAIN, N_TEST = 500, 30

NORM_MIN = np.array([0.0, 0.0, 0.0, 0.0, 4.7])
NORM_MAX = np.array([5.55, 1750.0, 1750.0, 32522.0, 10.0])


def normalize(vec, mode):
    a = np.asarray(vec, dtype=np.float32)
    if mode == "sum":
        return (a / (a.sum() + 1e-5)).astype(np.float32)
    if mode == "minmax":
        return ((a - NORM_MIN) / (NORM_MAX - NORM_MIN)).astype(np.float32)
    raise ValueError(mode)


def count_changes(actions, A):
    return sum(int((np.array(A[actions[i]]) ^ np.array(A[actions[i + 1]])).sum())
              for i in range(len(actions) - 1))


def count_pumps(actions, A):
    n = np.zeros(len(A[0]), dtype=int)
    for a in actions:
        n += np.array(A[a], dtype=int)
    return int(n[:3].sum()), int(n[3:].sum())


def train_guided_fixed(train_inps, guide, seed, log, gamma, norm_mode):
    import torch
    from water_gym import WaterGym, Actions0
    from dqn_model import DqnGRU
    from genetic_algo import GeneticAlgorithm
    from pso import ParticleSwarmOptimization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = Actions0
    if guide == "GA":
        opt_class, opt_method = GeneticAlgorithm, "genetic_algorithm"
    else:
        opt_class, opt_method = ParticleSwarmOptimization, "particle_swarm_optimization"

    params = dict(minutes=MINUTES, weights=WEIGHTS, act_type=0,
                  ga_start=GA_START, ga_end=GA_END, eps_start=EPS_START, eps_end=EPS_END,
                  gamma=gamma, batch_size=BATCH, learning_rate=LR, sync_freq=SYNC_FREQ,
                  min_replay_size=MIN_REPLAY)

    policy_model = DqnGRU(input_dim=5, hidden_dim=HIDDEN, output_dim=len(actions), num_layers=LAYERS).to(device)
    target_model = DqnGRU(input_dim=5, hidden_dim=HIDDEN, output_dim=len(actions), num_layers=LAYERS).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)
    replay = deque(maxlen=MEMSIZE)
    update_count = 0

    inps = copy.deepcopy(list(train_inps))
    random.shuffle(inps)
    num_samples = len(inps)

    for i in range(num_samples):
        progress = i / (num_samples - 1) if num_samples > 1 else 1.0
        ga_prob = max(0.0, min(1.0, GA_START - (GA_START - GA_END) * progress))
        epsilon = max(0.0, min(1.0, EPS_START - (EPS_START - EPS_END) * progress))

        guide_optimizer = opt_class(**params)
        action_list = [0]

        gym = WaterGym(inps[i], MINUTES, WEIGHTS, act_type=0)
        state_, reward, done, info = gym.reset()
        ga_state = state_

        input_q = deque([], NSTATES)
        for _ in range(NSTATES - 1):
            input_q.append(np.zeros(len(state_), dtype=np.float32))
        input_q.append(normalize(state_, norm_mode))
        state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)

        losses = []
        guided_count = 0
        step_count = 0
        status = 1
        while status:
            step_count += 1
            state_prev = state.detach().cpu().clone()
            with torch.no_grad():
                q_val, _ = policy_model(state.unsqueeze(0))
                q_val_np = q_val.detach().cpu().numpy().squeeze(0)

            if random.random() < ga_prob:
                action = getattr(guide_optimizer, opt_method)(
                    ga_state, action_list, next_inflow=gym.outfalls[gym.clock + 1])
                guided_count += 1
            else:
                if random.random() < epsilon:
                    action = np.random.randint(0, len(actions))
                else:
                    action = int(np.argmax(q_val_np))

            next_state_, reward, done, info = gym.step(action)
            ga_state = next_state_
            action_list.append(action)

            input_q.append(normalize(next_state_, norm_mode))
            next_state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            replay.append((state_prev, action, reward, next_state.detach().cpu().clone(), done))
            state = next_state

            if len(replay) >= MIN_REPLAY:
                mb = random.sample(replay, BATCH)
                s1 = torch.stack([x[0] for x in mb]).to(device)
                ab = torch.tensor([x[1] for x in mb], dtype=torch.long, device=device)
                rb = torch.tensor([x[2] for x in mb], dtype=torch.float32, device=device)
                s2 = torch.stack([x[3] for x in mb]).to(device)
                db = torch.tensor([x[4] for x in mb], dtype=torch.float32, device=device)
                q1, _ = policy_model(s1)
                cur_q = q1.gather(1, ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qon, _ = policy_model(s2)
                    na = torch.argmax(qon, dim=1, keepdim=True)
                    qtn, _ = target_model(s2)
                    nq = qtn.gather(1, na).squeeze(1)
                    tgt = rb + gamma * (1.0 - db) * nq
                loss = loss_fn(cur_q, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                update_count += 1
                if update_count % SYNC_FREQ == 0:
                    target_model.load_state_dict(policy_model.state_dict())

            if done:
                status = 0
        if (i + 1) % 100 == 0:
            log(f"  [{guide} s{seed}] ep {i+1}/{num_samples} "
                f"loss={np.mean(losses) if losses else float('nan'):.5f} "
                f"ga_prob={ga_prob:.3f} eps={epsilon:.3f} updates={update_count}")

    return policy_model


def test_guided_fixed(model, test_inps, norm_mode):
    import torch
    from water_gym import WaterGym, Actions0, Level
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rows = []
    for inp in test_inps:
        gym = WaterGym(inp, MINUTES, WEIGHTS, act_type=0)
        s0, r, done, info = gym.reset()
        input_q = deque([], NSTATES)
        for _ in range(NSTATES - 1):
            input_q.append(np.zeros(len(s0), dtype=np.float32))
        input_q.append(normalize(s0, norm_mode))
        state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
        acts = [0]
        levels = [s0[-1]]
        dry = 0
        while True:
            with torch.no_grad():
                qv, _ = model(state.unsqueeze(0))
            a = int(np.argmax(qv.cpu().numpy()))
            acts.append(a)
            s_, r, done, info = gym.step(a)
            levels.append(s_[-1])
            try:
                dry += int(info[-1])
            except Exception:
                pass
            input_q.append(normalize(s_, norm_mode))
            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            if done:
                break
        hi = max(levels)
        p100, p170 = count_pumps(acts, Actions0)
        nsw = count_changes(acts, Actions0)
        rows.append(dict(max_level_m=hi, overflow_flag=int(hi >= Level[-1]),
                         n_switches=nsw, n_intervals_100=p100, n_intervals_170=p170,
                         n_dryrun_proxy=dry,
                         action0_locked=int(nsw == 0 and p100 == 0 and p170 == 0)))
    return rows


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    guide, seed = task["guide"], task["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_paths
    tr, _, te = data_paths.load_fixed_split()
    train_inps = [data_paths.resolve_path(p) for p in tr[:N_TRAIN]]
    test_inps = [data_paths.resolve_path(p) for p in te[:N_TEST]]

    buf = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; buf.append(line); print(line, flush=True)

    gamma = task["gamma"]
    norm_mode = task["norm"]
    t0 = time.perf_counter()
    log(f"START {guide}-guided-fixed seed{seed}  gamma={gamma} norm={norm_mode}")
    model = train_guided_fixed(train_inps, guide, seed, log, gamma, norm_mode)
    rows = test_guided_fixed(model, test_inps, norm_mode)
    dt = time.perf_counter() - t0
    ov = np.mean([r["overflow_flag"] for r in rows])
    ml = np.mean([r["max_level_m"] for r in rows])
    lk = np.mean([r["action0_locked"] for r in rows])
    res = dict(guide=guide, seed=seed, n_train=N_TRAIN, n_test=N_TEST, gamma=gamma, norm=norm_mode,
               overflow_rate=round(float(ov), 4), mean_max_level_m=round(float(ml), 4),
               action0_lock_rate=round(float(lk), 4),
               mean_n_intervals_100=round(float(np.mean([r["n_intervals_100"] for r in rows])), 2),
               mean_n_switches=round(float(np.mean([r["n_switches"] for r in rows])), 2),
               mean_dryrun_proxy=round(float(np.mean([r["n_dryrun_proxy"] for r in rows])), 3),
               elapsed_s=round(dt, 1))
    log(f"DONE {guide} seed{seed}  overflow={ov:.3f} max_level={ml:.3f} act0={lk:.3f} ({dt:.0f}s)")
    return res, buf


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--norm", choices=["sum", "minmax"], default="minmax")
    ap.add_argument("--tag", default="")
    ap.add_argument("--guides", default="GA,PSO")
    ap.add_argument("--seeds", default="1,2")
    args = ap.parse_args()

    global OUT_CSV
    if args.tag:
        OUT_CSV = OUT_CSV.replace(".csv", f"_{args.tag}.csv")
    guides = [g.strip() for g in args.guides.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    tasks = [dict(guide=g, seed=s, gamma=args.gamma, norm=args.norm)
             for g in guides for s in seeds]
    print(f"config: gamma={args.gamma} norm={args.norm} tag={args.tag or '(none)'} "
          f"tasks={len(tasks)} -> {OUT_CSV}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["guide", "seed", "n_train", "n_test", "gamma", "norm", "overflow_rate",
              "mean_max_level_m", "action0_lock_rate", "mean_n_intervals_100",
              "mean_n_switches", "mean_dryrun_proxy", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    t0 = time.perf_counter()
    n = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res, buf = fut.result()
            n += 1
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            print(f">>> {n}/{len(tasks)} {res['guide']} seed{res['seed']} "
                  f"overflow={res['overflow_rate']} max_level={res['mean_max_level_m']}", flush=True)
    print(f"\nall done {time.perf_counter()-t0:.0f}s")
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    print(df.to_string(index=False))
    print("\nby guide (mean over seeds):")
    print(df.groupby("guide")[["overflow_rate", "mean_max_level_m", "action0_lock_rate",
                               "mean_n_intervals_100", "mean_dryrun_proxy"]].mean().to_string())


if __name__ == "__main__":
    main()
