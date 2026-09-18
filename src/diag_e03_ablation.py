"""E3 diagnostic step 3 (redesigned): isolate which training defect drives the
~100% overflow collapse, by toggling one defect at a time.

DOES NOT modify any original code path. This is a standalone harness that
re-implements the Regular DDQN `train()` / `test_model()` loop from
dqn_from_demon_v1.py with flags for each suspected defect:

  norm_mode : 'sum'    -> original  state / (state.sum() + 1e-5)
              'minmax'  -> water_gym.maxval/minval component standardization
  gamma     : discount (original hardcodes 0.1)
  sync_mode : 'step'    -> original  target sync on env-step index  j % sync_freq
              'update'  -> target sync on optimizer-update count % sync_freq
  weights   : reward weights (E3 uses [0.25,0.4,0.25,0.1]; manuscript Regular [1,1,0,0])

Conditions:
  D0 sum   / g0.1 / step   / w[.25,.4,.25,.1]   (current behaviour, baseline)
  D1 minmax/ g0.1 / step   / w[.25,.4,.25,.1]   (normalization only)
  D2 sum   / g0.9 / step   / w[.25,.4,.25,.1]   (gamma only)
  D3 sum   / g0.1 / update / w[.25,.4,.25,.1]   (sync only)
  D4 minmax/ g0.9 / step   / w[.25,.4,.25,.1]   (norm + gamma)
  D5 minmax/ g0.9 / update / w[.25,.4,.25,.1]   (all three)
  D6 sum   / g0.1 / step   / w[1,1,0,0]         (manuscript 2-criterion reward)

Small config: 500 train scenarios (first 500 of the seed-42 fixed split, same
set for every condition), 30 test scenarios (first 30), seeds {1,2}.
Metrics: overflow rate, mean max_level_m, action-0 lock rate, mean n_intervals_100.

Writes results/summary/E03_ablation.csv incrementally and E03_ablation.log.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import json
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.chdir(SRC)  # water_gym/sim_swmm use ../data relative paths

OUT_CSV = os.path.join(ROOT, "results", "summary", "E03_ablation.csv")
OUT_LOG = os.path.join(ROOT, "results", "summary", "E03_ablation.log")

NSTATES = 5
HIDDEN = 32
LAYERS = 3
MINUTES = 2
MEMSIZE = 3000
BATCH = 20
LR = 1e-3
LEVEL_MAX = 10.0  # water_gym.Level[-1]

# state vector = [rain, inflow, outflow, vol, level]; mapping to water_gym.maxval/minval
NORM_MIN = np.array([0.0, 0.0, 0.0, 0.0, 4.7])
NORM_MAX = np.array([5.55, 1750.0, 1750.0, 32522.0, 10.0])

CONDITIONS = {
    "D0": dict(norm_mode="sum",    gamma=0.1, sync_mode="step",   sync_freq=10,  weights=[0.25, 0.4, 0.25, 0.1]),
    "D1": dict(norm_mode="minmax", gamma=0.1, sync_mode="step",   sync_freq=10,  weights=[0.25, 0.4, 0.25, 0.1]),
    "D2": dict(norm_mode="sum",    gamma=0.9, sync_mode="step",   sync_freq=10,  weights=[0.25, 0.4, 0.25, 0.1]),
    "D3": dict(norm_mode="sum",    gamma=0.1, sync_mode="update", sync_freq=200, weights=[0.25, 0.4, 0.25, 0.1]),
    "D4": dict(norm_mode="minmax", gamma=0.9, sync_mode="step",   sync_freq=10,  weights=[0.25, 0.4, 0.25, 0.1]),
    "D5": dict(norm_mode="minmax", gamma=0.9, sync_mode="update", sync_freq=200, weights=[0.25, 0.4, 0.25, 0.1]),
    "D6": dict(norm_mode="sum",    gamma=0.1, sync_mode="step",   sync_freq=10,  weights=[1.0, 1.0, 0.0, 0.0]),
    # --- step 1: gamma sweep (4-criterion, sum norm, step sync = current code) ---
    # gamma 0.1 == D0, gamma 0.9 == D2 (reuse those; only run the gaps)
    "GS03":  dict(norm_mode="sum", gamma=0.3,  sync_mode="step", sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    "GS05":  dict(norm_mode="sum", gamma=0.5,  sync_mode="step", sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    "GS07":  dict(norm_mode="sum", gamma=0.7,  sync_mode="step", sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    "GS095": dict(norm_mode="sum", gamma=0.95, sync_mode="step", sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    # --- step 2: cumulative defect removal at candidate gamma (episode-based sync = prior-paper "10 samples") ---
    # G?0 (gamma only) already covered: gamma0.7 == GS07, gamma0.9 == D2
    "G7_1": dict(norm_mode="minmax", gamma=0.7, sync_mode="step",    sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    "G7_2": dict(norm_mode="minmax", gamma=0.7, sync_mode="episode", sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    "G9_1": dict(norm_mode="minmax", gamma=0.9, sync_mode="step",    sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
    "G9_2": dict(norm_mode="minmax", gamma=0.9, sync_mode="episode", sync_freq=10, weights=[0.25, 0.4, 0.25, 0.1]),
}


def normalize(vec, mode):
    a = np.asarray(vec, dtype=float)
    if mode == "sum":
        return a / (a.sum() + 1e-5)
    if mode == "minmax":
        return (a - NORM_MIN) / (NORM_MAX - NORM_MIN)
    raise ValueError(mode)


def count_changes(actions, Actions):
    c = 0
    for i in range(len(actions) - 1):
        c += int((np.array(Actions[actions[i]]) ^ np.array(Actions[actions[i + 1]])).sum())
    return c


def count_pumps(actions, Actions):
    npumps = np.zeros(len(Actions[0]), dtype=int)
    for a in actions:
        npumps += np.array(Actions[a], dtype=int)
    return int(npumps[:3].sum()), int(npumps[3:].sum())


def build_state(input_q, device):
    import torch
    return torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)


def train_ablation(train_inps, cond, seed, log):
    import torch
    from water_gym import WaterGym, Actions0
    from dqn_model import DqnGRU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = Actions0
    norm_mode = cond["norm_mode"]
    gamma = cond["gamma"]
    sync_mode = cond["sync_mode"]
    sync_freq = cond["sync_freq"]
    weights = cond["weights"]

    policy_model = DqnGRU(input_dim=5, hidden_dim=HIDDEN, output_dim=len(actions), num_layers=LAYERS).to(device)
    target_model = DqnGRU(input_dim=5, hidden_dim=HIDDEN, output_dim=len(actions), num_layers=LAYERS).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)

    epsilon = 1.0
    replay = deque(maxlen=MEMSIZE)
    update_count = 0

    inps = list(train_inps)
    random.shuffle(inps)
    num_samples = len(inps)
    ep_losses, ep_rewards = [], []

    for i in range(num_samples):
        gym = WaterGym(inps[i], MINUTES, weights, 0)
        state_, reward, done, info = gym.reset()
        input_q = deque([], NSTATES)
        for _ in range(NSTATES - 1):
            input_q.append(np.zeros(len(state_)))
        input_q.append(normalize(state_, norm_mode))
        state = build_state(input_q, device)

        j = 0
        losses = []
        total_reward = 0.0
        while True:
            j += 1
            state_prev = state.cpu()
            q_val, _ = policy_model(state.unsqueeze(0))
            q_np = q_val.cpu().data.numpy()
            if random.random() < epsilon:
                action = np.random.randint(0, len(actions))
            else:
                action = int(np.argmax(q_np))

            state_, reward, done, info = gym.step(action)
            total_reward += reward
            input_q.append(normalize(state_, norm_mode))
            state = build_state(input_q, device)

            replay.append((state_prev, action, reward, state.cpu(), done))

            if len(replay) > BATCH:
                mb = random.sample(replay, BATCH)
                s1 = torch.stack([x[0] for x in mb]).to(device)
                ab = torch.tensor([x[1] for x in mb], device=device)
                rb = torch.tensor([x[2] for x in mb], dtype=torch.float32, device=device)
                s2 = torch.stack([x[3] for x in mb]).to(device)
                db = torch.tensor([x[4] for x in mb], dtype=torch.float32, device=device)

                q1, _ = policy_model(s1)
                with torch.no_grad():
                    q1_next, _ = policy_model(s2)
                    a_star = torch.argmax(q1_next, dim=1, keepdim=True)
                    q2, _ = target_model(s2)
                    tq = q2.gather(1, a_star).squeeze(1)
                Y = rb + gamma * (1.0 - db) * tq
                X = q1.gather(1, ab.long().unsqueeze(1)).squeeze(1)
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                update_count += 1
                if sync_mode == "update" and update_count % sync_freq == 0:
                    target_model.load_state_dict(policy_model.state_dict())

            if sync_mode == "step" and j % sync_freq == 0:
                target_model.load_state_dict(policy_model.state_dict())
            if done:
                break

        if sync_mode == "episode" and (i + 1) % sync_freq == 0:
            target_model.load_state_dict(policy_model.state_dict())

        if epsilon > 0.2:
            epsilon -= (1.0 / num_samples)
        ep_losses.append(float(np.mean(losses)) if losses else float("nan"))
        ep_rewards.append(total_reward)
        if (i + 1) % 100 == 0:
            log(f"    [{cond['tag']} s{seed}] ep {i+1}/{num_samples} "
                f"loss={ep_losses[-1]:.5f} reward={total_reward:.1f} eps={epsilon:.3f} updates={update_count}")

    return policy_model, update_count, ep_losses, ep_rewards


def test_ablation(policy_model, test_inps, norm_mode):
    import torch
    from water_gym import WaterGym, Actions0, Level

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = Actions0
    policy_model.eval()
    rows = []
    for inp in test_inps:
        gym = WaterGym(inp, MINUTES, [0.25, 0.4, 0.25, 0.1], 0)
        s0, reward, done, info = gym.reset()
        input_q = deque([], NSTATES)
        for _ in range(NSTATES - 1):
            input_q.append(np.zeros(len(s0)))
        input_q.append(normalize(s0, norm_mode))
        state = build_state(input_q, device)

        act_list = [0]
        levels = [s0[-1]]
        while True:
            with torch.no_grad():
                q_val, _ = policy_model(state.unsqueeze(0))
            action = int(np.argmax(q_val.cpu().data.numpy()))
            act_list.append(action)
            s_, reward, done, info = gym.step(action)
            levels.append(s_[-1])
            input_q.append(normalize(s_, norm_mode))
            state = build_state(input_q, device)
            if done:
                break

        highest = max(levels)
        n_sw = count_changes(act_list, Actions0)
        p100, p170 = count_pumps(act_list, Actions0)
        rows.append(dict(
            scenario_id=os.path.basename(inp).replace(".inp", ""),
            max_level_m=highest,
            overflow_flag=int(highest >= Level[-1]),
            n_switches=n_sw,
            n_intervals_100=p100,
            n_intervals_170=p170,
            action0_locked=int(n_sw == 0 and p100 == 0 and p170 == 0),
        ))
    return rows


def run_task(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)

    tag = task["tag"]
    seed = task["seed"]
    n_train = task["n_train"]
    n_test = task["n_test"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_paths
    train_all, _, test_all = data_paths.load_fixed_split()
    train_inps = [data_paths.resolve_path(p) for p in train_all[:n_train]]
    test_inps = [data_paths.resolve_path(p) for p in test_all[:n_test]]

    cond = dict(CONDITIONS[tag]); cond["tag"] = tag
    logbuf = []

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        logbuf.append(line)
        print(line, flush=True)

    t0 = time.perf_counter()
    log(f"START {tag} seed{seed}  cond={ {k: cond[k] for k in ('norm_mode','gamma','sync_mode','sync_freq','weights')} }")
    model, updates, losses, rewards = train_ablation(train_inps, cond, seed, log)
    rows = test_ablation(model, test_inps, cond["norm_mode"])
    dt = time.perf_counter() - t0

    ov = np.mean([r["overflow_flag"] for r in rows])
    ml = np.mean([r["max_level_m"] for r in rows])
    lk = np.mean([r["action0_locked"] for r in rows])
    p100 = np.mean([r["n_intervals_100"] for r in rows])
    p170 = np.mean([r["n_intervals_170"] for r in rows])
    sw = np.mean([r["n_switches"] for r in rows])
    res = dict(
        tag=tag, seed=seed, n_train=n_train, n_test=n_test,
        norm_mode=cond["norm_mode"], gamma=cond["gamma"],
        sync_mode=cond["sync_mode"], sync_freq=cond["sync_freq"],
        weights="|".join(str(w) for w in cond["weights"]),
        updates=updates,
        overflow_rate=round(float(ov), 4),
        mean_max_level_m=round(float(ml), 4),
        action0_lock_rate=round(float(lk), 4),
        mean_n_intervals_100=round(float(p100), 2),
        mean_n_intervals_170=round(float(p170), 2),
        mean_n_switches=round(float(sw), 2),
        final_train_loss=round(float(np.nanmean(losses[-50:])), 6),
        final_train_reward=round(float(np.mean(rewards[-50:])), 2),
        elapsed_s=round(dt, 1),
    )
    log(f"DONE {tag} seed{seed}  overflow={ov:.3f} max_level={ml:.3f} "
        f"act0_lock={lk:.3f} i100={p100:.1f}  ({dt:.0f}s)")
    return res, logbuf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--conditions", default="D0,D1,D2,D3,D4,D5,D6")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    tags = [t.strip() for t in args.conditions.split(",")]
    tasks = [dict(tag=t, seed=s, n_train=args.n_train, n_test=args.n_test)
             for t in tags for s in seeds]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["tag", "seed", "n_train", "n_test", "norm_mode", "gamma", "sync_mode",
              "sync_freq", "weights", "updates", "overflow_rate", "mean_max_level_m",
              "action0_lock_rate", "mean_n_intervals_100", "mean_n_intervals_170",
              "mean_n_switches", "final_train_loss", "final_train_reward", "elapsed_s"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    logf = open(OUT_LOG, "w")

    def flog(m):
        logf.write(m + "\n"); logf.flush()

    flog(f"start {time.ctime()}  tasks={len(tasks)} workers={args.workers} "
         f"n_train={args.n_train} n_test={args.n_test}")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res, logbuf = fut.result()
            results.append(res)
            for l in logbuf:
                flog(l)
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(res)
            flog(f"  >> wrote {res['tag']} seed{res['seed']}  "
                 f"overflow={res['overflow_rate']} max_level={res['mean_max_level_m']} "
                 f"act0={res['action0_lock_rate']}")
            print(f">>> {len(results)}/{len(tasks)} done", flush=True)

    flog(f"all done in {time.perf_counter()-t0:.0f}s")

    # aggregate over seeds
    import pandas as pd
    df = pd.DataFrame(results)
    agg = df.groupby("tag").agg(
        overflow_rate=("overflow_rate", "mean"),
        mean_max_level_m=("mean_max_level_m", "mean"),
        action0_lock_rate=("action0_lock_rate", "mean"),
        mean_n_intervals_100=("mean_n_intervals_100", "mean"),
        mean_n_switches=("mean_n_switches", "mean"),
    ).reindex(tags)
    agg.to_csv(os.path.join(ROOT, "results", "summary", "E03_ablation_agg.csv"))
    print("\n=== E3 ablation (mean over seeds) ===")
    print(agg.to_string())
    flog("\n" + agg.to_string())
    logf.close()


if __name__ == "__main__":
    main()
