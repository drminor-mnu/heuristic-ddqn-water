import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso import ParticleSwarmOptimization


ROOT = Path(__file__).resolve().parents[1]


def parsed(filename):
    return ast.parse((ROOT / filename).read_text(encoding="utf-8"))


def function(tree, name):
    return next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_training_wrapper_selects_particle_swarm_optimizer():
    tree = parsed("dqn_from_demon_v1.py")
    wrapper = ast.unparse(function(tree, "train_guided_byPSO"))
    assert "ParticleSwarmOptimization" in wrapper
    assert "particle_swarm_optimization" in wrapper


def test_pso_computes_each_discrete_action_fitness_once_per_state():
    optimizer = ParticleSwarmOptimization(
        swarm_size=10, num_iterations=4, patience=10, act_type=0
    )
    calls = []
    original = optimizer.objective_function

    def counted(action, state, action_list, next_inflow=0.0):
        calls.append(action)
        return original(action, state, action_list, next_inflow)

    optimizer.objective_function = counted
    action = optimizer.particle_swarm_optimization(
        [0.1, 1.0, 0.0, 100.0, 5.0], [0]
    )

    assert 0 <= action < len(optimizer.Actions)
    assert calls == list(range(len(optimizer.Actions)))


def test_evaluation_wrapper_selects_pso_trainer():
    tree = parsed("perform_evaluate.py")
    wrapper = ast.unparse(function(tree, "evaluate_dpn_guided_PSO"))
    assert "train_guided_byPSO" in wrapper
    assert "PSO" in wrapper


def test_runner_uses_exact_requested_configuration_and_names():
    source = (ROOT / "run_dqn_guided_pso.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "dqn_guided_PSO_w0_25_w0_4_w0_25_w0_1_ga0_6_min2_tr0_6_acttype0" in source

    split_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "stratified_split_data"
    )
    split = {kw.arg: ast.literal_eval(kw.value) for kw in split_call.keywords}
    assert split == {"trate": 0.6, "vrate": 0.0, "stratify": True}

    params_assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PARAMS" for target in node.targets)
    )
    assert ast.literal_eval(params_assignment.value) == {
        "epochs": 1,
        "minutes": 2,
        "weights": [0.25, 0.4, 0.25, 0.1],
        "ga_start": 0.6,
        "ga_end": 0.0,
        "eps_start": 0.3,
        "eps_end": 0.05,
        "gamma": 0.1,
        "batch_size": 20,
        "learning_rate": 0.001,
        "sync_freq": 200,
        "act_type": 0,
    }
