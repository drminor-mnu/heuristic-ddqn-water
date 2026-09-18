import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parsed(filename):
    return ast.parse((ROOT / filename).read_text(encoding="utf-8"))


def keyword_values(call):
    values = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            pass
    return values


def find_call(tree, name):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def test_runner_uses_requested_split_and_local_search_configuration():
    tree = parsed("run_local_search_test.py")
    split = keyword_values(find_call(tree, "stratified_split_data"))
    evaluate = keyword_values(find_call(tree, "_evaluate_heuristic"))

    params_assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "params"
                for target in node.targets)
    )
    params = ast.literal_eval(params_assignment.value)

    assert split == {"trate": 0.6, "vrate": 0.0, "stratify": True}
    assert params == {
        "minutes": 2,
        "weights": [0.25, 0.4, 0.25, 0.1],
        "act_type": 0,
        "neighborhood_size": 3,
        "num_restarts": 1,
    }
    assert evaluate["optimize_label"] == "Local Search (neighborhood=3)"
    assert evaluate["result_file"] == "local_search_test"


def test_evaluator_supports_exact_name_and_keeps_legacy_fallback():
    tree = parsed("perform_evaluate.py")
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_evaluate_heuristic"
    )
    positional = [argument.arg for argument in function.args.args]
    defaults = [ast.literal_eval(value) for value in function.args.defaults]
    assert positional[-1] == "result_file"
    assert defaults[-1] is None

    source = ast.unparse(function)
    assert "result_file or f'{result_prefix}_{str_w}'" in source
