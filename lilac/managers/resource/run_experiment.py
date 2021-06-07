"""lilac runコマンドの実体."""
import json
import os

from lilac.ensemble.stacking_runner import StackingRunner
from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.tasks.run_cv import RunCv
from lilac.tuner.tasks_runner import TasksRunnerWithOptuna
from lilac.utils.utils import MyEncoder


def run_tasks(Task, members, n_trials,  base_params, token, app_name, channel, do_notify, do_plot, tune_fs, tune_th):
    # 使用するevaluatorからdirectionを作成
    direction = EvaluatorFactory(base_params["target_col"]).run(
        base_params["evaluator_flag"]).get_direction()
    # 1段目実行
    task_runner = TasksRunnerWithOptuna(
        Task, direction, base_params["seed"], token, app_name, channel, do_notify, do_plot, tune_fs, tune_th)
    tasks = task_runner.run(base_params, members, n_trials)

    # 結果取り出し
    output_list = []
    for task in tasks:
        output_path = task.output()
        with output_path.open("r") as f:
            cv_output = json.load(f)
        output_list.append(cv_output)
    return tasks, output_list


def run_stacking(stackings, base_params, tasks, output_list, use_original_features):
    layers = stackings["layers"]

    ensemble_params = stackings.get("params")
    if ensemble_params:
        # stacking全体のパラメータを書き換える
        print(f"Update params with : {ensemble_params}")
        base_params.update(ensemble_params)

    # stacking
    stacking_runner = StackingRunner(
        layers, base_params, use_original_features)
    return stacking_runner.run(output_list)


def logging(task_results, stacking_results, members, stackings):
    print("===== Lilac Execution Result =====")
    print("")
    layers_num = len(stacking_results)
    print(f"[Layer: 0/{layers_num}]")
    for i, output in enumerate(task_results):
        print(
            f"[{', '.join(map(str, members[i].values()))}]:\t{output['evaluator']} = {output['score']}")

    for layer_i, layer in enumerate(stacking_results):
        print(f"[Layer: {layer_i+1}/{layers_num}]")
        for result_i, result in enumerate(layer):
            print(
                f"[{stackings['layers'][layer_i][result_i]}]:\t{result['evaluator']} = {result['score']}")
    print("")


def run_experiment(settings_path, key, trials, app_name, channel, notify, plot, tune_fs, tune_th, output_dir):
    with open(settings_path, "r") as f:
        settings = json.load(f)
    base_params = settings.pop("default")
    base_params["settings_path"] = settings_path

    members = settings["run"][key]["members"]
    stackings = settings["stacking"][settings["run"][key]["stacking_key"]]
    token = os.environ["SLACK_TOKEN"]
    # luigiのパラメータではないのでpopしておく
    use_original_features = base_params.pop("use_original_features")

    # tasks実行
    tasks, task_results = run_tasks(RunCv, members, trials,
                                    base_params, token, app_name, channel, notify, plot, tune_fs, tune_th)

    # stacking実行
    stacking_results = run_stacking(
        stackings, base_params, tasks, task_results, use_original_features)

    # logging
    logging(task_results, stacking_results, members, stackings)

    result = {
        "details": [task_results] + stacking_results,
        "settings": {
            "members": members,
            "stackings": stackings,
            "trials": trials,
            "tune_fs": tune_fs,
            "tune_th": tune_th
        }
    }
    result["output"] = result["details"][-1][0]

    output_path = f"{output_dir}/{key}_{trials}_{tune_fs}_{tune_th}.json"

    if not trials:
        print("Used default hyperparameters.")
    else:
        print(
            f"Tuned hyperparameters with optuna. (trials : {trials})")

    print(
        f"CV score ({result['output']['evaluator']}) : {result['output']['score']}")
    print(f"Output path: {output_path}")

    with open(output_path, "w") as f:
        json.dump(result, f, cls=MyEncoder)
