import json
import yaml
from pathlib import Path

import pandas as pd

from lilac.tasks.run_cv import RunCv
from lilac.ensemble.stacking_runner import StackingRunner
from lilac.tuner.tasks_runner import TasksRunnerWithOptuna
from lilac.evaluators.evaluator_factory import EvaluatorFactory
import os


def run_tasks(Task, members, n_trials, stackings, base_params, token, app_name, channel, do_notify, do_plot, tune_fs, tune_th):
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

    # 1段目CV表示
    print("Layer 0")
    for i, output in enumerate(output_list):
        print(
            f"[{', '.join(map(str,members[i].values()))}]: {output['evaluator']} = {output['score']}")
    print("=============================")

    # １つ目のrunで使ったデータセットの、特徴量選択前を使う
    # group kfoldの特徴量がない可能性があるため.
    path = Path(tasks[0].output().path)
    train = pd.read_csv(path.parent.parent.parent/"train.csv")
    test = pd.read_csv(path.parent.parent.parent/"test.csv")

    # key_colはgroup kfoldのときに使われる
    folds_gen_params = {"fold_num":  base_params["fold_num"],
                        "seed":  base_params["seed"],
                        "key_col":  base_params["group_kfolds_col"],
                        "target_col": base_params["target_col"]}
    trainer_params = {
        "target_col": base_params["target_col"],
        "bagging_num": base_params["bagging_num"],
        "base_class": base_params["base_class"],
        "seed": base_params["seed"],
        "allow_less_than_base": base_params["allow_less_than_base"]
    }

    # stacking
    stacking_runner = StackingRunner(
        stackings, base_params["target_col"], base_params["folds_generator_flag"], folds_gen_params,
        base_params["trainer_flag"], trainer_params)
    result = stacking_runner.run(output_list, train, test)
    return result


def run_experiment(args):
    with open(args.config_path, "r") as f:
        base_params = yaml.load(f)

    with open(base_params["experiment_path"], "r") as f:
        config = json.load(f)

    members = config["run"][args.key]["members"]
    stackings = config["stacking"][config["run"][args.key]["stacking_key"]]
    token = os.environ["SLACK_TOKEN"]

    result = run_tasks(RunCv, members, args.trials, stackings,
                       base_params, token, args.app_name, args.channel, args.notify, args.plot, args.tune_fs, args.tune_th)

    output_path = f"{args.output_dir}/{args.key}_{args.trials}_{args.tune_fs}_{args.tune_th}.json"

    if not args.trials:
        print("Used default hyperparameters.")
    else:
        print(
            f"Tuned hyperparameters with optuna. (trials : {args.trials})")

    print(f"CV score ({result['evaluator']}) : {result['score']}")
    print(f"Output path: {output_path}")
    return result, output_path
