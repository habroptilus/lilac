import luigi
import json
import optuna
from lilac.tuner.optuna_params_factory import OptunaParamsFactory
from lilac.utils.utils import SlackApi, plot_feature_importance
from pathlib import Path
import pandas as pd


def objective(params, Task, tune_fs, tune_th):
    def inner_objective(trial):
        # 予測モデルのパラメータをoptunaで置き換え
        model_params = OptunaParamsFactory.run(params["model_flag"], trial)
        params.update(model_params)

        # 特徴量計算モデルのパラメータをoptunaで置き換え
        if tune_fs:
            params.update(OptunaParamsFactory.run("fs", trial))

        # 特徴量選択のthresholdをoptunaで置き換え
        if tune_th:
            params.update(OptunaParamsFactory.run("threshold", trial))

        # 実行
        task = Task(**params)
        luigi.build([task], workers=1, local_scheduler=True)
        output_path = task.output()
        with output_path.open("r") as f:
            output = json.load(f)
        return output["score"]
    return inner_objective


class TasksRunnerWithOptuna:
    """optunaでの最適化をサポートし、複数のタスクを回す."""

    def __init__(self, Task, direction, seed, token, app_name, channel, do_notify, do_plot, tune_fs, tune_th):
        self.seed = seed
        self.direction = direction
        self.Task = Task
        self.notifier = SlackApi(token, app_name, channel)
        self.do_notify = do_notify
        self.do_plot = do_plot
        self.tune_fs = tune_fs
        self.tune_th = tune_th

    def run(self, base_params, members, hp_tunes):
        """実行し、実行済みのtaskを返す.(output_pathを得るため)"""

        tasks = []
        for i, member_params in enumerate(members):
            print(
                f"[{i+1}/{len(members)}]")
            for k, v in member_params.items():
                print(f"{k} = {v}")

            params = base_params.copy()
            params.update(member_params)

            try:
                if hp_tunes:
                    # Search hyperparameters
                    study = optuna.create_study(
                        direction=self.direction, sampler=optuna.samplers.RandomSampler(seed=self.seed))

                    study.optimize(objective(params, self.Task, self.tune_fs, self.tune_th),
                                   n_trials=hp_tunes)

                    best_params = study.best_params

                    # get best params(実行する必要はなく、結果のpathを取得するために作成)
                    params.update(best_params)
                else:
                    # チューニングしない場合はここで実行する
                    luigi.build([self.Task(**params)],
                                workers=1, local_scheduler=True)
            except Exception as e:
                if self.do_notify:
                    self.notifier.send_message(f"Task{i+1} Failure. {e}")
                raise RuntimeError(f"Task{i+1} Failure. {e}")

            print(params)
            task = self.Task(**params)
            if self.do_notify or self.do_plot:
                self.success_log_to_slack(i, task)
            tasks.append(task)
        return tasks

    def success_log_to_slack(self, i, task):
        output_path = task.output()
        with output_path.open("r") as f:
            cv_output = json.load(f)

        self.notifier.send_message(
            f"Task{i+1} Success. Score: {cv_output['score']}")
        if self.do_plot:
            fs_dir = Path(output_path.path).parent.parent
            img_path = str(fs_dir/"importance.png")
            importance = pd.read_csv(
                str(fs_dir/"feature_importances.csv"), index_col=0)
            plot_feature_importance(importance, img_path)

            self.notifier.upload_file(img_path)
