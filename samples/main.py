"""main.py"""
from lilac.experiment.arg_parse import get_experiment_parser
from lilac.experiment.run_experiment import run_experiment


def create_sub(pred, output_path):
    raise Exception("Implement create_sub method.")


if __name__ == "__main__":
    # 引数を受け取る
    parser = get_experiment_parser()
    args = parser.parse_args()

    # 実行
    result, output_path = run_experiment(args)

    # 提出ファイル作成
    create_sub(result["pred"], output_path)
