from lilac.experiment.arg_parse import get_experiment_parser
from lilac.experiment.run_experiment import run_experiment
from lilac.utils.utils import MyEncoder
import json


def main():
    # 引数を受け取る
    parser = get_experiment_parser()
    args = parser.parse_args()

    # 実行
    result, output_path = run_experiment(args)

    with open(output_path, "w") as f:
        json.dump(result, f, cls=MyEncoder)


if __name__ == "__main__":
    main()
