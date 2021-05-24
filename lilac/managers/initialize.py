
from pathlib import Path

from lilac.managers.arg_parse import get_init_parser
from lilac.managers.resources.config import Config
from lilac.managers.resources.experiment import Experiment


def initialize(experiment_path, config_path, features_dir, output_dir, result_dir,
               data_dir, submission_dir):

    dir_list = [features_dir,
                output_dir, result_dir, data_dir, submission_dir]
    path_list = [Path(d) for d in dir_list]
    features_gen_dir = Path("features/generators")
    path_list.append(features_gen_dir)

    for p in path_list:
        p.mkdir(parents=True)

    with (features_gen_dir.parent/"__init__.py").open('w') as f:
        f.write("additional_features = {}\n")

    current_dir = Path(__file__).resolve().parent
    Config(
        f"{current_dir}/templates/config.yaml", config_path).run()

    Experiment(
        f"{current_dir}/templates/experiment.json", experiment_path).run()

    with Path(".gitignore").open("a") as f:
        f.write("""catboost_info
data
result
submissions
        """)


def main():
    parser = get_init_parser()
    args = parser.parse_args()

    initialize(args.experiment_path, args.config_path,
               args.features_dir, args.output_dir, args.result_dir,
               args.data_dir, args.submission_dir)


if __name__ == "__main__":
    main()
