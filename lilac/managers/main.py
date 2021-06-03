"""CLIエンドポイント"""
import argparse
import json
from .resource.initialize import initialize
from .resource.run_experiment import run_experiment

from lilac.utils.utils import MyEncoder


def command_new(args):
    initialize(args.settings_path,
               args.features_dir, args.output_dir, args.result_dir,
               args.data_dir, args.submission_dir)


def command_run(args):
    result, output_path = run_experiment(args.settings_path, args.key, args.trials, args.app_name,
                                         args.channel, args.notify, args.plot,
                                         args.tune_fs, args.tune_th, args.output_dir)

    with open(output_path, "w") as f:
        json.dump(result, f, cls=MyEncoder)


def main():
    parser = argparse.ArgumentParser(description='Lilac argument parser')
    subparsers = parser.add_subparsers()

    # new コマンドの parser を作成
    parser_new = subparsers.add_parser('new', help='see `new -h`')

    parser_new.add_argument(
        '--output-dir', default="result/outputs", help='output directory')
    parser_new.add_argument(
        '--result-dir', default="result/luigi", help='luigi directory')
    parser_new.add_argument(
        '--features-dir', default="result/features", help='features directory')
    parser_new.add_argument(
        '--data-dir', default="data", help='data directory')
    parser_new.add_argument(
        '--submission-dir', default="submissions", help='submission directory')
    parser_new.add_argument(
        '--settings-path', default='settings.json', help='setting file path')
    parser_new.set_defaults(handler=command_new)

    # commit コマンドの parser を作成
    parser_run = subparsers.add_parser('run', help='see `run -h`')
    parser_run.add_argument('key')
    parser_run.add_argument('-t', '--trials', type=int, default=None,
                            help="the number of trials of hyperpareter tuning")
    parser_run.add_argument('-n', '--notify', action='store_true',
                            help="notify result when finishing execution")
    parser_run.add_argument(
        '-p', '--plot', action='store_true', help="plot feature importance bar graph")
    parser_run.add_argument('-th', '--tune-th', action='store_true',
                            help="tune threshold of feature selection")
    parser_run.add_argument('-fs', '--tune-fs', action='store_true',
                            help="tune lgbm model parameters of feature importance")
    parser_run.add_argument(
        '--app-name', default='lilac-bot', help="slack notifier's name")
    parser_run.add_argument('--channel', default='notification-model-training',
                            help="channel name where notifier will use")
    parser_run.add_argument(
        '--output-dir', default="result/outputs", help='output directory')
    parser_run.add_argument(
        '--result-dir', default="result/luigi", help='luigi directory')
    parser_run.add_argument(
        '--features-dir', default="result/features", help='features directory')
    parser_run.add_argument(
        '--data-dir', default="data", help='data directory')
    parser_run.add_argument(
        '--submission-dir', default="submissions", help='submission directory')
    parser_run.add_argument(
        '--settings-path', default='settings.json', help='setting file path')

    parser_run.set_defaults(handler=command_run)

    # コマンドライン引数をパースして対応するハンドラ関数を実行
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        # 未知のサブコマンドの場合はヘルプを表示
        parser.print_help()
