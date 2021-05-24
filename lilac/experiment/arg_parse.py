from lilac.managers.arg_parse import get_init_parser


def get_experiment_parser():
    parser = get_init_parser()
    parser.add_argument('key')
    parser.add_argument('-t', '--trials', type=int, default=None)
    parser.add_argument('-n', '--notify', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-th', '--tune-th', action='store_true')
    parser.add_argument('-fs', '--tune-fs', action='store_true')
    parser.add_argument('--app-name', default='lilac-bot')
    parser.add_argument('--channel', default='notification-model-training')

    return parser
