import argparse


def get_init_parser():
    parser = argparse.ArgumentParser(
        description='Competition Template.')

    parser.add_argument('--output-dir', default="result/outputs")
    parser.add_argument('--result-dir', default="result/luigi")
    parser.add_argument('--features-dir', default="result/features")
    parser.add_argument('--data-dir', default="data")
    parser.add_argument('--submission-dir', default="submissions")
    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--experiment-path', default='experiment.json')
    return parser
