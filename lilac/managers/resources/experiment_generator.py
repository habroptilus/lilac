from lilac.managers.resources.experiment import Experiment


class ExperimentGenerator:
    """とりあえずそのまま出力する."""

    def __init__(self, output_path):
        self.experiment = Experiment(
            "../templates/experiment.json", output_path)

    def run(self):
        self.experiment.run()
