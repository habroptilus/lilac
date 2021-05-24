from lilac.managers.resources.config import Config


class ConfigGenerator:
    """とりあえずそのまま出力する."""

    def __init__(self, output_path):
        self.config = Config(
            "../templates/config.yaml", output_path)

    def run(self):
        self.config.run()
