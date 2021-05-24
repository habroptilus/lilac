import yaml


class Config:
    def __init__(self, source_path, output_path, update_dict=None):
        self.source_path = source_path
        self.update_dict = update_dict
        self.output_path = output_path

    def load_source(self):
        with open(self.source_path, "r") as f:
            self.loaded = yaml.load(f)

    def update(self):
        if self.update_dict:
            self.loaded.update(self.update_dict)

    def dump(self):
        with open(self.output_path, "w") as f:
            yaml.dump(self.loaded, f)

    def run(self):
        self.load_source()
        self.update()
        self.dump()
