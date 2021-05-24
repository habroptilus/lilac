from .basic_trainer import BasicTrainer
from .down_sampling_bagging_trainer import DownSamplingBaggingTrainer


class TrainerFactory:
    def __init__(self, trainer_str, params):
        self.params = params
        if trainer_str == "basic":
            self.required_params = []
            self.Trainer = BasicTrainer
        elif trainer_str == "down_sample_bagging":
            self.required_params = [
                "target_col", "bagging_num", "base_class", "seed", "allow_less_than_base"]
            self.Trainer = DownSamplingBaggingTrainer
        else:
            raise Exception(f"Invalid trainer flag. {trainer_str}")

    def run(self):
        params = {e: self.params[e]
                  for e in self.required_params}  # 必要なものだけ取り出す
        return self.Trainer(**params)
