from lilac.ensemble.ensemble_runner_factory import EnsembleRunnerFactory


class StackingRunner:
    """1段目の出力を受け取り,stackingを実行する."""

    def __init__(self, stackings, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, num_class):
        """stackings : どのモデルを使って何層stackingするか."""
        self.stackings = stackings
        params = {"target_col": target_col, "folds_generator_flag":
                  folds_generator_flag, "folds_gen_params": folds_gen_params,
                  "trainer_flag":  trainer_flag,
                  "trainer_params": trainer_params,
                  "num_class": num_class}
        self.factory = EnsembleRunnerFactory(params)

    def run(self, output_list, train, test):
        if (len(self.stackings) == 0) and (len(output_list) > 1):
            raise Exception(
                f"1st layers output: {len(output_list)} but no stacking layers follows.")

        input_list = output_list
        for i, layer in enumerate(self.stackings):
            layer_results = []
            for ensemble_flag in layer:
                runner = self.factory.run(ensemble_flag)
                result = runner.run(input_list, train, test)
                layer_results.append(result)
            input_list = layer_results
            self.layer_logging(input_list, i+1, layer)
        if len(input_list) != 1:
            raise Exception("Stacking output should be single.")
        return input_list[0]

    def layer_logging(self, output_list, layer_num, layer):
        """layerごとに結果を出力する"""
        print(f"Layer {layer_num}")
        for i, output in enumerate(output_list):
            print(
                f"[{layer[i]}]: {output['evaluator']} = {output['score']}")
        print("=============================")
