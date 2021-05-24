class BasicTrainer:
    """普通の学習を行う."""

    def run(self, train, valid, model_factory):
        # 作成
        model = model_factory.run()
        # 学習
        model.fit(train, valid)

        return {"model": model}

    def return_flag(self):
        return "basic"
