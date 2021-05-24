import numpy as np
import torch

from transformers import BertTokenizer, BertModel
from .text_vectorizer_base import TextVectorizerBase


class BertVectorizer(TextVectorizerBase):
    """Bertの学習済みモデルを使って768次元のベクトルを出力する."""

    def __init__(self, col_prefix, seed, max_len):
        super().__init__(col_prefix, seed)
        self.max_len = max_len

    def fit(self, docs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(
            self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        return self

    def _transform(self, docs):
        result = []
        for doc in docs:
            inp = self.tokenizer.encode(doc)
            len_inp = len(inp)

            if len_inp >= self.max_len:
                inputs = inp[:self.max_len]
                masks = [1] * self.max_len
            else:
                inputs = inp + [0] * (self.max_len - len_inp)
                masks = [1] * len_inp + [0] * (self.max_len - len_inp)

            inputs_tensor = torch.tensor(
                [inputs], dtype=torch.long).to(self.device)
            masks_tensor = torch.tensor(
                [masks], dtype=torch.long).to(self.device)

            outputs = self.bert_model(inputs_tensor, masks_tensor)
            seq_out = outputs.last_hidden_state

            if torch.cuda.is_available():
                # 0番目は [CLS] token, 768 dim の文章特徴量
                vector = seq_out[0][0].cpu().detach().numpy()
            else:
                vector = seq_out[0][0].detach().numpy()
            result.append(vector)
        return np.array(result)
