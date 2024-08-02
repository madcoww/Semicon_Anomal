"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import torch.nn as nn
from transformers import AlbertModel

class CustomAlBert(nn.Module):
    def __init__(self, model_name, device, tokenizer):
        super(CustomAlBert, self).__init__()
        self.model_name = model_name
        self.device = device

        self.albert = AlbertModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.output_dim = len(self.tokenizer)

        self.albert.resize_token_embeddings(self.output_dim)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.albert.config.hidden_size, self.output_dim)
        # 모델을 지정된 장치로 이동
        self.to(self.device)

    def forward(self, input_ids, attention_mask=None):
        # 모델 입력을 장치로 이동
        input_ids = input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 모델 실행
        outputs = self.albert(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        last_hidden_state = self.dropout(last_hidden_state)

        logits = self.linear(last_hidden_state)

        return logits
