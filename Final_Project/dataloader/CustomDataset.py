"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=None, device=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        payload = self.dataframe.iloc[idx]['payload']
        answer = self.dataframe.iloc[idx]['answer']

        inputs = self.tokenizer.encode_plus(
            payload,
            None,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer.encode_plus(
            answer,
            None,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'target_ids': targets['input_ids'].flatten(),
            'target_mask': targets['attention_mask'].flatten()
        }
