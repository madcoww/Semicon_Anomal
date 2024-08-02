"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, file_path, seed):
        self.file_path = file_path
        self.seed = seed

    def split_dataset(self, json_path, test_size=0.2):
        df = pd.read_json(json_path, lines=True)

        train, val = train_test_split(df, test_size=test_size, random_state=self.seed)

        return train, val