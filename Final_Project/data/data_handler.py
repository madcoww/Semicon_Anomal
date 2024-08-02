"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import pandas as pd
import random

class DataHandler:
    def __init__(self, file_path, seed):
        self.file_path = file_path
        self.seed = seed

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            df = df.drop(columns=['site_code', 'CVE', 'create_date', 'info'])
            df = df[df['label'] != 2]
            df.loc[df['label'] == 0, 'code'] = 'Not Attack'
        except ValueError as e:
            print(f"파일 읽기 오류: {e}")
            return None
        return df

    def extract_by_path(self, ex_path, cluster_path):
        train = pd.read_csv(ex_path)
        val = pd.read_csv(cluster_path)
        val_filtered = val[~val['payload'].isin(train['payload'])]
        val_sampled = val_filtered.groupby('cluster').apply(
            lambda x: x if len(x) <= 50 else x.sample(50, random_state=self.seed)
        ).reset_index(drop=True)
        return train, val_sampled

    def all_by_path(self, ex_path, val):
        train = pd.read_csv(ex_path)
        # Test
        val = val.sample(n=60000, random_state=self.seed)

        val_filtered = val[~val['payload'].isin(train['payload'])]

        train = train.reset_index(drop=False)
        val_filtered = val_filtered.reset_index(drop=False)

        return train, val_filtered

    def extract_by_df(self, ex_path, val):
        train = pd.read_csv(ex_path)
        val_filtered = val[~val['payload'].isin(train['payload'])]
        val_sampled = val_filtered.groupby('cluster').apply(
        lambda x: x if len(x) <= 30 else x.sample(30, random_state=self.seed)
        ).reset_index(drop=True)
        return train, val_sampled


