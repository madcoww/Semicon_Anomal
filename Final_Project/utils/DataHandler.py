"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import pandas as pd
from urllib.parse import unquote


class DataHandler:

    @staticmethod
    def split_dataset(json_path):
        df = pd.read_json(json_path, lines=True)

        df = df.dropna(subset=['payload', 'attack_string', 'attack_type', 'answer'])

        value_counts = df['attack_type'].value_counts()

        single_counts = value_counts[value_counts == 1].index.tolist()

        filtered_df = df[~df['attack_type'].isin(single_counts)]

        filtered_df = filtered_df[
            ~((filtered_df['label'] == 1) & (filtered_df['attack_type'].isin(['None', 'nan'])))
        ]
        filtered_df = filtered_df[
            ~((filtered_df['label'] == 1) & (filtered_df['attack_string'].isin(['None', 'nan'])))
        ]
        filtered_df = filtered_df[
            ~((filtered_df['label'] == 0) & (filtered_df['attack_type'] != 'nan'))
        ]

        filtered_df['payload'] = filtered_df['payload'].apply(unquote)
        filtered_df['answer'] = filtered_df['answer'].apply(unquote)

        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))

        train_df = filtered_df[:train_size]
        test_df = filtered_df[train_size+val_size:]
        val_df = filtered_df[train_size:train_size+val_size]

        return train_df, test_df, val_df

    @staticmethod
    def ext_payload(file_path, txt_path):
        df = pd.read_json(file_path, lines=True)

        df = df.dropna(subset=['payload', 'attack_string', 'attack_type', 'answer'])

        value_counts = df['attack_type'].value_counts()

        single_counts = value_counts[value_counts == 1].index.tolist()

        filtered_df = df[~df['attack_type'].isin(single_counts)]

        filtered_df = filtered_df[
            ~((filtered_df['label'] == 1) & (filtered_df['attack_type'].isin(['None', 'nan'])))
        ]
        filtered_df = filtered_df[
            ~((filtered_df['label'] == 1) & (filtered_df['attack_string'].isin(['None', 'nan'])))
        ]
        filtered_df = filtered_df[
            ~((filtered_df['label'] == 0) & (filtered_df['attack_type'] != 'nan'))
        ]

        filtered_df['payload'] = filtered_df['payload'].apply(unquote)
        filtered_df['answer'] = filtered_df['answer'].apply(unquote)

        # if 'answer' in filtered_df.columns:
        #     unique_answers = filtered_df['answer'].drop_duplicates()
        #
        #     with open(txt_path, 'w', encoding='utf-8') as f:
        #         for answer in unique_answers:
        #             f.write(f"{answer}\n")

        if 'payload' in filtered_df.columns and 'answer' in filtered_df.columns:
            payloads = filtered_df['payload']
            answers = filtered_df['answer']

            with open(txt_path, 'w', encoding='utf-8') as f:
                for payload, answer in zip(payloads, answers):
                    f.write(f"{payload}\t{answer}\n")
