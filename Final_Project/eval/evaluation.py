"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import pandas as pd
import re
from urllib.parse import unquote

class Evaluation:

    @staticmethod
    def clean_payload(results):
        pres = []
        for result in results:
            # 다양한 패턴을 한 번에 검색
            match = re.search(
                r'(?:Label\s*:\s*(\d+)|공격\s*여부\(0:비공격,\s*1:공격\)\s*:\s*(\d+)|공격\s*여부\s*[:는를]\s*(\d+)|Payload는\s*공격\((\d+)\))',
                result
            )

            if match:
                # 여러 그룹 중 하나에 값이 있을 수 있으므로, 첫 번째로 찾은 값을 사용
                label_number = match.group(1) or match.group(2) or match.group(3) or match.group(4)
                pres.append(int(label_number))
            else:
                try:
                    pres.append(int(result))  # 정수로 변환
                except ValueError:
                    pres.append(-1)  # 변환 실패 시 -1 반환

        pred_df = pd.DataFrame({
            'pred': pres,
            'result': results
        })
        return pred_df

    @staticmethod
    def extract_result(results):
        labels = []
        attack_strings = []
        attack_types = []

        for result in results:
            m_label = re.search(r'Label\s*:\s*(\d+)', result)
            m_string = re.search(r'Attack Syntax\s*:\s*(.+?)(?=\n|$)', result)
            m_type = re.search(r'Attack Type\s*:\s*(.+?)(?=\n|$)', result)

            label = int(m_label.group(1)) if m_label else None
            # Decoding
            attack_string = unquote(m_string.group(1)) if m_string else None

            # attack_string = m_string.group(1) if m_string else None
            attack_type = m_type.group(1) if m_type else None

            labels.append(label)
            attack_strings.append(attack_string)
            attack_types.append(attack_type)

        pred_df = pd.DataFrame({
            'pred': labels,
            'attack_string': attack_strings,
            'attack_type': attack_types,
            'answer': results
        })
        return pred_df


    @staticmethod
    def cal_confusion_matrix(result, val):
        pred_df = Evaluation.clean_payload(result)
        pred_label = pred_df['pred'].tolist()
        test_label = [int(label) for label in val["label"]]
        valid_classes = {0, 1, 2}
        matrix = [[0 for _ in valid_classes] for _ in valid_classes]
        for pred, actual in zip(pred_label, test_label):
            if pred in valid_classes:
                matrix[pred][actual] += 1
        print("\t\t\t\tActual")
        print("\t\t", end="")
        print("\t\t0\t1\t2")
        for i in valid_classes:
            print(f"Predicted\t{i}\t", end="")
            for j in valid_classes:
                print(f"{matrix[i][j]}\t", end="")
            print()

    @staticmethod
    def list2txt(results, txt_path="/SSD/ai_test/result/model_result.txt"):
        with open(txt_path, "w") as file:
            for result in results:
                file.write(result + "\n")
                file.write("-" * 200 + "\n")

    @staticmethod
    def result2csv(valset, results, csv_path):
        pred_df = Evaluation.extract_result(results)

        valset['pred'] = pred_df['pred'].tolist()
        valset['correct'] = valset['label'] == valset['pred']
        valset['attack_string'] = pred_df['attack_string'].tolist()
        valset['attack_type'] = pred_df['attack_type'].tolist()
        valset['answer'] = pred_df['answer'].tolist()

        valset_filtered = valset[valset['correct']]

        valset_filtered.to_csv(csv_path, index=False)

    @staticmethod
    def result2json(valset, results, json_path):
        pred_df = Evaluation.extract_result(results)

        if 'Unnamed: 0' in valset.columns:
            valset = valset.rename(columns={'Unnamed: 0': 'index'})

        valset['pred'] = pred_df['pred'].tolist()
        valset['correct'] = valset['label'] == valset['pred']
        valset['attack_string'] = pred_df['attack_string'].tolist()
        valset['attack_type'] = pred_df['attack_type'].tolist()
        valset['answer'] = pred_df['answer'].tolist()

        valset_filtered = valset[valset['correct']]

        valset_json = valset_filtered.to_json(orient='records', force_ascii=False, lines=True)

        # Write the JSON output to a file
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(valset_json)
