"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
from config.config import Config
from data.data_handler import DataHandler
from data.clustering import Clustering
from model.gpt import Gpt
from eval.evaluation import Evaluation
import time

def few():

    # Config 설정
    config = Config()

    # 데이터 핸들링
    data_handler = DataHandler(config.FILE_PATH, config.SEED)

    """
    클러스터링 과정 생략(이전에 생성된 클러스터링 데이터 활용)
    """

    # 생성된 클러스터링 데이터 활용(EX_PATH)
    train, val = data_handler.extract_by_path(config.EX_PATH, config.CLUSTER_PATH)

    # gpt-3.5-turbo-0125
    gpt_model = Gpt(config.GPT_MODEL, config.GPT_FEW_TEMPLATE)

    start_time = time.time()

    results, total_token = gpt_model.few_shot(train, val)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Type : Few-Shot Learning")
    print("Times(s) : ", elapsed_time)
    print("Total Tokens : ", total_token)

    # 결과 평가 및 저장
    # Evaluation.list2txt(results)
    Evaluation.cal_confusion_matrix(results, val)
    Evaluation.result2csv(val, results, config.FINAL_GPT_FEW_PATH)

def zero():

    # Config 설정
    config = Config()

    # 데이터 핸들링
    data_handler = DataHandler(config.FILE_PATH, config.SEED)

    """
    클러스터링 과정 생략(이전에 생성된 클러스터링 데이터 활용)
    """

    # 생성된 클러스터링 데이터 활용(EX_PATH)
    train, val = data_handler.extract_by_path(config.EX_PATH, config.CLUSTER_PATH)

    # gpt-3.5-turbo-0125
    gpt_model = Gpt(config.GPT_MODEL, config.GPT_ZERO_TEMPLATE)

    start_time = time.time()

    results, total_token = gpt_model.zero_shot(val)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Type : Zero-Shot Learning")
    print("Times(s) : ", elapsed_time)
    print("Total Tokens : ", total_token)

    # 결과 평가 및 저장
    # Evaluation.list2txt(results)
    Evaluation.cal_confusion_matrix(results, val)
    Evaluation.result2csv(val, results, config.FINAL_GPT_ZERO_PATH)

if __name__ == "__main__":
    # few()
    zero()
