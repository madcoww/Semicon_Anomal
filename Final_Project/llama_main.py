"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
from config.config import Config
from data.data_handler import DataHandler
from data.clustering import Clustering
from model.llama import Llama3
from eval.evaluation import Evaluation
import huggingface_hub
import time

def few():

    # Config 설정
    config = Config()

    # 데이터 핸들링
    data_handler = DataHandler(config.FILE_PATH, config.SEED)
    df = data_handler.load_data()
    if df is None:
        return
    # # 클러스터링
    # clustering = Clustering(df, config.CLUSTER_PATH, config.SEED)
    # X_sample, tfidf = clustering.vectorize(clustering.sample_data(df))
    # X_reduced_sample, svd = clustering.reduce_dimensions(X_sample)
    # best_kmeans, best_n_clusters, best_silhouette = clustering.find_best_kmeans(X_reduced_sample)
    # df_clustered = clustering.apply_clustering(df, tfidf, svd, best_kmeans)
    # clustering.to_csv(df_clustered)

    # train, val = data_handler.extract_by_df(config.EX_PATH, df_clustered)



    # 생성된 클러스터링 데이터 활용
    # train, val = data_handler.extract_by_path(config.EX_PATH, config.CLUSTER_PATH)
    train, val = data_handler.all_by_path(config.EX_PATH, df)

    # LLama-3-8B
    llama_model = Llama3(config.LLAMA_PATH, config.LLAMA_FEW_TEMPLATE)
    llama_model.load_model()

    start_time = time.time()

    results = llama_model.few_shot(train, val)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Type : Few-Shot Learning")
    print("Times(s) : ", elapsed_time)

    # 결과 평가 및 저장
    # Evaluation.list2txt(results)
    Evaluation.cal_confusion_matrix(results, val)
    Evaluation.result2csv(val, results, config.FINAL_LLAMA_FEW_PATH)
    Evaluation.result2json(val, results, config.FINAL_JSON_PATH)


def zero():

    # Config 설정
    config = Config()

    # 데이터 핸들링
    data_handler = DataHandler(config.FILE_PATH, config.SEED)

    """
    클러스터링 과정 생략(이전에 생성된 클러스터링 데이터 활용)
    """

    # 생성된 클러스터링 데이터 활용
    train, val = data_handler.extract_by_path(config.EX_PATH, config.CLUSTER_PATH)

    # LLama-3-8B
    llama_model = Llama3(config.LLAMA_MODEL, config.LLAMA_ZERO_TEMPLATE)
    llama_model.load_model()

    start_time = time.time()

    results = llama_model.zero_shot(val)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Type : Zero-Shot Learning")
    print("Times(s) : ", elapsed_time)

    # 결과 평가 및 저장
    # Evaluation.list2txt(results)
    Evaluation.cal_confusion_matrix(results, val)
    Evaluation.result2csv(val, results, config.FINAL_LLAMA_ZERO_PATH)

if __name__ == "__main__":
    few()
    # zero()
