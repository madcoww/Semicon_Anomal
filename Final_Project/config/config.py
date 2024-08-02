"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
class Config:
    """
    <Seed>
    """
    SEED = 42
    """
    <File Path>
    """
    FILE_PATH = "/SSD/ai_test/dataset_v2.6/dataset_v2.6_utf.csv"
    EX_PATH = "/SSD/ai_test/dataset_v2.6/ex_dataset_8.csv"
    CLUSTER_PATH = "/SSD/ai_test/result/cluster_dataset.csv"
    FINAL_LLAMA_FEW_PATH = "/SSD/ai_test/result/llama3_few_result.csv"
    FINAL_LLAMA_ZERO_PATH = "/SSD/ai_test/result/llama3_zero_result.csv"
    FINAL_GPT_FEW_PATH = "/SSD/ai_test/result/gpt_few_result.csv"
    FINAL_GPT_ZERO_PATH = "/SSD/ai_test/result/gpt_zero_result.csv"
    FINAL_JSON_PATH = "/SSD/ai_test/result/llama3_result.json"

    """
    <Model Path>
    """
    LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    GPT_MODEL = "gpt-3.5-turbo-0125"
    LLAMA_PATH = "/SSD/ai_test/Meta-Llama-3-8B-Instruct"

    """
    <Prompt Example>
    """
    LLAMA_FEW_TEMPLATE = (f"You are a helpful AI assistant. Based on the payload, if you determine it is not an attack, "
                     f"classify it as 0; if you determine it is an attack, classify it as 1; if you are unsure, classify it as 2. And if possible,"
                     f" please also specify the attack segment and attack type.")
    LLAMA_ZERO_TEMPLATE = "You are an assistant that distinguishes payload attacks."
    GPT_FEW_TEMPLATE = "너는 페이로드 공격을 구분하는 어시스턴트야\n"
    GPT_ZERO_TEMPLATE = "너는 페이로드 공격을 구분하는 어시스턴트야 페이로드를 보고 공격이 아니라고 판단되면  공격 여부를 0(비공격)으로, 공격이라고 판단되면 1(공격)로 출력해줘 그리고 공격 구문과 공격 유형, 이유에 대해서도 설명해줘"

    """
    <Fine-Tuning Parameter>
    """
    ALBERT_MODEL = "albert/albert-base-v2"
    ALBERT_MODEL_2 = "albert/albert-xlarge-v2"
    MAX_LENGTH = 512
    BATCH_SIZE = 32
    EPOCHS = [15]
    LEARNING_RATE = [3e-5]
    SAVE_MODEL_PATH = "/SSD/ai_test/cmodel"
    SAVE_TOKENIZER_30_PATH = "/SSD/ai_test/ctokenizer_30"
    SAVE_TEST_MODEL_PATH = "/SSD/ai_test/cmodel_test"
    SAVE_TOKENIZER_32_PATH = "/SSD/ai_test/ctokenizer_32"

    """
    <Tokenizer>
    * txt_path : 단어 집합을 얻기 위해 학습할 데이터
    * vocab_size : 단어 집합의 크기
    * limit_alphabet : 병합 전의 초기 토큰의 허용 개수.
    * min_frequency : 최소 해당 횟 수만큼 등장한 쌍(pair)의 경우에만 병합 대상이 된다.
    """
    TXT_PATH = "/SSD/ai_test/result/payload.txt"
    VOCAB_SIZE = 32000
    LIMIT_ALPHABET = 1000
    MIN_FREQUENCY = 0
    MY_SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + \
                        [f'[UNK{i}]' for i in range(10)] + \
                        [f'[unused{i}]' for i in range(50)]

    ADD_TOKEN = ['[Label : 1]', 'Label : 0]', '[Attack Syntax :]', '[Attack Type :]', '[nan]', '[SQL Injection]', '[Cross - Site Scripting ( XSS )]',
                 '[Path Traversal]', '[Remote Code Execution ( RCE )]', '[XML External Entity ( XXE ) Injection]', '[</]', '[//]', '[://]', '[../]',
                 '[.../]', '[...]', '[..]', '[/**/]', '[((]', '[))]', '[||]', '[<?]', '[<!]', '[\..]', '[< script >]', '[< / script >]']