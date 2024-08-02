"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
from utils.DataHandler import DataHandler
from dataloader.CustomDataset import CustomDataset
from model.CustomAlbert import CustomAlBert
from model.AlbertTrainer import Trainer
from model.AlbertTest import AlbertTest
from tokenizer.CustomTokenizer import CustomTokenizer
from transformers import BertTokenizer
import os

class AlBertMain:
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = self.config.MAX_LENGTH
        self.batch_size = self.config.BATCH_SIZE
        self.lr = self.config.LEARNING_RATE
        self.epochs = self.config.EPOCHS
        self.test_loader = None
        self.best_model_path = None

    def build_tokenizer(self):
        CTokenizer = CustomTokenizer(
            self.config.FINAL_JSON_PATH,
            self.config.TXT_PATH,
            self.config.VOCAB_SIZE,
            self.config.LIMIT_ALPHABET,
            self.config.MIN_FREQUENCY,
            self.config.MY_SPECIAL_TOKENS,
            self.config.ADD_TOKEN)

        CTokenizer.data2txt()
        CTokenizer.train_save()

    def load_tokenizer(self):
        # # Build Tokenizer
        # self.build_tokenizer()

        # # Custom Tokenizer 30000, 32000
        # self.tokenizer = BertTokenizer.from_pretrained(self.config.SAVE_TOKENIZER_30_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.SAVE_TOKENIZER_32_PATH)

        # # Test 50000
        # self.tokenizer = BertTokenizer.from_pretrained("/SSD/ai_test/ctokenizer_50")
        print("Tokenizer Size :", len(self.tokenizer))

    def load_model(self):
        if self.tokenizer is None:
            self.load_tokenizer()
        self.model = CustomAlBert(self.config.ALBERT_MODEL, self.device, self.tokenizer)

    def split_train_val(self):
        return DataHandler.split_dataset(self.config.FINAL_JSON_PATH)

    def create_datasets(self):
        train, test, val = self.split_train_val()

        if self.tokenizer is None:
            self.load_tokenizer()
        trainset = CustomDataset(train, self.tokenizer, self.max_length, self.device)
        testset = CustomDataset(test, self.tokenizer, self.max_length, self.device)
        valset = CustomDataset(val, self.tokenizer, self.max_length, self.device)

        print("Train Set Size: ", len(trainset))
        print("Val Set Size: ", len(valset))
        print("Test Set Size: ", len(testset))

        # Single GPU
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)

        self.test_loader = test_loader

        return train_loader, val_loader

    def run(self):
        train_loader, val_loader = self.create_datasets()

        best_model_state = None
        best_val_loss = float('inf')
        best_hyperparameters = {}

        for epochs in self.config.EPOCHS:
            for lr in self.config.LEARNING_RATE:
                print(f"Training with EPOCHS={epochs}, LEARNING RATE={lr}")
                self.load_model()

                # 옵티마이저 및 손실 함수 설정
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                # Trainer 인스턴스 생성
                trainer = Trainer(self.model, train_loader, val_loader, optimizer, criterion, self.device)

                # 모델 학습
                start_training = time.time()
                trainer.train(epochs=epochs)
                end_training = time.time()

                trian_time = end_training - start_training

                print(trian_time)
                # 검증 데이터셋으로 평가
                val_loss = trainer.validate()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    best_hyperparameters = {'epochs': epochs, 'learning_rate': lr}

                    # save_dir = self.config.SAVE_MODEL_PATH
                    save_dir = self.config.SAVE_TEST_MODEL_PATH

                    model_filename = f"al_base_AdamW_m2_v2_50_epoch{epochs}_lr{lr}.pt"
                    save_path = os.path.join(save_dir, model_filename)

                    # 경로가 존재하지 않으면 생성
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    self.best_model_path = save_path
                    torch.save(best_model_state, save_path)
                    print(f"Model saved at {save_path} with Validation Loss: {best_val_loss}")
                    print(f"Best hyperparameters: {best_hyperparameters}")

    def test(self):
        self.create_datasets()
        self.best_model_path = "/SSD/ai_test/cmodel_test/al_base_AdamW_m2_v2_50_epoch15_lr3e-05.pt"
        albertTest = AlbertTest(self.config.ALBERT_MODEL, self.best_model_path, self.test_loader, self.device)
        start_testing = time.time()
        albertTest.test()
        end_testing = time.time()

        test_time = end_testing - start_testing
        print(test_time)

if __name__ == "__main__":
    bert_main = AlBertMain()
    bert_main.run()
    # bert_main.test()
