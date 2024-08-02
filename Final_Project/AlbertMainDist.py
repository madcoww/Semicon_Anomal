"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
from utils.DataHandler import DataHandler
from dataloader.CustomDataset import CustomDataset
from model.CustomAlbert import CustomAlBert
from model.AlbertTrainerDist import Trainer
from model.AlbertTest import AlbertTest
from transformers import BertTokenizer
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


class AlBertMain:
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = self.config.MAX_LENGTH
        self.batch_size = self.config.BATCH_SIZE
        self.lr = self.config.LEARNING_RATE
        self.epochs = self.config.EPOCHS
        self.test_loader = None
        self.best_model_path = None
        self.rank = 0
        self.world_size = None

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.config.SAVE_TOKENIZER_32_PATH)
        print("Tokenizer Size :", len(self.tokenizer))

    def load_model(self):
        if self.tokenizer is None:
            self.load_tokenizer()
        self.model = CustomAlBert(self.config.ALBERT_MODEL_2, self.device, self.tokenizer)

    def split_train_val(self):
        return DataHandler.split_dataset(self.config.FINAL_JSON_PATH)

    def create_datasets(self, rank, world_size):
        train, test, val = self.split_train_val()

        if self.tokenizer is None:
            self.load_tokenizer()
        trainset = CustomDataset(train, self.tokenizer, self.max_length, self.device)
        testset = CustomDataset(test, self.tokenizer, self.max_length, self.device)
        valset = CustomDataset(val, self.tokenizer, self.max_length, self.device)

        print(f"Rank {rank} - Train Set Size: ", len(trainset))
        print(f"Rank {rank} - Val Set Size: ", len(valset))
        print(f"Rank {rank} - Test Set Size: ", len(testset))

        # Distributed
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank)

        train_loader = DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(valset, batch_size=self.batch_size, sampler=val_sampler)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)

        self.test_loader = test_loader

        return train_loader, val_loader

    def run(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        setup(rank, world_size)
        train_loader, val_loader = self.create_datasets(self.rank, self.world_size)

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
                trainer = Trainer(self.model, train_loader, val_loader, optimizer, criterion, self.device, rank)

                # 모델 학습
                start_training = time.time()
                trainer.train(epochs=epochs)
                end_training = time.time()

                train_time = end_training - start_training

                print(train_time)
                # 검증 데이터셋으로 평가
                val_loss = trainer.validate()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    best_hyperparameters = {'epochs': epochs, 'learning_rate': lr}

                    save_dir = self.config.SAVE_TEST_MODEL_PATH

                    model_filename = f"al_xLarge_AdamW_m2_v2_50_epoch{epochs}_lr{lr}.pt"
                    save_path = os.path.join(save_dir, model_filename)

                    # 경로가 존재하지 않으면 생성
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    self.best_model_path = save_path
                    torch.save(best_model_state, save_path)
                    print(f"Model saved at {save_path} with Validation Loss: {best_val_loss}")
                    print(f"Best hyperparameters: {best_hyperparameters}")

        cleanup()

    def test(self):
        self.create_datasets(self.rank, self.world_size)
        self.best_model_path = ""
        albertTest = AlbertTest(self.config.ALBERT_MODEL, self.best_model_path, self.test_loader, self.device)
        start_testing = time.time()
        albertTest.test()
        end_testing = time.time()

        test_time = end_testing - start_testing
        print(test_time)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.1.34.181'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    bert_main = AlBertMain()
    bert_main.run(rank, world_size)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

    # bert_main = AlBertMain()
    # bert_main.test()
