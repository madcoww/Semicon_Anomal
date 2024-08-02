"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import torch
from tqdm import tqdm
from peft import LoraConfig

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.initial_params = None
        self.final_params = None

    def train(self, epochs):
        self.initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        self.print_requires_grad(self.model)

        self.model.train()  # 모델을 학습 모드로 설정

        for epoch in range(epochs):
            total_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits_flat = logits.view(-1, logits.size(-1))
                target_ids_flat = target_ids.view(-1)

                loss = self.criterion(logits_flat, target_ids_flat)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {total_loss / len(self.train_loader)}")

        self.final_params = {name: param.clone() for name, param in self.model.named_parameters()}
        param_changes = {name: (self.initial_params[name] != self.final_params[name]).sum().item() for name in self.initial_params}
        # 파라미터 변화 출력
        print("\nParameter changes after training:")
        for name, change in param_changes.items():
            print(f"{name}: changed elements = {change}")

    def validate(self):
        self.model.eval()  # 모델을 평가 모드로 설정
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits_flat = logits.view(-1, logits.size(-1))
                target_ids_flat = target_ids.view(-1)

                loss = self.criterion(logits_flat, target_ids_flat)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)

        print(f"Validation Loss: {average_loss}")
        return average_loss

    # 파라미터 확인
    def print_requires_grad(self, model):
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")