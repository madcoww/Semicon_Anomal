"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
from transformers import BertTokenizer
from model.CustomAlbert import CustomAlBert
from tqdm import tqdm
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

class AlbertTest:
    def __init__(self, model_name, best_model_path, test_loader, device):
        self.tokenizer = BertTokenizer.from_pretrained('/SSD/ai_test/ctokenizer_32')
        self.model = CustomAlBert(model_name, device, self.tokenizer)
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def generate_text(self, input_ids, attention_mask, max_length=512):
        decoded_sentence = []

        with torch.no_grad():
            logits = self.model.forward(input_ids, attention_mask)  # [batch_size, sequence_length, vocab_size]

            for i in range(min(max_length, logits.size(1))):
                # i번째 시퀀스 위치의 logits에 softmax 적용
                probs = F.softmax(logits[:, i, :], dim=-1)  # [batch_size, vocab_size]

                # 각 배치의 시퀀스 위치에서 가장 높은 확률을 가진 토큰 인덱스 선택
                predicted_tokens = torch.argmax(probs, dim=1)  # [batch_size]
                # predicted_tokens = torch.multinomial(probs, 1)

                for token_id in predicted_tokens:
                    if token_id.item() in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        continue
                    decoded_token = self.tokenizer.convert_ids_to_tokens(token_id.item())
                    decoded_sentence.append(decoded_token)

        return ' '.join(decoded_sentence)

    def test(self):
        self.model.eval()
        input_texts = []
        candidate_texts = []
        reference_texts = []
        bleu_scores = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                candidate_text = self.generate_text(input_ids, attention_mask, 512)

                reference_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]
                reference_text = ' '.join(reference_text)

                input_text = [self.tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
                input_text = ' '.join(input_text)

                reference_tokens = reference_text.split()
                candidate_tokens = candidate_text.split()

                smoothing_function = SmoothingFunction().method1
                bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)

                print(f"\n\nCandidate Text : {candidate_text}\nReference Text : {reference_text}\nBLEU Score : {bleu_score}")

                input_texts.append(input_text)
                candidate_texts.append(candidate_text)
                reference_texts.append(reference_text)
                bleu_scores.append(bleu_score)

        average_bleu = sum(bleu_scores) / len(bleu_scores)
        print(f"\n\nAverage BLEU Score: {average_bleu}")

        # df = pd.DataFrame({
        #     'payload': input_texts,
        #     'reference_text': reference_texts,
        #     'candidate_text': candidate_texts,
        #     'bleu_score': bleu_scores
        #
        # })
        # df.to_csv("/SSD/ai_test/result/adamW_m2_v0_32_result.csv", index=False)
