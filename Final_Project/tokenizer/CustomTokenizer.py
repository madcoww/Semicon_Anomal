"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
from utils.DataHandler import DataHandler
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from config.config import Config


class CustomTokenizer:
    def __init__(self, file_path, txt_path, vocab_size, limit_alphabet, min_frequency, my_special_tokens, add_token):
        self.config = Config()
        self.tokenizer = None
        self.vocab_txt = None
        self.file_path = file_path
        self.txt_path = txt_path
        self.vocab_size = vocab_size
        self.limit_alphabet = limit_alphabet
        self.min_frequency = min_frequency
        self.my_special_tokens = my_special_tokens
        self.add_token = add_token

    def data2txt(self):
        DataHandler.ext_payload(self.file_path, self.txt_path)

    def train_save(self):
        self.tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            strip_accents=False,
            lowercase=False,
            wordpieces_prefix="##")

        self.tokenizer.train(
            files=self.txt_path,
            vocab_size=self.vocab_size,
            limit_alphabet=self.limit_alphabet,
            min_frequency=self.min_frequency,
            special_tokens=self.my_special_tokens,
            show_progress=True,)

        saved_files = self.tokenizer.save_model('./')
        self.vocab_txt = [file for file in saved_files if 'vocab.txt' in file][0]

        self.tokenizer = BertTokenizer(vocab_file=self.vocab_txt, do_lower_case=False)
        self.tokenizer.add_tokens(self.add_token)

        # self.tokenizer.save_pretrained(self.config.SAVE_TOKENIZER_30_PATH)
        # self.tokenizer.save_pretrained(self.config.SAVE_TOKENIZER_32_PATH)
        self.tokenizer.save_pretrained("/SSD/ai_test/ctokenizer_test")


