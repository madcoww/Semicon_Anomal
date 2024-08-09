"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import unicodedata
import re
import torch

def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def preprocess_text(text, remove_space=True, lower=False):
    if remove_space:
        text = " ".join(text.strip().split())
    if lower:
        text = text.lower()
    return text

def whitespace_tokenize(text):
    return text.strip().split()

class WordPieceTokenizer:
    def __init__(self, vocab, max_input_chars_per_word=200):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_tokens = [f"[UNK{i}]" for i in range(50)]  # List of UNK tokens
        self.max_input_chars_per_word = max_input_chars_per_word
        self.unk_token_index = 0  # Index for tracking the current UNK token

        # For mapping UNK tokens to their original text chunks
        self.unk_mapping = {}

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = preprocess_text(text, remove_space=True, lower=False)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # Map the UNK token to its original text chunk
                unk_token = self.unk_tokens[self.unk_token_index]
                self.unk_mapping[unk_token] = token
                output_tokens.append(unk_token)
                self.unk_token_index = (self.unk_token_index + 1) % len(self.unk_tokens)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                unk_token = self.unk_tokens[self.unk_token_index]
                self.unk_mapping[unk_token] = token
                output_tokens.append(unk_token)
                self.unk_token_index = (self.unk_token_index + 1) % len(self.unk_tokens)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens

    def encode_plus(self, text, max_length=None, padding='max_length', truncation=True, return_tensors=None):
        tokens = self.tokenize(text)

        if max_length is not None and truncation and len(tokens) + 2 > max_length:
            tokens = tokens[:max_length - 2]

        if padding == 'max_length' and max_length is not None:
            padding_length = max_length - (len(tokens) + 2)
            if padding_length < 0:
                padding_length = 0
        else:
            padding_length = 0

        if padding == 'max_length':
            tokens = [self.cls_token] + tokens + [self.sep_token] + [self.pad_token] * padding_length
        else:
            tokens = [self.cls_token] + tokens + [self.sep_token]

        token_ids = [self.vocab.get(token, self.vocab.get(self.unk_tokens[0])) for token in tokens]
        attention_mask = [1 if token != self.pad_token else 0 for token in tokens]

        if return_tensors == 'pt':
            token_ids = torch.tensor(token_ids).unsqueeze(0)  # Batch size of 1
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # Batch size of 1

        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }

    def decode(self, token_ids, skip_special_tokens=False):
        # Convert token_ids to tokens using inverse vocab
        tokens = [self.inv_vocab.get(token_id, self.unk_tokens[0]) for token_id in token_ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {self.cls_token, self.sep_token, self.pad_token}]

        # Replace each UNK token with its original text chunk
        tokens = [self.unk_mapping.get(token, token) if token in self.unk_tokens else token for token in tokens]

        text = ' '.join(tokens)
        text = text.replace("##", "")
        text = text.strip()

        return text

    def add_tokens(self, new_tokens):
        current_size = len(self.vocab)

        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.vocab) - 1] = token

        return len(self.vocab) - current_size

# Example usage:
if __name__ == "__main__":

    ADD_TOKEN = ['[Label : 1]', 'Label : 0]', '[Attack Syntax :]', '[Attack Type :]', '[nan]', '[SQL Injection]', '[Cross - Site Scripting ( XSS )]',
                 '[Path Traversal]', '[Remote Code Execution ( RCE )]', '[XML External Entity ( XXE ) Injection]', '[</]', '[//]', '[://]', '[../]',
                 '[.../]', '[...]', '[..]', '[/**/]', '[((]', '[))]', '[||]', '[<?]', '[<!]', '[\..]', '[< script >]', '[< / script >]']

    vocab_file_path = "/SSD/ai_test/vocab.txt"
    vocab = {}
    with open(vocab_file_path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            token = line.strip()
            vocab[token] = index

    tokenizer = WordPieceTokenizer(vocab=vocab)

    text = f'''http : / / - ip - : - port - / dvwa / instructions. php / dvwa / instructions. php? doc = 1 ) ) ) / * * / AND / * * / 7902 = ( SELECT / * * / UPPER ( XMLType ( CHR ( 60 ) | | CHR ( 58 ) | | CHR ( 113 ) | | CHR ( 122 ) | | CHR ( 106 ) | | CHR ( 106 ) | | CHR ( 113 ) | | ( SELECT / * * / ( CASE / * * / WHEN / * * / ( 7902 = 7902 ) / * * / THEN / * * / 1 / * * / ELSE / * * / 0 / * * / END ) / * * / FROM / * * / DUAL ) | | CHR ( 113 ) | | CHR ( 107 ) | | CHR ( 120 ) | | CHR ( 98 ) | | CHR ( 113 ) | | CHR ( 62 ) ) ) / * * / FROM / * * / DUAL ) / * * / AND / * * / ( ( ( 4741 = 4741 sqlmap / 1. 5. 1. 40 # dev ( http : / / sqlmap. org ) PHPSESSID = jlpd56fppfes3q8njmedlrscgq ; security = impossible,'''

    encoding = tokenizer.encode_plus(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    decoded_text = tokenizer.decode(encoding['input_ids'].squeeze().tolist(), skip_special_tokens=True)

    print("Tokens:", encoding["tokens"])
    print("Input_ids", encoding['input_ids'])
    print("Decoded text:", decoded_text)
    print("Origin Text: ", text)