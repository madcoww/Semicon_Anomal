"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DistributedDataParallelKwargs

class Llama3:
    def __init__(self, model_name, system_prompt):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tokenizer = None
        self.model = None
        self.accelerator = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            # device_map='auto',
            # quantization_config=bnb_config,
        )

        # # 분산(Accelerator)
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        # self.model = self.accelerator.prepare(self.model)

    def few_shot(self, trainset, valset):

        torch.cuda.empty_cache()

        bos, eos = "<|begin_of_text|>", "<|end_of_text|>\n"
        shi, ehi = "<|start_header_id|>", "<|end_header_id|>\n"
        ei = "<|eot_id|>\n"

        PROMPT = bos + shi + "system" + ehi + self.system_prompt + ei

        result = []
        for i in range(len(trainset)):
            PROMPT += (
                    shi + "user" + ehi + f"Payload: {trainset['payload'][i]}" + ei +
                    shi + "assistant" + ehi +
                    f"Label: {trainset['label'][i]}\nAttack Syntax: {trainset['attack_syntax'][i]}\nAttack Type: {trainset['attack_type'][i]}" + ei
            )

        for i in range(len(valset)):
            current_instruction = (
                    shi + "user" + ehi + f"Payload: {valset['payload'][i]}" + ei +
                    shi + "assistant" + ehi
            )

            message = PROMPT + current_instruction
            tokenized_inputs = self.tokenizer(
                message,
                return_tensors="pt",
                padding=True,
                truncation=True)

            input_ids = tokenized_inputs["input_ids"].to(self.model.device)
            attention_mask = tokenized_inputs["attention_mask"].to(self.model.device)

            outputs = self.model.generate(input_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=1024,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          do_sample=True,
                                          temperature=0.7,
                                          top_p=1)

            answer = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            result.append(answer.strip())
        torch.cuda.empty_cache()
        return result

    def zero_shot(self, valset):

        torch.cuda.empty_cache()

        bos, eos = "<|begin_of_text|>", "<|end_of_text|>\n"
        shi, ehi = "<|start_header_id|>", "<|end_header_id|>\n"
        ei = "<|eot_id|>\n"

        PROMPT = bos + shi + "system" + ehi + self.system_prompt + ei

        result = []
        for i in range(len(valset)):
            PROMPT += (
                    shi + "user" + ehi + f"Payload: {valset['payload'][i]}\nLabel(0: non-attack, 1: attack): \nAttack Syntax: \nAttack Type:" + ei +
                    shi + "assistant" + ehi
            )

            message = PROMPT

            tokenized_inputs = self.tokenizer(
                message,
                return_tensors="pt",
                padding=True,
                truncation=True)

            input_ids = tokenized_inputs["input_ids"].to(self.model.device)
            attention_mask = tokenized_inputs["attention_mask"].to(self.model.device)

            outputs = self.model.generate(input_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=4096,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          do_sample=True,
                                          temperature=0.7,
                                          top_p=1)

            answer = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            result.append(answer.strip())
        torch.cuda.empty_cache()
        return result
