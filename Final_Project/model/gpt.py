"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
import os
from openai import OpenAI

class Gpt:
    def __init__(self, model_name, prompt_template):
        self.model_name = model_name
        self.prompt_template = prompt_template

    def few_shot(self, train, val):
        client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)

        SYSTEM_PROMPT = {"role": "system", "content": self.prompt_template}

        few_examples = []
        for i in range(len(train)):
            user_example = {"role": "user", "content": f"Payload : {train['payload'][i]}"}
            assistant_example = {
                "role": "assistant",
                "content": (
                    f"공격 여부(0:비공격, 1:공격) : {train['label'][i]}\n"
                    f"공격 구문 : {train['attack_syntax'][i]}\n"
                    f"공격 유형 : {train['attack_type'][i]}\n"
                )
            }
            few_examples.extend([user_example, assistant_example])

        results = []
        total_token = 0

        # 각 검증 데이터에 대해 예측 수행
        for i in range(len(val)):
            current_instruction = {"role": "user", "content": f"Payload: {val['payload'][i]}"}

            messages = [
                SYSTEM_PROMPT,
                *few_examples,
                current_instruction
            ]

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                top_p=1,
                max_tokens=3000
            )

            if completion.choices:
                output = completion.choices[0].message.content
                results.append(output)
                total_token += completion.usage.total_tokens
            else:
                results.append("No response")

        return results, total_token

    def zero_shot(self, val):
        client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

        val = val.reset_index(drop=True)

        SYSTEM_PROMPT = {"role": "system", "content": self.prompt_template}

        results = []
        total_token = 0

    # 각 검증 데이터에 대해 예측 수행
        for i in range(len(val)):
            few_examples = []

            user_example = {"role": "user", "content": f"Payload : {val['payload'][i]}"}
            assistant_example = {
                "role": "assistant",
                "content": (
                    f"공격 여부(0:비공격, 1:공격) : \n"
                    f"공격 구문 : \n"
                    f"공격 유형 : \n"
                )
            }
            few_examples.extend([user_example, assistant_example])
            messages = [
                SYSTEM_PROMPT,
                *few_examples
            ]

            completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    top_p=1,
                    max_tokens=3000
                )

            if completion.choices:
                output = completion.choices[0].message.content
                results.append(output)
                total_token += completion.usage.total_tokens
            else:
                results.append("No response")

        return results, total_token