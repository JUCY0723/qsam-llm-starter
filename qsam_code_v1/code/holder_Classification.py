import dashscope
from dashscope import Generation
import pandas as pd
import time
import re
from tqdm import tqdm

dashscope.api_key = "sk-f39ab33acbc44d3ba8cdea5aac2e1d7a"

# 1. few-shot例子（可自行增删）
classic_shots = [
    ("The newspaper strongly opposes this policy.", "The newspaper", 1),
    ("Xinhua criticized the plan.", "Xinhua", 1),
    ("The newspaper published the statistics on Friday.", "The newspaper", 2),
    ("The policy was announced on Monday.", "", 2),
    ("According to experts, the proposal will fail.", "experts", 3),
    ("The mayor said that the situation is under control.", "mayor", 3)
]

fewshot_examples = classic_shots

# 2. Prompt构造
def make_prompt(sentence, holder, fewshot_examples):
    base_prompt = """
You are an expert information extractor. Please classify each sentence into one of the following three types, and ONLY output the category number (1/2/3). Do not provide any explanation.

1. Media commentary sentence: The media itself expresses subjective opinion or judgment.
2. Factual statement sentence: Purely reports objective facts or background data.
3. Third-party quotation sentence: Cites opinions of experts, officials, the public, etc.

Classification examples:
"""
    example_text = ""
    for s, h, label in fewshot_examples:
        example_text += f'Sentence: "{s}"\nHolder: "{h}"\nCategory: {label}\n\n'
    note_text = """Note: Classify as 1 ONLY if the media itself expresses an explicit subjective opinion or judgment. Otherwise, do not classify as 1.
"""
    prompt = f"{base_prompt}{example_text}{note_text}Sentence: \"{sentence}\"\nHolder: \"{holder}\"\nCategory:"
    return prompt

# 3. API调用函数
def call_qwen(sentence, holder, fewshot_examples, model_name="qwen-max", max_retry=3):
    prompt = make_prompt(sentence, holder, fewshot_examples)
    for attempt in range(max_retry):
        try:
            response = Generation.call(
                model=model_name,
                prompt=prompt,
                temperature=0.1,
            )
            if (response is None) or (not hasattr(response, "output")) or (not hasattr(response.output, "text")):
                print(f"API returned None | Sentence: {sentence}")
                return None
            result = response.output.text.strip()
            match = re.search(r'[1-3]', result)
            if match:
                return int(match.group())
            else:
                print(f"Could not extract class | Sentence: {sentence} | Output: {result}")
                return None
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(2)
            else:
                print(f"Error: {e} | Sentence: {sentence}")
                return None

# 4. 主流程，适配你的csv
if __name__ == "__main__":
    df = pd.read_csv('holder_target_test_with_qwen_pred1.csv')
    df = df.fillna("")
    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Qwen 3-way classification with holder"):
        sentence = str(row['sentence'])
        holder = str(row['holder_pred'])
        cls = call_qwen(sentence, holder, fewshot_examples, model_name="qwen-max")
        results.append(cls)
        time.sleep(0.8)  # 不建议小于0.8，否则API可能限流
    df['qwen_class'] = results
    df.to_csv('holder_target_test_with_qwen_pred1_with_type.csv', index=False)
    print("All done, saved to holder_target_test_with_qwen_pred1_with_type.csv")
