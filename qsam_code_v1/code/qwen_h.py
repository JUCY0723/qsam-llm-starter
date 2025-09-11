import pandas as pd
import dashscope
from dashscope import Generation
import time
from tqdm import tqdm
import re

dashscope.api_key = "sk-f39ab33acbc44d3ba8cdea5aac2e1d7a"

FEWSHOT_PROMPT = """Here are several examples. For each sentence, extract ONLY the exact phrase as the opinion holder. If there is no holder, output nothing.

Example 1:
Sentence: "African observers generally approved of his victory , while Western governments denounced it ."
Holder: Western governments

Example 2:
Sentence: "All of this leaves in the air a feeling of military uncertainty , after a two-way trip in 24 hours which will undoubtedly leave deep wounds in armed forces which clearly have not given up their wish to exert a decisive influence in political life ."
Holder: armed forces

Example 3:
Sentence: "It denounces human rights violation but supports Israel for perishing the Muslims in Palestine and Lebanon , '' Abbas criticized ."
Holder: It

Now for the following sentence:
"""

def make_holder_fewshot_prompt(sentence):
    return FEWSHOT_PROMPT + f'Sentence: "{sentence}"\nHolder:'

def call_qwen_holder(sentence, model_name="qwen-max-latest", max_retry=3):
    prompt = make_holder_fewshot_prompt(sentence)
    for attempt in range(max_retry):
        try:
            response = Generation.call(
                model=model_name,
                prompt=prompt,
                top_p=0.8,
                temperature=0.2,
            )
            if (response is None) or (not hasattr(response, "output")) or (not hasattr(response.output, "text")):
                print(f"API返回异常，未获得结果 | Sentence: {sentence}")
                print("response内容：", response)
                return ""
            result = response.output.text.strip()
            match = re.search(r'Holder:\s*(.*)', result, re.IGNORECASE)
            if match:
                holder = match.group(1).strip()
            else:
                holder = result.strip()
            return holder
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(2)
            else:
                print(f"Error: {e} | Sentence: {sentence}")
                return ""

if __name__ == "__main__":
    try:
        df = pd.read_csv('holder_target_test - 副本.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv('holder_target_test - 副本.csv', encoding='gbk')

    if 'sentence' not in df.columns:
        raise Exception("找不到'sentence'列，请检查输入文件表头。")

    holders = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Qwen批量Holder抽取"):
        sentence = str(row['sentence'])
        holder = call_qwen_holder(sentence, model_name="qwen-max-latest")
        holders.append(holder)
        time.sleep(0.8)

    df['holder_pred'] = holders
    df.to_csv('holder_target_test_with_qwen_pred1.csv', index=False, encoding='utf-8-sig')
    print("✅ 全部完成，结果已保存为 holder_target_test_with_qwen_pred1.csv")
