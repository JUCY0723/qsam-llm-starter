import pandas as pd
import dashscope
from dashscope import Generation
import time
from tqdm import tqdm
import re

dashscope.api_key = "sk-f39ab33acbc44d3ba8cdea5aac2e1d7a"

def make_prompt(sentence):
    prompt = f"""
You are a sentiment analysis expert. For the following sentence,
 output only a single floating-point sentiment score between -1 and 1, 
 with two decimal places. Do not output any explanation or label.

Score definition:
- Range: Any real number between -1 (extremely negative) and 1 (extremely positive).
- Meaning:
    - -1: Extremely negative (very pessimistic/bearish/opposed/bad news).
    - 0: Neutral (no clear positive or negative tendency).
    - 1: Extremely positive (very optimistic/bullish/supportive/good news).
    - Scores closer to -1 mean the sentiment is more negative, more unfavorable for the financial subject (e.g., company, stock, market).
    - Scores closer to +1 mean the sentiment is more positive, more favorable for the financial subject.
    - Scores around 0 mean the text expresses a neutral or objective attitude, with no obvious position.

Sentence: {sentence}
Output:
"""
    return prompt

def call_qwen(sentence, model_name="qwen-max-latest", max_retry=3):
    prompt = make_prompt(sentence)
    for attempt in range(max_retry):
        try:
            response = Generation.call(
                model=model_name,
                prompt=prompt,
                top_p=0.8,
                temperature=0.2,
            )
            # 健壮性判断，防止API无返回或结构变动时报错
            if (response is None) or (not hasattr(response, "output")) or (not hasattr(response.output, "text")):
                print(f"API返回异常，未获得结果 | Sentence: {sentence}")
                print("response内容：", response)
                return None
            result = response.output.text.strip()
            match = re.search(r'-?\d+\.\d+', result)
            if match:
                score = round(float(match.group()), 2)
                return score
            else:
                print(f"模型输出无法提取分值 | Sentence: {sentence} | Output: {result}")
                return None
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(2)
            else:
                print(f"Error: {e} | Sentence: {sentence}")
                return None

if __name__ == "__main__":
    # 自动兼容编码（优先utf-8-sig，失败则gbk）
    try:
        df = pd.read_csv('sentiment_test副本.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv('sentiment_test副本.csv', encoding='gbk')

    # 自动识别列名
    if 'sentence' in df.columns:
        col = 'sentence'
    elif '内容' in df.columns:
        col = '内容'
    else:
        raise Exception("找不到'sentence'或'内容'这两列，请检查输入文件表头。")

    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Qwen情感分值分析"):
        sentence = str(row[col])
        score = call_qwen(sentence, model_name="qwen-max-latest")
        results.append({
            "sentence": sentence,
            "qwen_pred_score": score,
        })
        time.sleep(0.8)

    pd.DataFrame(results).to_csv('sentiment_test_with_qwen_pred.csv', index=False, encoding='utf-8-sig')
    print("全部完成，结果已保存为sentiment_test_with_qwen_pred.csv")
