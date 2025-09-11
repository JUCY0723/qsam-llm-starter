import openai
import pandas as pd
import re
import time
from tqdm import tqdm

openai.api_key = "sk-27d182b13f394b7c8f334dc4751506ea"
openai.api_base  = "https://api.deepseek.com/v1"

def make_prompt(sentence):
    return f"""You are a sentiment analysis expert. For the following sentence, output only a single floating-point sentiment score between -1 and 1, with two decimal places. Do not output any explanation or label.

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
Output:"""

def call_deepseek(sentence, model="deepseek-chat", max_retry=3):
    prompt = make_prompt(sentence)
    for attempt in range(max_retry):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.8
            )
            result = response.choices[0].message.content.strip()
            match = re.search(r'-?\d+\.\d+', result)
            if match:
                return round(float(match.group()), 2)
            else:
                print(f"无法提取分值: {result}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
    return None

if __name__ == "__main__":
    # 只需改这里，换成 sentiment_train 副本文件
    try:
        df = pd.read_csv('sentiment_train副本.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv('sentiment_train副本.csv', encoding='gbk')

    if 'sentence' in df.columns:
        col = 'sentence'
    elif '内容' in df.columns:
        col = '内容'
    else:
        raise Exception("找不到'sentence'或'内容'这两列，请检查输入文件表头。")

    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="DeepSeek情感分值分析"):
        sentence = str(row[col])
        score = call_deepseek(sentence)
        results.append({
            "sentence": sentence,
            "deepseek_pred_score": score,
        })
        time.sleep(1.2)  # 限速保护

    pd.DataFrame(results).to_csv('sentiment_train_with_deepseek_pred.csv', index=False, encoding='utf-8-sig')
    print("全部完成，结果已保存为 sentiment_train_with_deepseek_pred.csv")
