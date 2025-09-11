import openai
import pandas as pd
import sys
import time

# =========== 配置区 ==========
OPENAI_API_KEY = "	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"   # ← 请填你的API Key
INPUT_CSV = "sentiment_train副本.csv"         # ← 输入文件名
OUTPUT_CSV = "sentiment_train_with_chatgpt_pred.csv"  # ← 输出文件名
MODEL_NAME = "gpt-4o"             # 或 "gpt-4"（若有权限）

# ======= 只用gbk读取CSV =======
def read_csv_gbk(filepath):
    try:
        df = pd.read_csv(filepath, encoding='gbk')
        print(f'文件读取成功，编码方式：gbk')
        print(df.head())
        print("字段：", df.columns)
        return df
    except Exception as e:
        print(f"用gbk编码读取失败，错误信息：{e}")
        sys.exit(1)

# ======= 调用ChatGPT获取情感分值 =======
def get_sentiment_score(sentence):
    prompt = (
        "You are a sentiment analysis expert. For the following sentence, "
        "output only a single floating-point sentiment score between -1 and 1, with two decimal places. "
        "Do not output any explanation or label.\n\n"
        f"Sentence: {sentence}"
    )
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,  # 最多等30秒
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用API出错：{e}")
        return None

# ======= 主程序 =======
if __name__ == '__main__':
    openai.api_key = OPENAI_API_KEY
    df = read_csv_gbk(INPUT_CSV)
    if 'sentence' not in df.columns:
        print("CSV文件缺少'sentence'这一列，请检查文件内容！")
        sys.exit(1)

    scores = []
    for i, s in enumerate(df['sentence']):
        print(f"处理第 {i+1} 条：{s}")
        score = get_sentiment_score(s)
        print(f"得分：{score}")
        scores.append(score)
        time.sleep(1)  # 防止超速限（如需调试可临时注释）

    df['chatgpt_pred_score'] = scores
    df.to_csv(OUTPUT_CSV, encoding='gbk', index=False)
    print(f"\n全部完成，结果已保存至 {OUTPUT_CSV}")
