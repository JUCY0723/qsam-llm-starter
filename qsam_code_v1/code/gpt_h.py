import os

import openai
import pandas as pd
import sys
import time
import re

# =========== 配置区 ==========

OPENAI_API_KEY = "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"   # ← 请填你的API Key
INPUT_CSV = "holder_target_test副本.csv"         # ← 输入文件名
OUTPUT_CSV = "holder_target_test_with_chatgpt_pred.csv"  # ← 输出文件名
MODEL_NAME = "gpt-4o"             # 或 "gpt-4"
# ======= 先打印一下当前工作目录和文件列表，帮助排查 =======
print("工作目录：", os.getcwd())
print("当前目录下文件：", os.listdir(os.getcwd()))

# ======= 自动尝试多种编码读取 CSV =======
def read_csv_with_auto_encoding(filepath):
    if not os.path.isfile(filepath):
        print(f"Error: 找不到文件：{filepath}")
        sys.exit(1)
    for enc in ('gbk', 'utf-8', 'utf-8-sig'):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"成功以 {enc} 编码读取 CSV")
            return df
        except Exception as e:
            print(f">>> 用 {enc} 打开失败：{e}")
    print("所有编码尝试失败，程序退出。")
    sys.exit(1)

# ======= 调用 ChatGPT 提取 Holder =======
def get_holder(sentence):
    prompt = (
        "Please extract the opinion holder from the following sentence.\n"
        "Output ONLY the exact phrase as the holder. If there is no holder, output nothing.\n\n"
        f'Sentence: "{sentence}"\nHolder:'
    )
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r'Holder:\s*(.*)', text, re.IGNORECASE)
        return m.group(1).strip() if m else text
    except Exception as e:
        print(f"调用 API 出错：{e}")
        return ""

# ======= 主程序入口 =======
if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY

    # 1. 读取 CSV
    df = read_csv_with_auto_encoding(INPUT_CSV)

    # 2. 检查列名
    if 'sentence' not in df.columns:
        print("CSV 文件缺少 'sentence' 列，请检查！")
        sys.exit(1)

    # 3. 逐条调用 API
    holders = []
    for idx, s in enumerate(df['sentence'], start=1):
        print(f"[{idx}/{len(df)}] 句子：{s}")
        holders.append(get_holder(str(s)))
        time.sleep(1)

    # 4. 保存结果
    df['holder_pred'] = holders
    df.to_csv(OUTPUT_CSV, encoding='gbk', index=False)
    print(f"\n全部完成，结果保存在：{OUTPUT_CSV}")
