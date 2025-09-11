import openai
import pandas as pd
import re
import time
from tqdm import tqdm

openai.api_key = "sk-27d182b13f394b7c8f334dc4751506ea"
openai.api_base  = "https://api.deepseek.com/v1"

def make_holder_prompt(sentence):
    return f"""Please extract the opinion holder from the following sentence.
Output ONLY the exact phrase as the holder. If there is no holder, output nothing.

Sentence: "{sentence}"
Holder:"""

def call_deepseek_holder(sentence, model="deepseek-chat", max_retry=3):
    prompt = make_holder_prompt(sentence)
    for attempt in range(max_retry):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.8
            )
            result = response.choices[0].message.content.strip()
            # 提取 "Holder:" 后面内容（支持为空）
            match = re.search(r'Holder:\s*(.*)', result, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            else:
                # 兼容输出没有"Holder:"前缀的情况
                return result
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
    return ""

if __name__ == "__main__":
    # 输入是 holder_target_test - 副本.csv，支持编码自动兼容
    try:
        df = pd.read_csv('holder_target_test - 副本.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv('holder_target_test - 副本.csv', encoding='gbk')

    if 'sentence' not in df.columns:
        raise Exception("找不到'sentence'这列，请检查输入文件表头。")

    holders = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="DeepSeek批量Holder抽取"):
        sentence = str(row['sentence'])
        holder = call_deepseek_holder(sentence)
        holders.append(holder)
        time.sleep(1.2)  # DeepSeek接口建议慢速，防止限流

    df['holder_pred'] = holders
    df.to_csv('holder_target_test_with_deepseek_pred.csv', index=False, encoding='utf-8-sig')
    print("✅ 全部完成，结果已保存为 holder_target_test_with_deepseek_pred.csv")
