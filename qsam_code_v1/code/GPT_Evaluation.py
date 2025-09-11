import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 自动尝试三种常见编码读取csv
def robust_read_csv(filepath):
    for enc in ['utf-8-sig', 'utf-8', 'gbk']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"{filepath} 读取成功，编码方式：{enc}")
            return df
        except Exception as e:
            print(f"{filepath} 尝试 {enc} 失败，错误信息：{e}")
    raise Exception(f"所有常用编码均读取失败，请检查 {filepath} 文件格式！")

# 1. 读取人工分数和GPT预测分数
gt = robust_read_csv('sentiment_test.csv')
pred = robust_read_csv('sentiment_test_with_chatgpt_pred.csv')
df = gt.merge(pred[['sentence', 'chatgpt_pred_score']], on='sentence', how='left')

# 2. 去掉预测为空的行
df = df[~df['chatgpt_pred_score'].isnull()]

# 3. 转换为float（防止有字符串类型）
y_true = df['score_human'].astype(float)
y_pred = df['chatgpt_pred_score'].astype(float)

# 4. 统一评测
print('GPT大模型 MSE:', mean_squared_error(y_true, y_pred))
print('GPT大模型 MAE:', mean_absolute_error(y_true, y_pred))
print('GPT大模型 R2:', r2_score(y_true, y_pred))
print('GPT大模型 Pearson相关系数:', np.corrcoef(y_true, y_pred)[0, 1])
