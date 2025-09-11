import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 合并
gt = pd.read_csv('sentiment_test.csv', encoding='utf-8-sig')
pred = pd.read_csv('sentiment_test_with_qwen3_pred.csv', encoding='utf-8-sig')
df = gt.merge(pred[['sentence', 'qwen3_score']], on='sentence', how='left')

# 去掉预测为空的行
df = df[~df['qwen3_score'].isnull()]

# 统一评测
y_true = df['score_human']
y_pred = df['qwen3_score']
print('Qwen大模型 MSE:', mean_squared_error(y_true, y_pred))
print('Qwen大模型 MAE:', mean_absolute_error(y_true, y_pred))
print('Qwen大模型 R2:', r2_score(y_true, y_pred))
print('Qwen大模型 Pearson相关系数:', np.corrcoef(y_true, y_pred)[0,1])
