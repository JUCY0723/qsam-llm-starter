import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. 读取测试集
test = pd.read_csv('sentiment_test.csv', encoding='utf-8-sig')

# 2. VADER初始化
analyzer = SentimentIntensityAnalyzer()

# 3. 获取每条句子的vader分值
def get_vader_score(text):
    vs = analyzer.polarity_scores(str(text))
    return vs['compound']  # [-1, 1]之间的分数

test['vader_score'] = test['sentence'].apply(get_vader_score)

# 4. 对比人工分值
y_true = test['score_human']
y_pred = test['vader_score']

print('VADER Mean Squared Error (MSE):', mean_squared_error(y_true, y_pred))
print('VADER Mean Absolute Error (MAE):', mean_absolute_error(y_true, y_pred))
print('VADER R² Score:', r2_score(y_true, y_pred))
print('VADER Pearson相关系数:', np.corrcoef(y_true, y_pred)[0,1])

# 5. 保存结果
test.to_csv('sentiment_test_with_vader.csv', index=False, encoding='utf-8-sig')
print('VADER预测结果已保存至 sentiment_test_with_vader.csv')
