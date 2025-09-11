import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. 读取训练和测试集（都带有score_human列！）
train = pd.read_csv('sentiment_train.csv', encoding='utf-8-sig')
test = pd.read_csv('sentiment_test.csv', encoding='utf-8-sig')

# 2. 特征提取（TF-IDF向量化）
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train['sentence'])
X_test = vectorizer.transform(test['sentence'])

# 3. 支持向量回归（SVR）训练
reg = SVR(kernel='linear')
reg.fit(X_train, train['score_human'])

# 4. 预测
y_pred = reg.predict(X_test)
y_true = test['score_human']

# 5. 评估
print('Mean Squared Error (MSE):', mean_squared_error(y_true, y_pred))
print('Mean Absolute Error (MAE):', mean_absolute_error(y_true, y_pred))
print('R^2 Score:', r2_score(y_true, y_pred))
print('Pearson相关系数:', np.corrcoef(y_true, y_pred)[0,1])

# 6. 结果保存（可选）
result_df = test.copy()
result_df['svr_pred_score'] = y_pred
result_df.to_csv('sentiment_test_with_svr_pred.csv', index=False, encoding='utf-8-sig')
print('预测结果已保存至 sentiment_test_with_svr_pred.csv')
