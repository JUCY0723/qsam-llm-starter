import pandas as pd
import numpy as np
import re
from collections import Counter
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np

# ========== 1. 读取数据 ==========
def robust_read_csv(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding='gbk')

df_train = robust_read_csv('sentiment_train_with_qwen_pred.csv')
df_pred = robust_read_csv('sentiment_test_with_qwen_pred.csv')
df_human = robust_read_csv('sentiment_test.csv')
df_test = pd.merge(df_pred, df_human, on='sentence', how='left')

# ========== 2. 特征工程函数 ==========
def build_features(df, top_words):
    pos_set = set(['good', 'great', 'positive', 'happy', 'improve', 'success', 'support', 'increase', 'optimistic', 'growth', 'benefit', 'rise', 'up', 'bullish', 'profit', 'strong', 'opportunity', 'innovative', 'record', 'lead', 'boost'])
    neg_set = set(['bad', 'poor', 'negative', 'sad', 'worse', 'risk', 'decrease', 'loss', 'decline', 'down', 'bearish', 'fail', 'weak', 'crisis', 'fall', 'drop', 'miss', 'concern', 'uncertain', 'problem', 'impact'])
    topic_words = ['policy', 'government', 'company', 'reform', 'down', 'up', 'profit', 'impact', 'market', 'stock', 'energy', 'project', 'new', 'public']
    emotion_phrases = ['值得关注', '影响较大', '存在风险', '利好', '不确定性', '信心', '创新', '破纪录', '盈利增长', '面临挑战']

    def count_polar_words(text):
        words = str(text).lower().split()
        pos = sum(1 for w in words if w in pos_set)
        neg = sum(1 for w in words if w in neg_set)
        return pd.Series([pos, neg])

    def count_phrase(text, phrase_list):
        s = str(text).lower()
        return sum([s.count(phrase) for phrase in phrase_list])

    df['len'] = df['sentence'].apply(len)
    df['word_count'] = df['sentence'].apply(lambda x: len(str(x).split()))
    df[['pos_cnt', 'neg_cnt']] = df['sentence'].apply(count_polar_words)
    df['punct_cnt'] = df['sentence'].apply(lambda x: sum([str(x).count('!'), str(x).count('?'), str(x).count(',')]))
    df['positive_ratio'] = (df['pos_cnt'] + 1) / (df['word_count'] + 1)
    df['negative_ratio'] = (df['neg_cnt'] + 1) / (df['word_count'] + 1)
    df['negation_cnt'] = df['sentence'].apply(lambda x: len(re.findall(r'\b(no|not|never|无|没|否认|未)\b', str(x).lower())))
    df['topic_cnt'] = df['sentence'].apply(lambda x: count_phrase(x, topic_words))
    df['emotion_phrase_cnt'] = df['sentence'].apply(lambda x: count_phrase(x, emotion_phrases))
    df['question_mark'] = df['sentence'].apply(lambda x: '?' in str(x))
    df['exclamation_mark'] = df['sentence'].apply(lambda x: '!' in str(x))
    df['comma_count'] = df['sentence'].apply(lambda x: str(x).count(','))
    df['negation_start'] = df['sentence'].apply(lambda x: str(x).strip().startswith(('No', 'Not', '无', '未')))
    df['has_percent'] = df['sentence'].apply(lambda x: '%' in str(x) or 'percent' in str(x).lower())
    df = df.sort_index()
    df['qwen_pred_score_rolling'] = df['qwen_pred_score'].rolling(window=2, min_periods=1, center=True).mean()
    df['pos_cnt_x_len'] = df['pos_cnt'] * df['len']
    df['neg_cnt_x_word_count'] = df['neg_cnt'] * df['word_count']
    df['qwen_x_positive_ratio'] = df['qwen_pred_score'] * df['positive_ratio']
    df['qwen_sq'] = df['qwen_pred_score'] ** 2

    # 新增特征
    df['qwen_x_pos_cnt'] = df['qwen_pred_score'] * df['pos_cnt']
    df['qwen_x_neg_cnt'] = df['qwen_pred_score'] * df['neg_cnt']
    df['unique_word_ratio'] = df['sentence'].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1))
    df['numeric_ratio'] = df['sentence'].apply(lambda x: len(re.findall(r'\d+', str(x))) / (len(str(x).split()) + 1))
    df['qwen_pred_score_std'] = df['qwen_pred_score'].rolling(window=3, min_periods=1, center=True).std().fillna(0)

    # top_words特征
    for w in top_words:
        df[f'has_{w}'] = df['sentence'].apply(lambda x: int(w in str(x).lower()))
    return df

# ========== 3. top_words 统计与保存 ==========
if not (os.path.exists('top_words.pkl')):
    top_words = [w for w, _ in Counter(" ".join(df_train['sentence'].str.lower()).split()).most_common(10)]
    with open('top_words.pkl', 'wb') as f:
        pickle.dump(top_words, f)
else:
    with open('top_words.pkl', 'rb') as f:
        top_words = pickle.load(f)

# ========== 4. 构建特征 ==========
df_train = build_features(df_train, top_words)
df_test = build_features(df_test, top_words)

# ========== 5. 特征列 ==========
feature_cols = [
    'qwen_pred_score', 'len', 'word_count', 'pos_cnt', 'neg_cnt', 'punct_cnt',
    'positive_ratio', 'negative_ratio', 'negation_cnt', 'topic_cnt', 'emotion_phrase_cnt',
    'question_mark', 'exclamation_mark', 'comma_count', 'negation_start', 'has_percent',
    'qwen_pred_score_rolling', 'pos_cnt_x_len', 'neg_cnt_x_word_count', 'qwen_x_positive_ratio', 'qwen_sq',
    'qwen_x_pos_cnt', 'qwen_x_neg_cnt', 'unique_word_ratio', 'numeric_ratio', 'qwen_pred_score_std'
] + [f'has_{w}' for w in top_words]

X_train = df_train[feature_cols]
y_train = df_train['score_human']
X_test = df_test[feature_cols]
y_test = df_test['score_human']

# ========== 6. XGBoost参数搜索 ==========
xgb_model = xgb.XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("XGB最佳参数:", grid_search.best_params_)

# ========== 7. 融合多模型（VotingRegressor） ==========
reg_xgb = grid_search.best_estimator_
reg_lgb = lgb.LGBMRegressor(random_state=42)
reg_rf = RandomForestRegressor(random_state=42)
reg_lr = LinearRegression()

voting = VotingRegressor([('xgb', reg_xgb), ('lgb', reg_lgb), ('rf', reg_rf), ('lr', reg_lr)])
voting.fit(X_train, y_train)
y_pred_train = voting.predict(X_train)
y_pred_test = voting.predict(X_test)

print("【训练集】")
print('R²:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))

print("【测试集】")
print('R²:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test, y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))

# ========== 8. 保存模型和结果 ==========
joblib.dump(voting, 'voting_super_calibrator.model')
print("Voting模型已保存为 voting_super_calibrator.model")
with open('top_words.pkl', 'wb') as f:
    pickle.dump(top_words, f)
print("top_words已保存为 top_words.pkl")

df_test['voting_super_calibrated'] = y_pred_test
df_test.to_csv('sentiment_test_super_calibrated.csv', index=False, encoding='utf-8-sig')
print('预测结果已保存为 sentiment_test_super_calibrated.csv')
