import pandas as pd

# 1. 读取文件
df = pd.read_csv('sentiment_test_super_calibrated.csv')

# 2. 假设有以下三列（请根据实际字段名调整）
# 'score_human'：人工分数
# 'score_pred_before'：校正前预测分数
# 'score_pred_after' ：校正后预测分数

# 3. 计算误差
df['err_before'] = (df['score_pred_before'] - df['score_human']).abs()
df['err_after'] = (df['score_pred_after'] - df['score_human']).abs()

# 4. 设定高误差阈值
high_err_thresh = 0.4  # 你可以调整，比如0.5

# 5. 标注高误差
df['is_high_err_before'] = df['err_before'] > high_err_thresh
df['is_high_err_after'] = df['err_after'] > high_err_thresh

# 6. 可以再加一列误差变化
df['err_change'] = df['err_after'] - df['err_before']

# 7. 统计高误差样本数量
print("校正前高误差样本数：", df['is_high_err_before'].sum())
print("校正后高误差样本数：", df['is_high_err_after'].sum())
print("依然高误差（两者都高误差）：", ((df['is_high_err_before']) & (df['is_high_err_after'])).sum())
print("校正后治愈（前高误差后低误差）：", ((df['is_high_err_before']) & (~df['is_high_err_after'])).sum())
print("校正后新高误差（前低误差后高误差）：", ((~df['is_high_err_before']) & (df['is_high_err_after'])).sum())

# 8. 筛出高误差样本（可导出做人工分析）
df_high_err_before = df[df['is_high_err_before']]
df_high_err_after = df[df['is_high_err_after']]

# 9. 保存结果，方便你后续人工标注
df.to_csv('sentiment_test_with_high_error_flag.csv', index=False)
df_high_err_before.to_csv('high_error_samples_before.csv', index=False)
df_high_err_after.to_csv('high_error_samples_after.csv', index=False)

print("标注结果已保存到 sentiment_test_with_high_error_flag.csv。")
