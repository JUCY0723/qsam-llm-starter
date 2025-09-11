import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 读取表格
df = pd.read_excel('holder_test_with_qwen_class.xlsx')

# 2. 检查必须字段
assert 'human_class' in df.columns, "缺少人工分类列human_class"
assert 'qwen_class' in df.columns, "缺少大模型预测分类列qwen_class"

# 3. 丢弃缺失或无效行
df = df.dropna(subset=['human_class', 'qwen_class'])

# 转为int型
df['human_class'] = df['human_class'].astype(int)
df['qwen_class'] = df['qwen_class'].astype(int)

# 4. 计算准确率
acc = accuracy_score(df['human_class'], df['qwen_class'])
print(f"千问分类准确率：{acc:.3f}")

# 5. 输出混淆矩阵
cm = confusion_matrix(df['human_class'], df['qwen_class'], labels=[1,2,3])
print("混淆矩阵（行是人工类别，列是模型类别）：")
print(cm)

# 6. 输出详细分类报告
print("\n详细分类报告（precision/recall/f1/支持度）：")
print(classification_report(df['human_class'], df['qwen_class'], labels=[1,2,3], digits=3, target_names=['媒体评述句', '事实陈述句', '第三方引述句']))

# 7. 可选：输出分错的句子
err_df = df[df['human_class'] != df['qwen_class']]
err_df[['sentence', 'holder', 'human_class', 'qwen_class']].to_excel('分类错误案例.xlsx', index=False)
print(f"分错案例已保存到 分类错误案例.xlsx")
