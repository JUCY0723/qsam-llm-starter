import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from tqdm import tqdm
import os

# 1. 参数配置
MODEL_PATH = 'models/roberta-base'  # 初始预训练权重目录
SAVE_DIR = 'models/roberta-finetuned-sentiment'  # 微调后模型保存目录
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据准备
train_df = pd.read_csv('sentiment_train.csv', encoding='utf-8-sig')
test_df = pd.read_csv('sentiment_test.csv', encoding='utf-8-sig')

# 3. 数据集类
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['sentence'])
        label = self.data.iloc[idx]['score_human']
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 4. 加载Tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1, problem_type="regression")
model = model.to(DEVICE)

# 5. DataLoader
train_dataset = SentimentDataset(train_df, tokenizer)
test_dataset = SentimentDataset(test_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. 优化器与学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)

# 7. 训练
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', ncols=100)
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        loop.set_postfix(loss=loss.item())

# 8. 保存模型与分词器
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"模型和tokenizer已保存至 {SAVE_DIR}")

# 9. 评估（推理）
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].cpu().numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze().cpu().numpy()
        all_preds.extend(preds.tolist() if isinstance(preds, np.ndarray) else [preds])
        all_labels.extend(labels.tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
print('RoBERTa回归 MSE:', mean_squared_error(all_labels, all_preds))
print('RoBERTa回归 MAE:', mean_absolute_error(all_labels, all_preds))
print('RoBERTa回归 R2:', r2_score(all_labels, all_preds))
print('RoBERTa回归 Pearson相关系数:', np.corrcoef(all_labels, all_preds)[0,1])

# 10. 保存预测结果
test_df['roberta_pred_score'] = all_preds
test_df.to_csv('sentiment_test_with_roberta_pred.csv', index=False, encoding='utf-8-sig')
print('RoBERTa回归结果已保存至 sentiment_test_with_roberta_pred.csv')
