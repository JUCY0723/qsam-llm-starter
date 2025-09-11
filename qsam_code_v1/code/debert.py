import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from tqdm import tqdm

# 1. 配置参数
MODEL_PATH = 'models/deberta-v3-base'
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 读取数据
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
        inputs = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 4. 指定use_fast=False，加载分词器和模型
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1, problem_type="regression")
model = model.to(DEVICE)

# 5. DataLoader
train_dataset = SentimentDataset(train_df, tokenizer)
test_dataset = SentimentDataset(test_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. 优化器与调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS
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

# 8. 评估
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
print('DeBERTa-v3回归 MSE:', mean_squared_error(all_labels, all_preds))
print('DeBERTa-v3回归 MAE:', mean_absolute_error(all_labels, all_preds))
print('DeBERTa-v3回归 R2:', r2_score(all_labels, all_preds))
print('DeBERTa-v3回归 Pearson相关系数:', np.corrcoef(all_labels, all_preds)[0,1])

# 9. 保存结果
test_df['deberta_pred_score'] = all_preds
test_df.to_csv('sentiment_test_with_deberta_pred.csv', index=False, encoding='utf-8-sig')
print('DeBERTa-v3回归结果已保存至 sentiment_test_with_deberta_pred.csv')
