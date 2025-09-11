import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# 1. 参数
MODEL_DIR = '/root/autodl-tmp/sentiment/models/roberta-finetuned-sentiment'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据加载
file_path = 'sentiment_test_super_calibrated.csv'
df = pd.read_csv(file_path)

# 3. 误差与负例筛选
df['error'] = np.abs(df['voting_super_calibrated'] - df['score_human'])
threshold = df['error'].quantile(0.89)
df['sample_type'] = df['error'].apply(lambda x: 'negative' if x > threshold else 'positive')
positive_count = (df['sample_type'] == 'positive').sum()
negative_count = (df['sample_type'] == 'negative').sum()
print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")

negative_samples = df[df['sample_type'] == 'negative'].copy()
negative_samples = negative_samples.dropna(subset=['sentence'])

# 4. 加载Tokenizer和微调后的模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(DEVICE)
model.eval()

# 5. 定义推理用Dataset和DataLoader
class InferDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'attention_mask': attention_mask
        }

infer_dataset = InferDataset(negative_samples['sentence'].tolist(), tokenizer)
infer_loader = DataLoader(infer_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. 推理
pred_scores = []
with torch.no_grad():
    for batch in tqdm(infer_loader, desc="RoBERTa回归推理", ncols=100):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze().cpu().numpy()
        # 保证batch size为1时数据结构不变
        if preds.ndim == 0:
            preds = np.array([preds])
        pred_scores.extend(preds.tolist())

negative_samples['roberta_pred_score'] = pred_scores

# 7. 保存结果
negative_samples.to_csv('negative_samples_with_roberta_regression.csv', index=False, encoding='utf-8-sig')
print("保存完成：negative_samples_with_roberta_regression.csv")

# 8. 展示前10条对比
print(negative_samples[['sentence', 'score_human', 'voting_super_calibrated', 'roberta_pred_score']].head(10))
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# 1. 参数
MODEL_DIR = '/root/autodl-tmp/sentiment/models/roberta-finetuned-sentiment'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据加载
file_path = 'sentiment_test_super_calibrated.csv'
df = pd.read_csv(file_path)

# 3. 误差与负例筛选
df['error'] = np.abs(df['voting_super_calibrated'] - df['score_human'])
threshold = df['error'].quantile(0.89)
df['sample_type'] = df['error'].apply(lambda x: 'negative' if x > threshold else 'positive')
positive_count = (df['sample_type'] == 'positive').sum()
negative_count = (df['sample_type'] == 'negative').sum()
print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")

negative_samples = df[df['sample_type'] == 'negative'].copy()
negative_samples = negative_samples.dropna(subset=['sentence'])

# 4. 加载Tokenizer和微调后的模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(DEVICE)
model.eval()

# 5. 定义推理用Dataset和DataLoader
class InferDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'attention_mask': attention_mask
        }

infer_dataset = InferDataset(negative_samples['sentence'].tolist(), tokenizer)
infer_loader = DataLoader(infer_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. 推理
pred_scores = []
with torch.no_grad():
    for batch in tqdm(infer_loader, desc="RoBERTa回归推理", ncols=100):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze().cpu().numpy()
        # 保证batch size为1时数据结构不变
        if preds.ndim == 0:
            preds = np.array([preds])
        pred_scores.extend(preds.tolist())

negative_samples['roberta_pred_score'] = pred_scores

# 7. 保存结果
negative_samples.to_csv('negative_samples_with_roberta_regression.csv', index=False, encoding='utf-8-sig')
print("保存完成：negative_samples_with_roberta_regression.csv")

# 8. 展示前10条对比
print(negative_samples[['sentence', 'score_human', 'voting_super_calibrated', 'roberta_pred_score']].head(10))
