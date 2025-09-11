import pandas as pd
from collections import Counter

# ======== 配置区：只需改下面这3个文件名 ========
PRED_CSV   = "holder_target_test_with_chatgpt_pred.csv"   # GPT模型输出文件
GOLD_CSV   = "holder_target_test.csv"                     # gold人工标注文件
OUTPUT_CSV = "holder_eval_with_chatgpt.csv"               # 输出评估文件

# ============ 数据加载 ============
pred_df = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
gold_df = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")

for df in (pred_df, gold_df):
    df["sentence"] = df["sentence"].astype(str).str.strip()
    if 'holder_pred' in df.columns:
        df["holder_pred"] = df["holder_pred"].fillna("").astype(str).str.strip()
    if 'holder' in df.columns:
        df["holder"] = df["holder"].fillna("").astype(str).str.strip()

pred_df = pred_df.rename(columns={"holder_pred":"holder_pred"})
gold_df = gold_df.rename(columns={"holder":"holder_gold"})

df = pd.merge(gold_df, pred_df, on="sentence", how="left").fillna("")

# ============ 严格评估 ============
metrics = Counter()
for _, row in df.iterrows():
    g_h, p_h = row["holder_gold"], row["holder_pred"]
    if p_h:
        metrics["tp"] += (p_h == g_h and g_h != "")
        metrics["fp"] += (p_h != g_h)
    if g_h and p_h != g_h:
        metrics["fn"] += 1

def calc_prf(cnt):
    tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = (2*p*r/(p+r) if p+r else 0.0)
    return round(p,4), round(r,4), round(f1,4)

strict_p, strict_r, strict_f1 = calc_prf(metrics)

print(f"\n===== GPT严格评估结果 =====")
print(f"Precision: {strict_p}, Recall: {strict_r}, F1: {strict_f1}, TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

# ============ 宽松评估（子串匹配） ============
def loose_match(gold, pred):
    gold, pred = gold.strip(), pred.strip()
    if not gold and not pred:
        return True
    if not gold or not pred:
        return False
    return (gold in pred) or (pred in gold)

metrics_loose = Counter()
for _, row in df.iterrows():
    g_h, p_h = row["holder_gold"], row["holder_pred"]
    if loose_match(g_h, p_h):
        if g_h:
            metrics_loose["tp"] += 1
    else:
        if p_h: metrics_loose["fp"] += 1
        if g_h: metrics_loose["fn"] += 1

loose_p, loose_r, loose_f1 = calc_prf(metrics_loose)

print(f"\n===== GPT宽松评估结果（子串匹配） =====")
print(f"Precision: {loose_p}, Recall: {loose_r}, F1: {loose_f1}, TP: {metrics_loose['tp']}, FP: {metrics_loose['fp']}, FN: {metrics_loose['fn']}")

# ============ 输出到csv ============
out_df = pd.DataFrame([
    {"type":"strict", "precision":strict_p, "recall":strict_r, "f1":strict_f1, "tp":metrics["tp"], "fp":metrics["fp"], "fn":metrics["fn"]},
    {"type":"loose",  "precision":loose_p, "recall":loose_r, "f1":loose_f1, "tp":metrics_loose["tp"], "fp":metrics_loose["fp"], "fn":metrics_loose["fn"]}
])
out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n评估结果已保存至 {OUTPUT_CSV}")
