import json
import pandas as pd

def extract_holder_target_v3(file_path, output_csv='holder_target_output.csv', max_len=50):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for doc_key, doc_value in data.items():
        if not isinstance(doc_value, dict):
            continue
        for sent_key, sent_value in doc_value.items():
            if not (isinstance(sent_value, dict) and 'sentence_tokenized' in sent_value):
                continue
            sentence_tokens = sent_value.get('sentence_tokenized', [])
            # 过滤超长句
            if len(sentence_tokens) > max_len:
                continue
            sentence_text = ' '.join(sentence_tokens)
            for ds_key, ds_value in sent_value.items():
                if not (ds_key.startswith('ds') and isinstance(ds_value, dict)):
                    continue
                # holder
                holder = ds_value.get('holders_tokenized', [])
                holder_text = ' ; '.join([' '.join(h) for h in holder if len(h) > 0])
                att_num = ds_value.get('att_num', 1)
                for i in range(att_num):
                    att = ds_value.get(f'att{i}', None)
                    if att is None:
                        continue
                    target = att.get('targets_tokenized', [])
                    target_text = ' ; '.join([' '.join(t) for t in target if len(t) > 0])
                    # **只保留holder和target都不为空的情况**
                    if holder_text and target_text:
                        results.append({
                            'sentence': sentence_text,
                            'holder': holder_text,
                            'target': target_text
                        })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f'抽取完成，已保存为 {output_csv}，共{len(results)}条（只保留holder和target都不为空的记录，且已过滤超长句）。')

# 用法示例（只保留holder和target都不为空，且单句<=50词）
extract_holder_target_v3('train_fold_9.json', 'holder_target_train9.csv', max_len=50)
