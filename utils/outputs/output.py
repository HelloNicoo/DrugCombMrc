import json
ori_path = "/home/zpq/pythonProject/drug_combo_spert7/epoch_7_capsule"
input_path = "/home/zpq/pythonProject/drug_combo_spert7/utils/outputs/final_pred_cap.jsonl"
with open(ori_path, 'r', encoding='utf-8') as f, open(input_path, 'w', encoding="utf-8") as w:
    lines = f.read().splitlines()
    for line in lines:
        line = json.loads(line)
        doc_id = line['doc_id'][0]
        drug_idxs = line['drug_idx']
        relation_label = line['pred']
        # if relation_label != 0:
        if True:
            w.write(json.dumps(
                {"doc_id": doc_id, "drug_idxs": drug_idxs, "relation_label": relation_label},
                ensure_ascii=False))
            w.write("\n")