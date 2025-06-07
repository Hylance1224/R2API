import json
import math

def calculate_Precision(recommend_list, removed_tpl_list, top_n):
    hit = 0.0
    if len(recommend_list) == 0:
        return hit
    for i in range(min(top_n, len(recommend_list))):
        if recommend_list[i] in removed_tpl_list:
            hit += 1
    return hit / top_n

def calculate_Recall(recommend_list, removed_tpl_list, top_n):
    hit = 0.0
    if len(recommend_list) == 0 or len(removed_tpl_list) == 0:
        return 0
    for i in range(min(top_n, len(recommend_list))):
        if recommend_list[i] in removed_tpl_list:
            hit += 1
    return hit / len(removed_tpl_list)

def calculate_NDCG(recommend_list, removed_tpl_list, top_n):
    DCG = 0.0
    IDCG = 0.0
    for i in range(min(top_n, len(recommend_list))):
        if recommend_list[i] in removed_tpl_list:
            DCG += 1 / math.log2(i + 2)
    for i in range(min(top_n, len(removed_tpl_list))):
        IDCG += 1 / math.log2(i + 2)
    return DCG / IDCG if IDCG > 0 else 0

def calculate_AP(recommend_list, removed_tpl_list, top_n):
    cor_list = []
    if len(recommend_list) == 0:
        return 0
    for i in range(min(top_n, len(recommend_list))):
        if recommend_list[i] in removed_tpl_list:
            cor_list.append(1.0)
        else:
            cor_list.append(0.0)
    sum_cor_list = sum(cor_list)
    if sum_cor_list == 0:
        return 0

    summary = 0.0
    for i in range(len(cor_list)):
        summary += (sum(cor_list[:i + 1]) / (i + 1)) * cor_list[i]
    return summary / sum_cor_list

# def evaluate_metrics_from_file(greedy_file_path, top_n_list=[1, 5, 10], total_api_count=940):
#     removed_apis_list = []
#     predict_apis_list = []
#
#     with open(greedy_file_path, 'r') as f:
#         for line in f:
#             item = json.loads(line)
#             removed = [api for api in item["remove_apis"] if api != 1]  # 1 是 padding
#             predicted = item["predict_apis"]
#             removed_apis_list.append(removed)
#             predict_apis_list.append(predicted)
#
#     metrics_result = {n: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'map': 0.0, 'coverage': 0.0} for n in top_n_list}
#     count = len(removed_apis_list)
#     print(count)
#     for removed, predicted in zip(removed_apis_list, predict_apis_list):
#         for n in top_n_list:
#             topn_preds = predicted[:n]
#             metrics_result[n]['precision'] += calculate_Precision(topn_preds, removed, n)
#             metrics_result[n]['recall'] += calculate_Recall(topn_preds, removed, n)
#             metrics_result[n]['ndcg'] += calculate_NDCG(topn_preds, removed, n)
#             metrics_result[n]['map'] += calculate_AP(topn_preds, removed, n)
#
#     for n in top_n_list:
#         appeared_apis = set()
#         for predicted in predict_apis_list:
#             appeared_apis.update(predicted[:n])
#         coverage = len(appeared_apis) / total_api_count
#         metrics_result[n]['coverage'] = coverage
#
#     for n in top_n_list:
#         metrics_result[n]['precision'] /= count
#         metrics_result[n]['recall'] /= count
#         metrics_result[n]['ndcg'] /= count
#         metrics_result[n]['map'] /= count
#
#     return metrics_result
# better = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5015, 5017, 5018, 5019, 5020, 5021, 5022, 5023, 5024, 5027, 5028, 5032, 5033, 5036, 5037, 5038, 5039, 5040, 5042, 5044, 5045, 5046, 5049]

def evaluate_metrics_from_file(greedy_file_path, top_n_list=[1, 5, 10], total_api_count=940):
    filtered_data = []

    with open(greedy_file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            mashup_name = item.get("mashup_name", "")
            filtered_data.append(item)
            # if mashup_name.isdigit() and int(mashup_name) in better:
            #     filtered_data.append(item)

    removed_apis_list = [[api for api in item['remove_apis'] if api != 1] for item in filtered_data]
    predict_apis_list = [item['predict_apis'] for item in filtered_data]

    metrics_result = {n: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'map': 0.0, 'coverage': 0.0} for n in top_n_list}
    count = len(removed_apis_list)
    print(f"Filtered mashups count: {count}")

    for removed, predicted in zip(removed_apis_list, predict_apis_list):
        for n in top_n_list:
            topn_preds = predicted[:n]
            metrics_result[n]['precision'] += calculate_Precision(topn_preds, removed, n)
            metrics_result[n]['recall'] += calculate_Recall(topn_preds, removed, n)
            metrics_result[n]['ndcg'] += calculate_NDCG(topn_preds, removed, n)
            metrics_result[n]['map'] += calculate_AP(topn_preds, removed, n)

    for n in top_n_list:
        appeared_apis = set()
        for predicted in predict_apis_list:
            appeared_apis.update(predicted[:n])
        coverage = len(appeared_apis) / total_api_count
        metrics_result[n]['coverage'] = coverage

    for n in top_n_list:
        metrics_result[n]['precision'] /= count
        metrics_result[n]['recall'] /= count
        metrics_result[n]['ndcg'] /= count
        metrics_result[n]['map'] /= count

    return metrics_result


# 用法
result = evaluate_metrics_from_file("output/decode_greedy29.jsonl", top_n_list=[3, 5, 10, 20])

for k, v in result.items():
    print(
        f"@{k} → & {v['precision']:.4f} & {v['recall']:.4f} & {v['map']:.4f} & {v['ndcg']:.4f} & {v['coverage']:.4f}")

