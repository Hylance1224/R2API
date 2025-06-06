# -*- conding: utf-8 -*-
"""
@File   : metric.py
@Time   : 2021/1/9
@Author : yhduan
@Desc   : None
"""

import numpy as np
from sklearn.metrics import ndcg_score

def metric_string(batch_target, batch_pred, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = 1
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    _target, _pred = batch_target, batch_pred
        #_target = _target.nonzero().squeeze().tolist()

    if isinstance(_target, int):
        _target = [_target]

    #_pred = _pred.argsort().tolist()
    for i, k in enumerate(top_k_list):
        _ndcg[i] += ndcg(_target, _pred[:k])
        _recall[i] += recall(_target, _pred[:k])
        _map[i] += ap(_target, _pred[:k])
        _pre[i] += precision(_target, _pred[:k])
    return _ndcg/batch_size, _recall/batch_size, _map/batch_size, _pre/batch_size

def metric(batch_target, batch_pred, top_k_list):
    def removeDump(list):
        dic = {}
        for idx, item in enumerate(np.nditer(list)):
            if dic.get(item.tolist()) == None:
                dic[item.tolist()] = idx
            else:
                list[idx] = 1
        return list
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.shape[0]
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        #_target = _target.nonzero().squeeze().tolist()

        _pred = removeDump(_pred)
        _target = [targ for targ in _target if targ >= 4]
        _pred = [targ for targ in _pred if targ >= 4]
        if isinstance(_target, int):
            _target = [_target]
        #_pred = _pred.argsort().tolist()


        #hit_set = list(set(_target) & set(_pred[:10]))
        # print(len(_pred))

        # > 2的 计算
        #if len(_target) <= 2:
        #    batch_size = batch_size-1
        #    continue


        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

    if batch_size == 0:
        return 0, 0, 0, 0
    return _ndcg/batch_size, _recall/batch_size, _map/batch_size, _pre/batch_size


def metric2(target, pred, top_k_list):
    target = target.nonzero().squeeze().tolist()
    if isinstance(target, int):
        target = [target]
    pred = pred.argsort(descending=True).tolist()
    if isinstance(pred, int):
        target = [pred]
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    for i, k in enumerate(top_k_list):
        _ndcg[i] += ndcg(target, pred[:k])
        _recall[i] += recall(target, pred[:k])
        _map[i] += ap(target, pred[:k])
        _pre[i] += precision(target, pred[:k])

    return _ndcg, _recall, _map, _pre

def ndcg(target, pred):
    dcg = 0
    c = 0
    for i in range(1, len(pred) + 1):
        rel = 0
        if pred[i - 1] in target:
            rel = 1
            c += 1
        dcg += (np.power(2, rel) - 1) / np.log2(i + 1)
    if c == 0:
        return 0
    idcg = 0
    for i in range(1, c + 1):
        idcg += (1 / np.log2(i + 1))
    return dcg / idcg


def ap(target, pred):
    p_at_k = np.zeros(len(pred))
    c = 0
    for i in range(1, len(pred) + 1):
        rel = 0
        if pred[i - 1] in target:
            rel = 1
            c += 1
        p_at_k[i - 1] = rel * c / i
    if c == 0:
        return 0.0
    else:
        return np.sum(p_at_k) / c


def precision(target, pred):
    hit_set = list(set(target) & set(pred))
    if len(pred) == 0:
        return 0
    return len(hit_set) / float(len(pred))


def recall(target, pred):
    hit_set = list(set(target) & set(pred))
    if len(target) == 0:
        return 0
    return len(hit_set) / float(len(target))
