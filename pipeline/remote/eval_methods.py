import numpy as np


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):

    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0): i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):

    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 1e-5)
    recall    = TP / (TP + FN + 1e-5)
    f1        = 2 * precision * recall / (precision + recall + 1e-5)
    return f1, precision, recall, TP, TN, FP, FN


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):

    if step_num is None or end is None:
        end = start
        step_num = 1

    search_range       = end - start
    search_lower_bound = start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)

    threshold = search_lower_bound
    m   = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0

    for i in range(step_num):
        threshold += search_range / float(step_num)
        target, latency = _calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m   = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1":        m[0],
        "precision": m[1],
        "recall":    m[2],
        "TP":        m[3],
        "TN":        m[4],
        "FP":        m[5],
        "FN":        m[6],
        "threshold": m_t,
        "latency":   m_l,
    }


def _calc_seq(score, label, threshold):
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency

