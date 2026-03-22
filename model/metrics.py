import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from scipy.stats import entropy
from typing import Type, Any, Callable, Union, List, Optional

from utilis.utils import normalized


def acc_cal(logits, label: List[int], method: str = 'top1'):
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]
        correct = np.sum([i == j for i, j in zip(label_pred, label)])
        total = len(label)

    result = correct / total * 100
    return round(float(result), 4)


def ClsAccCal(logits, label: List[int], method: str = 'top1'):
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]

        label_set = list(set(label))
        correct = np.zeros(len(label_set))
        total = np.zeros(len(label_set)) + 1e-6
        for i, j in zip(label_pred, label):
            correct[j] += (i == j)
            total[j] += 1

    result = np.array(correct) / np.array(total) * 100
    return result.round(4).tolist()


def mmf_acc_cal(logits, label: List[int], class_num_list: List[int], method: str = 'top1'):
    """
    Many-Medium-Few shot accuracy calculation
    
    Standard Long-Tail Recognition partitioning:
    - CIFAR-10: Top 3 classes (Many), Middle 4 classes (Medium), Bottom 3 classes (Few)
    - CIFAR-100/ImageNet: Fixed thresholds (>100: Many, 20-100: Medium, <20: Few)
    """
    correct = np.zeros(3)
    total = np.zeros(3) + 1e-6
    
    num_classes = len(class_num_list)
    
    # 根据类别数判断数据集类型
    if num_classes == 10:  # CIFAR-10
        sorted_indices = np.argsort(class_num_list)[::-1]
        mmf_id = [0] * num_classes
        
        for i, cls_idx in enumerate(sorted_indices):
            if i < 3:
                mmf_id[cls_idx] = 0  # Many
            elif i < 7:
                mmf_id[cls_idx] = 1  # Medium
            else:
                mmf_id[cls_idx] = 2  # Few
                
    else:  # CIFAR-100 / ImageNet
        # Many: 样本数 > 100, Medium: 20 ≤ 样本数 ≤ 100, Few: 样本数 < 20
        mmf_id = []
        for num in class_num_list:
            if num < 20:
                mmf_id.append(2)  # Few
            elif num <= 100:  
                mmf_id.append(1)  # Medium
            else:
                mmf_id.append(0)  # Many
            
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]
        total_correct = 0
        total_samples = 0
        
        for i, j in zip(label_pred, label):
            is_correct = (i == j)
            correct[mmf_id[j]] += is_correct
            total[mmf_id[j]] += 1
            # 累计总体准确率
            total_correct += is_correct
            total_samples += 1

    result = np.array(correct) / np.array(total) * 100
    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    result = result.tolist() + [round(overall_acc, 4)]
    return result



# Original Code from https://github.com/gpleiss/temperature_scaling
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        softmaxes = torch.Tensor(softmaxes)
        labels = torch.LongTensor(labels)

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# Expected Calibration Error (ECE) numpy implementation
def ECECal(probs, labels: List[int], bins: int = 15, sum=True):
    conf = np.max(probs, axis=-1)
    acc = np.argmax(probs, axis=-1) == labels

    bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
    # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
    split_idx = np.digitize(conf, bin_upper_bounds, right=True)
    data_len = len(split_idx)

    ece = np.zeros(bins)
    for i in range(bins):
        idx = split_idx == i
        if np.sum(idx) > 0:
            bin_avg_conf = np.mean(conf[idx])
            bin_avg_acc = np.mean(acc[idx])
            bin_prob = np.sum(idx) / data_len

            ece[i] = np.abs(bin_avg_acc - bin_avg_conf) * bin_prob

        # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

    return ece.sum()


# Reliability Values numpy implementation
def ECEAccCal(probs, labels, bins: int = 15):
    conf = np.max(probs, axis=-1)
    acc = np.argmax(probs, axis=-1) == labels

    bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
    # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
    split_idx = np.digitize(conf, bin_upper_bounds, right=True)
    data_len = len(labels)

    bin_acc = np.zeros(bins)
    bin_prob = np.zeros(bins)
    for i in range(bins):
        idx = split_idx == i
        if np.sum(idx) > 0:
            bin_avg_acc = np.mean(acc[idx])
            bin_acc[i] = bin_avg_acc
            bin_prob[i] = np.sum(idx) / data_len

        # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

    return bin_acc, bin_prob


def BierCal(probs, labels: List[int]):
    probs_correct = np.array([x[i] for x, i in zip(probs, labels)])
    return np.mean(np.power(probs_correct - 1, 2))


def EntropyCal(prob):
    result = np.mean(entropy(prob, axis=-1))
    return result


def SCECal(probs, labels: List[int], bins: int = 15):
    cls = list(set(labels))
    conf_all = np.max(probs, axis=-1)
    acc_all = np.argmax(probs, axis=-1) == labels

    conf_group, acc_group = group_data((conf_all, acc_all), labels, cls)

    eces = []
    for conf, acc in zip(conf_group, acc_group):
        conf = np.array(conf)
        acc = np.array(acc)
        bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
        # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
        split_idx = np.digitize(conf, bin_upper_bounds, right=True)
        data_len = len(split_idx)

        ece = np.zeros(bins)
        for i in range(bins):
            idx = split_idx == i
            if np.sum(idx) > 0:
                bin_avg_conf = np.mean(conf[idx])
                bin_avg_acc = np.mean(acc[idx])
                bin_prob = np.sum(idx) / data_len

                ece[i] = np.abs(bin_avg_acc - bin_avg_conf) * bin_prob

            # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

        eces.append(ece.sum())

    return np.mean(eces)


def group_data(data: tuple, label: List[int], cls: List[int]):
    # the idx should be output of np.unique(data), which is sorted.
    # assert len(set(idx)) == len(idx)
    tuple_num = len(data)
    data_group = [[[] for _ in cls] for _ in range(tuple_num)]
    for i, l in enumerate(label):
        for j in range(tuple_num):
            data_group[j][l].append(data[j][i])

    return data_group
