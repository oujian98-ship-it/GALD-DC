import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import importlib

from model.metrics import *
from model.balanced_softmax import balanced_softmax_probs
from model.model_init import model_init
from utilis.test_ft import test_ft
from dataloader.Custom_Dataloader import FeatureDataset

model_paths = {
    'stage2':        'networks.stage_2',
    'none_cifar':    'networks.resnet_cifar',
    'none':          'networks.resnet',
    'tail_cifar':    'networks.resnet_cifar_ensemble',
    'tail':          'networks.resnet_ensemble'
}


def get_metrics(probs, labels, cls_num_list):
    labels = [tensor.cpu().item() for tensor in labels]
    acc = acc_cal(probs, labels, method='top1')

    mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    logging.info('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    print('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    return acc, mmf_acc


# Read from main.py directly: test_set, dset_info, dataset_info, args
def evaluation(test_set, dset_info, dataset_info, args, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model for evaluate
    model = test_ft(datapath=dataset_info["path"],
                            args=args,
                            modelpath=args.eval,
                            crt_modelpath=None, test_cfg=None)
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    #### --------------- evaluate ---------------
    # Get number of batches
    num_batches = len(test_set)
    test_loss, correct, total = 0, 0, 0
    probs, logits_list, labels = [], [], []

    # Since we dont need to update the gradients, we use torch.no_grad()
    with torch.no_grad():
        for data in test_set:
            # Every data instance is an image + label pair
            img, label = data
            # Transfer data to target device
            img = img.to(device)
            label = label.to(device)
            labels.append(label)

            # Compute prediction for this batch
            logit = model(img)

            # Compute the loss
            test_loss += loss_fn(logit, label).item()

            # Store logits for Balanced Softmax
            logits_list.extend(list(logit.cpu().numpy()))

            # Calculate the index of maximum logit as the predicted label
            prob = F.softmax(logit, dim=1)
            probs.extend(list(prob.squeeze().cpu().numpy()))
            pred = prob.argmax(dim=1)

            # Record correct predictions
            correct += (pred == label).type(torch.float).sum().item()
            total += label.size(0)

    # -----------------Balanced Softmax Accuracy-------------------------------#
    probs = np.array(probs)
    logits_array = np.array(logits_list)
    labels = torch.cat(labels)
    _, mmf_acc = get_metrics(probs, labels, dset_info['per_class_img_num'])
    # Gather data and report
    test_loss /= num_batches
    accuracy = correct / total
    logging.info("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))
    print("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))

    bs_probs = balanced_softmax_probs(logits_array, cls_num_list=dset_info['per_class_img_num'])
    bs_acc, mmf_acc_bs = get_metrics(bs_probs, labels, dset_info['per_class_img_num'])

    logging.info("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))
    print("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))

    logging.info("BS (Balanced Softmax) Accuracy is: {}".format(bs_acc))
    print("BS (Balanced Softmax) Accuracy is:", bs_acc)

    logging.info("\n")
    print("\n\n")
    
    return test_loss, accuracy, bs_acc, mmf_acc, mmf_acc_bs
