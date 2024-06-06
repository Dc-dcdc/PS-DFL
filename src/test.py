import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


def tes_img(net_g, datatest, args):
    net_g.eval()
    
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):

            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        return accuracy, test_loss


def tes_img_2(net_g, datatest, args):
    net_g.eval()

    test_loss = 0
    correct = 0
    correct_test = 0
    correct_every_fault = [[0 for i in range(7)] for j in range(7)]
    for i in range(0, 7):

        data_loader = DataLoader(datatest[i], batch_size=args.bs)
        l = len(data_loader)
        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):

                target = torch.tensor([i for j in range(len(target))])
                if args.gpu != -1:
                    data, target = data.to(args.device), target.to(args.device)

                log_probs = net_g(data)

                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

                y_pred = log_probs.data.max(1, keepdim=True)[1]

                for j in range(len(y_pred)):
                    correct_every_fault[i][y_pred[j].item()] += round(float(1 / len(datatest[i])), 5)
                    # correct_every_fault[i][y_pred[j].item()] += 1          #用于计算precision  recall  F1
                    if y_pred[j].item() == i:
                        correct_test += 1
    len_datatest = 0
    for i in range(0, 7):
        len_datatest += len(datatest[i])

    precision=0
    for i in range(0, args.num_classes):
        if sum(correct_every_fault[i]) == 0:
            precision = 0
        else:
            precision +=  (correct_every_fault[i][i]/sum(correct_every_fault[i]))/args.num_classes

    recall = 0
    for i in range(0, args.num_classes):
        column_sum = 0
        for j in range(0, args.num_classes):
            column_sum += correct_every_fault[j][i]
        if column_sum == 0:
            recall += 0
        else:
            recall += (correct_every_fault[i][i]/column_sum)/args.num_classes
    if precision == 0 and recall == 0 :
        F1 = 0  # F1
    else:
       F1 = 200*(precision*recall)/(precision+recall) # F1


    test_loss /= len_datatest

    accuracy = 100.00 * correct_test / len_datatest
    return accuracy, test_loss, correct_every_fault,precision ,recall,F1



