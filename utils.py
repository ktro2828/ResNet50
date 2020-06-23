#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch.nn as nn


def cal_loss(output, label):
    crietion = nn.NLLLoss()
    return crietion(output, label)

def plot_result(history):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

    ax1.set_xlabel('epoch')
    a1.set_ylabel('loss')
    ax1.plot(history['epochs'], history['train_loss'], label='train loss')
    ax1.plot(history['epochs'], history['test_loss'], label='test loss')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.plot(history['epochs'], history['train_acc'], label='train accuracy')
    ax2.plot(history['epochs'], history['test_acc'], label='test accuracy')
    ax2.legend(loc='lower right')

    plt.show()
    fig.savefig('result_graph.png')
    plt.close()
