#!/usr/bin/env python

import os

import argparse
import line_notify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torchvision
import torch.optim as optimizers
from tqdm import tqdm

from args import argument_parser
from dataset import load_cifar10
from model import Resnet50
from utils import cal_loss, plot_result

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argument_parser()
args = parser.args()

# you can change token as fixed value then delete args.token in args.py
client = line_notify.LineNotify(token=args.token)

def main():

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    client.notify("==> Loading the dataset...")
    dataset = load_cifar10(batch=args.batch)
    train_dl = dataset['train']
    test_dl = dataset['test']

    client.notify("==> Loading the model...")
    net = Resnet50(output_dim=10).to(device)
    if args.weight_file is not None:
        weights = torch.load(weight_file)
        net.load_state_dict(weights, strict=False)

    if os.exists('./models') is False:
        os.makedirs('./models')

    optimizer = optimizers.Adam(net.parameters(), lr=1e-4)
    lr_scheduler = optimizers.lr_scheduler.StepLR(optimizer, 5, 0.1)

    history = {
        'epochs': np.arange(1, args.epochs+1),
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    client.notify('==> Start training...')
    for epoch in range(args.epoch):
        train(net, optimizer, train_dl, epoch, history)
        lr_scheduler.step()
        test(net, test_dl, epoch, history)s

    client.notify("==> Training Done")

    plot_result(history)
    client.notify('==> Saved plot')

def train(model,optimizer, train_dl, epoch, history):
    train_step = len(train_dl)
    model.train()

    train_loss = 0
    train_acc = 0

    for (inputs, labels) in tqdm(train_dl, desc='Epoch: {}/{}'.format(epoch+1, args.epoch)):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = cal_loss(outputs, labels)
        train_loss += loss.item()
        train_acc += accuracy_score(labels.tolist(), putputs.argmax(dim=1).tolist())

        optimizer.zero_grad()
        loss.baxkward(
        optimizer.step()

    train_loss /= train_step
    train_acc /= train_step

    client.notify('Epoch: {}/{},\
                    Train Loss: {:.4f},\
                     Train Acc: {:.4f}'.format(epoch+1, args.epoch, train_loss, train_acc))

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

def test(model, optimizer, test_dl, epoch, history):
    test_step = len(test_dl)
    model.eval()

    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for (inputs, labels) in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = cal_loss(outputs, labels)
            test_loss += loss.item()
            test_acc += accuracy_score(labels.tolist(), outputs.argmax(dim=1).tolist())

        test_loss /= test_step
        test_acc /= test_step
        client.notify('Epoch: {}/{}, \
                        Valid Loss: {:.4f}, \
                        Valid Acc: {:.4f}'.format(epoch+1, args.epoch, test_loss, test_acc))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

if __name__ == '__main__':
    main()
