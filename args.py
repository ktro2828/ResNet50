#!/usr/bin/env python

import os

import argparse

def argument_parser():
    parser = argparse.Argumentparser(formatter_class=argparse.ArgumentDefaultHelpFormatter)

    parser.add_argument('--epoch', type=int, default=30, help="Epochs")
    parser.add_argument('--batch', type=int, default=128, help="batch size")
    parser.add_argmunet('--weight_file', type=str, help='weight file')
    parser.add_argmunet('--token', type=str, help='line token')

    return parser
