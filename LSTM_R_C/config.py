# -*- coding: utf-8 -*-
import argparse
import pdb
def all_args(parser):
    parser.add_argument('-n_layer', type=int, default=2,
                       help='LSTM layer number')
    parser.add_argument('-h_dim', type=int, default=500,
                       help='LSTM hidden size')
    parser.add_argument('-in_dim', type=int, default=500,
                       help='embedding dim')
    parser.add_argument('-lr', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('-epochs', type=int, default=300,
                       help='epochs')
    parser.add_argument('-batch_size', type=int, default=32,
                       help='batch size')
    parser.add_argument('-data', type=str, default='./data',
                       help='data root')
    parser.add_argument('-model', type=str, default='',
                       help='load model path')
    parser.add_argument('-save_model', type=str, default='./model',
                       help='save model directory')
    parser.add_argument('-save_frequence', type=int, default=1,
                       help='how many epoch we save model')
    parser.add_argument('-gpu', type=bool, default=0,
                       help='use gpu, -1 is cpu')


parser = argparse.ArgumentParser()
all_args(parser)
opts = parser.parse_args()

#other parameters
opts.maxlength = 985
opts.vocab_size = 1643

#回归数
opts.key_num    = 8

#分类数
opts.class_num  = 4398

if opts.gpu >= 0:
    opts.gpu = True
else:
    opts.gpu = False