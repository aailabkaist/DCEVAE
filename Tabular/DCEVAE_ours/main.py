import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='number of gpu')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') #(1e-3)/2
parser.add_argument('--loss_fn', type=str, default='BCE', help='loss function')
parser.add_argument('--break_epoch', type=int, default=30, help='break epoch')
parser.add_argument('--act_fn', type=str, default='ReLU', help='activation function')

parser.add_argument('--a_y', type=float, default=1,  help='hyper-parameter for y')
parser.add_argument('--a_r', type=float, default=1, help='hyper-parameter for x_r')
parser.add_argument('--a_d', type=float, default=1, help='hyper-parameter for x_d')
parser.add_argument('--a_a', type=float, default=1, help='hyper-parameter for a recon from z')
parser.add_argument('--a_f', type=float, default=0.0, help='hyper-parameter for fairness')
parser.add_argument('--a_h', type=float, default=0.4, help='hyper-parameter for independency')
parser.add_argument('--u_kl',  type=float, default=1, help='hyper-parameter for u_kl')

parser.add_argument('--ur_dim', type=int, default=3, help='dim of ur')
parser.add_argument('--ud_dim', type=int, default=4, help='dim of ud')
parser.add_argument('--h_dim', type=int, default=100, help='dim of ud')
parser.add_argument('--run', type=int, default=1, help='# of run')

parser.add_argument('--gpu', type=int, default=0, help='number of gpu')

parser.add_argument('--retrain', type=bool, default=False, help='True/False')
parser.add_argument('--test', type=bool, default=True, help='True/False')
parser.add_argument('--tSNE', type=bool, default=True, help='True/False')
parser.add_argument('--clf', type=bool, default=True, help='True/False')
parser.add_argument('--balance', type=bool, default=False, help='True/False')
parser.add_argument('--early_stop', type=bool, default=True, help='True/False')

parser.add_argument('--rep', type=int, default=1, help='number rep')

args = parser.parse_args()

import torch
import os
import logging.handlers

from utils import setup_logger, make_loader, make_seperate_loader, make_whole_adult_loader, make_balancing_loader
from model import DCEVAE
from train import train
from test import test, generate_data
from fair_classifier import fair_whole_classifier, fair_seperate_classifier, baseline_classifier

def main(args):
    args.seed = args.run
    '''GPU setting'''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''Save path setting & mkdir'''
    src_path = os.path.dirname(os.path.realpath('__file__'))
    result_path = os.path.join(src_path, 'result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    args.save_path = os.path.join(result_path, \
                                  'a_r_{:s}_a_d_{:s}_a_y_{:s}_a_h_{:s}_a_f_{:s}_u_{:s}_ur_{:d}_ud_{:d}_run_{:d}'\
                                  .format(str(args.a_r), str(args.a_d), str(args.a_y), str(args.a_h), str(args.a_f), \
                                          str(args.u_kl), args.ur_dim, args.ud_dim, args.run))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    train_dir = os.path.join(args.save_path, 'train_log.txt')
    test_dir = os.path.join(args.save_path, 'test_log.txt')
    clf_dir = os.path.join(args.save_path, 'classifier_log.txt')

    '''Set Logger'''
    if args.retrain == False:
        setup_logger('log1', train_dir, filemode='a')
    else:
        setup_logger('log1', train_dir)
    setup_logger('log2', test_dir)
    setup_logger('log3', clf_dir)
    logger_1 = logging.getLogger('log1')
    logger_2 = logging.getLogger('log2')
    logger_3 = logging.getLogger('log3')

    logger_1.info(args)
    logger_2.info(args)
    logger_3.info(args)

    logger_1.info('This code uses ' + args.device)

    '''Load Dataset'''
    train_df = open(os.path.join(src_path, '../data/cfgan/list_attr_adult.txt'))
    if args.balance == True:
        train_loader, valid_loader, test_loader, input_dim = make_balancing_loader(train_df, args)
    else:
        train_loader, valid_loader, test_loader, input_dim = make_loader(train_df, args)

    args.input_dim = input_dim
    print(input_dim)
    model = DCEVAE(r_dim=input_dim['r'], d_dim=input_dim['d'], sens_dim=input_dim['a'], label_dim=input_dim['y'], \
                  args=args).to(args.device)

    '''Train Start'''
    model_path = os.path.join(args.save_path, 'model.pth')
    if not os.path.exists(model_path) or args.retrain == True:
        print('Train Start')
        train(model, train_loader, valid_loader, args, logger_1)

    '''Test Start'''
    if args.test == True:
        print('Test Start')
        test(test_loader, args, logger_2)

        '''Generate Data'''
        print('Generate Data')
        generate_data(train_loader, args, 'train')
        generate_data(valid_loader, args, 'valid')
        generate_data(test_loader, args, 'test')

    '''Fair Classifier'''
    if args.clf == True:
        baseline_classifier(train_loader, valid_loader, test_loader, args, logger_3)
        fair_whole_classifier(train_loader, valid_loader, test_loader, args, logger_3)

main(args)
