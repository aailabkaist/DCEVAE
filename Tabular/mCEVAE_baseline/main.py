import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='number of gpu')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') #(1e-3)/2
parser.add_argument('--loss_fn', type=str, default='BCE', help='loss function')
parser.add_argument('--break_epoch', type=int, default=50, help='break epoch')
parser.add_argument('--act_fn', type=str, default='ReLU', help='activation function')

parser.add_argument('--a_y', type=float, default=1, help='hyper-parameter for y')
parser.add_argument('--a_x', type=float, default=1, help='hyper-parameter for x')
parser.add_argument('--a_f', type=float, default=0, help='hyper-parameter for fairness')
parser.add_argument('--mmd', type=float, default=0.1, help='hyper-parameter for independency')
parser.add_argument('--mmd_a', type=float, default=0.5, help='hyper-parameter for u_kl')
parser.add_argument('--lambda_cf', type=float, default=1, help='hyper-parameter for cf')
parser.add_argument('--lambda_clp', type=float, default=1, help='hyper-parameter for clp')

parser.add_argument('--u_dim', type=int, default=7, help='dim of u')
parser.add_argument('--run', type=int, default=1, help='# of run')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--gpu', type=int, default=0, help='number of gpu')
parser.add_argument('--rep', type=int, default=0, help='number of gpu')

parser.add_argument('--retrain', type=bool, default=True, help='True/False')
parser.add_argument('--debug', type=bool, default=True, help='True/False')
parser.add_argument('--test', type=bool, default=True, help='True/False')
parser.add_argument('--tSNE', type=bool, default=False, help='True/False')
parser.add_argument('--clf', type=bool, default=True, help='True/False')
parser.add_argument('--balance', type=bool, default=False, help='True/False')
parser.add_argument('--early_stop', type=bool, default=True, help='True/False')

parser.add_argument('--dataset', type=str, default='adult', help='adult')

args = parser.parse_args()

import torch
import os
import logging.handlers

from utils import setup_logger, make_loader, make_seperate_loader, make_whole_adult_loader, make_balancing_loader
from model import CEVAE
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

    args.save_path = os.path.join(result_path, 'a_x_{:s}_a_y_{:s}_a_f_{:s}_mmd_{:s}_mmd_a_{:s}_run_{:d}'.format\
        (str(args.a_x), str(args.a_y), str(args.a_f), str(args.mmd), str(args.mmd_a), args.run))
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

    # setting = 'n_epochs: {:d}, batch_size: {:d} a_y: {:.4f}, a_f: {:.4f}'.format(args.n_epochs, args.batch_size, args.a_y, args.a_f)
    # logger_1.info(setting)
    logger_1.info('This code uses ' + args.device)

    '''Load Dataset'''
    train_df = open(os.path.join(src_path, '../data/cfgan/list_attr_adult.txt'))
    if args.balance == True:
        train_loader, valid_loader, test_loader, input_dim = make_balancing_loader(train_df, args)
    else:
        train_loader, valid_loader, test_loader, input_dim = make_loader(train_df, args)

    args.input_dim = input_dim
    model = CEVAE(x_dim=input_dim['x'], o_dim=input_dim['o'], sens_dim=input_dim['a'], label_dim=input_dim['y'], args=args).to(args.device)

    '''Train Start'''
    model_path = os.path.join(args.save_path, 'model.pth')
    if not os.path.exists(model_path) or args.retrain == True:
        print('Train Start')
        train(model, train_loader, valid_loader, args, logger_1)

    '''Test Start'''
    if args.test == True:
        print('Test Start')
        # logger_2.info(setting)
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