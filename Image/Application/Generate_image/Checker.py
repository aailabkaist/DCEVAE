from dataloader import get_loader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
import argparse
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mCEVAE', help='mCEVAE')
parser.add_argument('--int', type=str, default='M', help='intervention variable) M: mustache; S: smiling')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu')
parser.add_argument('--batch_size', type=int, default=64, help='number of gpu')

args = parser.parse_args()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)

# Image Folder (Create)
src_path = os.path.dirname(os.path.realpath('__file__'))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
data_df = os.path.join(src_path, 'data', 'celebA', 'images')

# local
src_path2 = os.path.abspath(os.path.join(src_path, os.pardir))
data_df = os.path.join(src_path2, 'data', 'celebA', 'images')

if args.model == 'mCEVAE':
    hyper = 'u_dim_10_1.0_40.0_10.0_10.0'

dir_gen1 = os.path.join(src_path, args.model, hyper)
if not os.path.exists(dir_gen1):
    os.mkdir(dir_gen1)

dir_gen2 = os.path.join(dir_gen1, str(args.int)+str(args.seed))
if not os.path.exists(dir_gen2):
    os.mkdir(dir_gen2)

# Load Data (Create)
real_path = os.path.join(dir_gen2, 'real')
rec_path = os.path.join(dir_gen2, 'rec')
cf_path = os.path.join(dir_gen2, 'cf')

if not os.path.exists(real_path):
    os.mkdir(real_path)
if not os.path.exists(rec_path):
    os.mkdir(rec_path)
if not os.path.exists(cf_path):
    os.mkdir(cf_path)

real_np = os.path.join(real_path, 'real')
filename = real_np+'.npy'
g = os.path.join(real_path, 'gender')
filename2 = g+'.npy'

data = np.load(filename)
data2 = np.load(filename2)

n0, n1 = 0,0
for i in range(data.shape[0]):
    #print(i, data[i,1], data2[i,1])
    if data[i,1] == 1:#data2[i,1] == 1:
        n1 += 1
    else:
        n0 += 1
print(n0)
print(n1)