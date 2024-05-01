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

parser.add_argument('--seed', type=int, default=3, help='random seed')
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
    hyper = 'u_dim_10_1.0_40.0_1.0_5.0'
elif args.model == 'DCEVAE':
    hyper = 'ud_ur_dim_5_5_1.00_40.00_1.00_3.20'
elif args.model == 'CEVAE':
    hyper = 'u_dim_10_1.0_40.0_1.0'
elif args.model == 'CVAE':
    hyper = 'u_dim_10_1.0_40.0_1.0'

dir_gen1 = os.path.join(src_path, args.model, hyper)
if not os.path.exists(dir_gen1):
    os.mkdir(dir_gen1)

dir_gen2 = os.path.join(dir_gen1, str(args.int)+str(args.seed))
if not os.path.exists(dir_gen2):
    os.mkdir(dir_gen2)

model_path = os.path.join(src_path, args.model, 'result_'+str(args.int),'model_'+str(args.seed)+'.pth')

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

attr_path = os.path.join(src_path, 'data', 'celebA', 'list_attr_celeba.txt')

if args.int == 'M':
    whole = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', \
             'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']
    sens = ['Mustache']
    des = []
elif args.int == 'S':
    whole = ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Mustache', 'Wearing_Lipstick']
    sens = ['Smiling']
    des = ['Mouth_Slightly_Open', 'Narrow_Eyes']

test_loader = get_loader(data_df, attr_path, whole, sens, des, mode='test')

# Model Loading
test_model = torch.load(model_path, map_location=args.device)
test_model.to(args.device)
test_model.eval()

def save(file_name, img, folder):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = os.path.join(folder, "%s.jpg" % file_name)
    plt.imsave(f, npimg)

real = []
rec = []
cf = []
n = 0
if args.int == 'M':
    gender = []

    M0 = 772
    M1 = 772

    a0_cf, a1_cf = 0, 0
    g0, g1 = 0, 0
    must = 0
    for cur_data, cur_sens, cur_rest, cur_des in tqdm(test_loader):
        cur_data, cur_sens, cur_rest, cur_des = \
            cur_data.to(args.device), cur_sens.to(args.device), cur_rest.to(args.device), cur_des.to(args.device)
        if args.model == 'mCEVAE' or args.model == 'CEVAE':
            x_fc, x_cf = test_model.image(cur_data, cur_sens, cur_rest, cur_des)
        elif args.model == 'DCEVAE' or args.model == 'CVAE':
            x_fc, x_cf = test_model.image(cur_data, cur_sens)

        for i in range(cur_data.shape[0]):
            # print('mustache', must)
            int = cur_sens[i].cpu().detach()

            if int == 0:
                a1_cf += 1
                if a1_cf > M1:
                    continue
            else:
                must += 1
                a0_cf += 1
                if a0_cf > M1:
                    continue
            save(str(n), cur_data[i, :, :, :].cpu(), real_path)
            save(str(n), x_fc[i, :, :, :].cpu().detach(), rec_path)
            save(str(n), x_cf[i, :, :, :].cpu().detach(), cf_path)

            g = cur_rest[i][2].cpu().detach()
            if g == 1:
                g1 += 1
            else:
                g0 += 1

            real.append([n, int])
            rec.append([n, int])
            cf.append([n, 1-int])
            gender.append([n, g])
            n = n + 1
            #
            # print(a0_cf, a1_cf)

        if a0_cf > M1 and a1_cf > M1:
            break

    # print(real[:10])
    # print(gender[:10])

    real_np = os.path.join(real_path, 'real')
    np.save(real_np, real)
    rec_np = os.path.join(rec_path, 'real')
    np.save(rec_np, rec)
    cf_np = os.path.join(cf_path, 'real')
    np.save(cf_np, cf)
    g_np = os.path.join(real_path, 'gender')
    np.save(g_np, gender)
    # print(a0_cf, a1_cf)
    print('ratio of G0, G1', g0, g1)
elif args.int == 'S':

    for cur_data, cur_sens, cur_rest, cur_des in tqdm(test_loader):
        cur_data, cur_sens, cur_rest, cur_des = \
            cur_data.to(args.device), cur_sens.to(args.device), cur_rest.to(args.device), cur_des.to(args.device)
        if args.model == 'mCEVAE' or args.model == 'CEVAE':
            x_fc, x_cf = test_model.image(cur_data, cur_sens, cur_rest, cur_des)
        elif args.model == 'DCEVAE' or args.model == 'CVAE':
            x_fc, x_cf = test_model.image(cur_data, cur_sens)

        for i in range(cur_data.shape[0]):
            int = cur_sens[i].cpu().detach()

            save(str(n), cur_data[i, :, :, :].cpu(), real_path)
            save(str(n), x_fc[i, :, :, :].cpu().detach(), rec_path)
            save(str(n), x_cf[i, :, :, :].cpu().detach(), cf_path)


            real.append([n, int])
            rec.append([n, int])
            cf.append([n, 1 - int])
            n = n + 1

    real_np = os.path.join(real_path, 'real')
    np.save(real_np, real)
    rec_np = os.path.join(rec_path, 'real')
    np.save(rec_np, rec)
    cf_np = os.path.join(cf_path, 'real')
    np.save(cf_np, cf)

filename = real_np+'.npy'

with open(filename, 'rb') as f:
    lines = [l.decode('utf8', 'ignore') for l in f.readlines()]