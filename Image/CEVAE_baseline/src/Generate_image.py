from dataloader import get_loader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu')
parser.add_argument('--batch_size', type=int, default=64, help='number of gpu')

parser.add_argument('--u_dim', type=int, default=10, help='dimension of u')
parser.add_argument('--beta1', type=float, default=1, help='beta1')
parser.add_argument('--beta2', type=float, default=40, help='beta2')
parser.add_argument('--beta3', type=float, default=1, help='beta3')

parser.add_argument('--int', type=str, default='M', help='intervention variable) M: mustache; S: smiling')

parser.add_argument('--a0', type=int, default=500)
parser.add_argument('--a1', type=int, default=500)

parser.add_argument('--retrain', type=bool, default=False, help='retrain or not')
args = parser.parse_args()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
hyp = "u_dim_{:d}_{:.1f}_{:.1f}_{:.1f}"\
                              .format(args.u_dim, args.beta1, args.beta2, args.beta3)

# Image Folder (Create)
src_path = os.path.dirname(os.path.realpath('__file__'))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
data_path = os.path.abspath(os.path.join(src_path, os.pardir))

data_df = os.path.join(data_path, 'data', 'celebA', 'images')
save_path = os.path.join(src_path, 'Generated_Dataset')
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = os.path.join(save_path, hyp)
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_path = os.path.join(save_path, args.int + str(args.seed))
if not os.path.exists(save_path):
    os.mkdir(save_path)

real_path = os.path.join(save_path, 'real')
rec_path = os.path.join(save_path, 'rec')
cf_path = os.path.join(save_path, 'cf')

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
src_path = os.path.dirname(os.path.realpath('__file__'))
src_path = os.path.abspath(os.path.join(src_path, os.pardir))
result_path = os.path.join(src_path, "result")
model_folder = os.path.join(result_path, hyp)
model_folder = os.path.join(model_folder, str(args.int)+str(args.seed))
model_path = os.path.join(model_folder, 'model.pth')
test_model = torch.load(model_path, map_location=args.device)
test_model.to(args.device)
test_model.eval()

def save(file_name, img, folder):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = os.path.join(folder, "%s.png" % file_name)
    plt.imsave(f, npimg)

n = 0
a0_cf = 0
a1_cf = 0

real = []
rec = []
cf = []

for cur_data, cur_sens, cur_rest, cur_des in tqdm(test_loader):
    cur_data, cur_sens, cur_rest, cur_des = \
        cur_data.to(args.device), cur_sens.to(args.device), cur_rest.to(args.device), cur_des.to(args.device)
    x_fc, x_cf = test_model.image(cur_data, cur_sens, cur_rest, cur_des)

    for i in range(cur_data.shape[0]):
        int = cur_sens[i].cpu().detach()

        if int == 0:
            a1_cf += 1
            if a1_cf > args.a1:
                continue
        else:
            a0_cf += 1
            if a0_cf > args.a0:
                continue

        save(str(n), cur_data[i, :, :, :].cpu(), real_path)
        save(str(n), x_fc[i, :, :, :].cpu().detach(), rec_path)
        save(str(n), x_cf[i, :, :, :].cpu().detach(), cf_path)
        n = n + 1

        real.append([n, int])
        rec.append([n, int])
        cf.append([n, 1-int])

        if a0_cf > args.a0 and a1_cf > args.a1:
            break

    if a0_cf > args.a0 and a1_cf > args.a1:
        real_np = os.path.join(real_path, 'real')
        np.save(real_np, real)
        rec_np = os.path.join(rec_path, 'real')
        np.save(rec_np, rec)
        cf_np = os.path.join(cf_path, 'real')
        np.save(cf_np, cf)
        break

print(a0_cf)
print(a1_cf)
filename = real_np+'.npy'

with open(filename, 'rb') as f:
    lines = [l.decode('utf8', 'ignore') for l in f.readlines()]
    print(lines)
