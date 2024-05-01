import torch
# entrypoints = torch.hub.list('pytorch/vision:v0.5.0', force_reload=True)
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import numpy as np
import pandas as pd
import csv

np.random.seed(0)


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, whole, selected_attrs, des_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.whole = whole
        self.selected_attrs = selected_attrs
        self.des_attrs = des_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.valid_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == 'valid':
            self.num_images = len(self.valid_dataset)
        else:
            self.num_images = len(self.test_dataset)

        print(mode, self.num_images)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        if False:
            df1 = pd.read_csv('info/list_attr_celeba.txt', sep="\s+")
            df2 = pd.read_csv('info/list_eval_partition.txt', sep="\s+", header=None)
            df2.columns = ['Filename', 'Partition']
            df2 = df2.set_index('Filename')
            df3 = df1.merge(df2, left_index=True, right_index=True)
            df3.to_csv('info/celeba-whole.csv')
            df3 = pd.read_csv('info/celeba-whole.csv', index_col=0)

            df3.loc[df3['Partition'] == 0].to_csv('info/celeba-train.csv')
            df3.loc[df3['Partition'] == 1].to_csv('info/celeba-valid.csv')
            df3.loc[df3['Partition'] == 2].to_csv('info/celeba-test.csv')

        if self.mode == 'train':
            attr_path_list = ['info/celeba-train.csv']
        elif self.mode == 'valid':
            attr_path_list = ['info/celeba-valid.csv']
        else:
            attr_path_list = ['info/celeba-test.csv']

        for attr_path in attr_path_list:

            f = open(attr_path, 'r')
            rdr = csv.reader(f)
            temp = 0
            for line in rdr:
                if temp == 0:
                    all_attr_names = line[1:]
                    for i, attr_name in enumerate(all_attr_names):
                        self.attr2idx[attr_name] = i
                        self.idx2attr[i] = attr_name
                else:
                    filename = line[0]
                    values = line[1:]
                    label = []
                    for attr_name in self.selected_attrs:
                        idx = self.attr2idx[attr_name]
                        label.append(values[idx] == '1')

                    des = []
                    for attr_name in self.des_attrs:
                        idx = self.attr2idx[attr_name]
                        des.append(values[idx] == '1')

                    rest = []
                    for attr_name in all_attr_names:
                        if attr_name in self.des_attrs + self.selected_attrs:
                            continue
                        if self.whole != 'whole':
                            if attr_name not in self.whole:
                                continue
                        idx = self.attr2idx[attr_name]
                        rest.append(values[idx] == '1')

                    if 'train' in attr_path:
                        self.train_dataset.append([filename, label, rest, des])
                    elif 'valid' in attr_path:
                        self.valid_dataset.append([filename, label, rest, des])
                    elif 'test' in attr_path:
                        self.test_dataset.append([filename, label, rest, des])
                temp += 1

            perm_list = np.random.permutation(temp-1)
            temp = 0
            f = open(attr_path, 'r')
            rdr = csv.reader(f)

            for line in rdr:
                if temp == 0:
                    all_attr_names = line[1:]
                    for i, attr_name in enumerate(all_attr_names):
                        self.attr2idx[attr_name] = i
                        self.idx2attr[i] = attr_name
                else:
                    filename = line[0]
                    values = line[1:]
                    label = []
                    for attr_name in self.selected_attrs:
                        idx = self.attr2idx[attr_name]
                        label.append(values[idx] == '1')

                    des = []
                    for attr_name in self.des_attrs:
                        idx = self.attr2idx[attr_name]
                        des.append(values[idx] == '1')

                    rest = []
                    for attr_name in all_attr_names:
                        if attr_name in self.des_attrs + self.selected_attrs:
                            continue
                        if self.whole != 'whole':
                            if attr_name not in self.whole:
                                continue
                        idx = self.attr2idx[attr_name]
                        rest.append(values[idx] == '1')


                    perm = perm_list[temp-1]

                    if 'train' in attr_path:
                        self.train_dataset[perm] += [filename, label]
                    elif 'valid' in attr_path:
                        self.valid_dataset[perm] += [filename, label]
                    elif 'test' in attr_path:
                        self.test_dataset[perm] += [filename, label]
                temp += 1

            print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'valid':
            dataset = self.valid_dataset
        else:
            dataset = self.test_dataset
        filename, label, rest, des, filename2, label2 = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        image2 = Image.open(os.path.join(self.image_dir, filename2))
        return self.transform(image), torch.FloatTensor(label), torch.FloatTensor(rest), torch.FloatTensor(des), \
               self.transform(image2), torch.FloatTensor(label2)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_prob(attr_path, selected_attrs):
    lines = [line.rstrip() for line in open(attr_path, 'r')]

    attr2idx = {}
    idx2attr = {}

    all_attr_names = lines[0].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    label = []

    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]

        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')

    label = np.asarray(label)
    return label.sum() / label.shape[0]


def get_loader(image_dir, attr_path, whole, selected_attrs, des_attrs, crop_size=128, image_size=64,
               batch_size=64, dataset='CelebA', mode='train'):
    """Build and return a data loader."""
    transform = []
    if mode == 'train' or mode == 'valid':
        transform.append(T.RandomHorizontalFlip())

    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, whole, selected_attrs, des_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'))
    return data_loader