import logging

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

def loss_function(x, x_reconstructed, true_samples, z):
    nnl = (x_reconstructed - x).pow(2).mean()
    mmd = compute_mmd(true_samples, z)
    loss = nnl + mmd
    return {'loss': loss, 'Negative-Loglikelihood': nnl, 'Maximum_Mean_Discrepancy': mmd}

def setup_logger(logger_name, log_file, level=logging.INFO, filemode='w'):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode=filemode)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

import numpy as np
import torch
import torch.utils.data as utils
# from torch.utils.data import DataLoader, Dataset

def make_whole_adult_loader(npzfile, args):
    np.random.seed(seed=args.seed)
    dat = np.load(npzfile)
    a_train = dat['attr_train']
    a_train = np.expand_dims(a_train, axis=1)
    x_train = dat['x_train']
    """Please change this later!"""
    #o_train = dat['o_train']
    o_train = dat['m_train']
    y_train = dat['y_train']
    y_train = np.expand_dims(y_train, axis=1)

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = shuffle[:valid_ct]
    train_inds = shuffle[valid_ct:]

    a_valid = a_train[valid_inds]
    x_valid = x_train[valid_inds]
    o_valid = o_train[valid_inds]
    y_valid = y_train[valid_inds]

    a_train = a_train[train_inds]
    x_train = x_train[train_inds]
    o_train = o_train[train_inds]
    y_train = y_train[train_inds]

    a_test = dat['attr_test']
    a_test = np.expand_dims(a_test, axis=1)
    x_test = dat['x_test']
    o_test = dat['o_test']
    y_test = dat['y_test']
    y_test = np.expand_dims(y_test, axis=1)

    train_set_x_tensor = torch.Tensor(x_train)
    train_set_o_tensor = torch.Tensor(o_train)
    train_set_a_tensor = torch.Tensor(a_train)
    train_set_y_tensor = torch.Tensor(y_train)

    train_set = utils.TensorDataset(train_set_x_tensor, train_set_o_tensor, train_set_a_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set_x_tensor = torch.Tensor(x_valid)
    valid_set_o_tensor = torch.Tensor(o_valid)
    valid_set_a_tensor = torch.Tensor(a_valid)
    valid_set_y_tensor = torch.Tensor(y_valid)

    valid_set = utils.TensorDataset(valid_set_x_tensor, valid_set_o_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    test_set_x_tensor = torch.Tensor(x_test)
    test_set_o_tensor = torch.Tensor(o_test)
    test_set_a_tensor = torch.Tensor(a_test)
    test_set_y_tensor = torch.Tensor(y_test)

    test_set = utils.TensorDataset(test_set_x_tensor, test_set_o_tensor, test_set_a_tensor, test_set_y_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    input_dim = {'x': x_train.shape[1], 'o': o_train.shape[1], 'a': a_train.shape[1], 'y': y_train.shape[1], 'idx': dat['x_idx_train']}
    return train_loader, valid_loader, test_loader, input_dim

def make_seperate_loader(train_df, test_df, args):
    np.random.seed(seed=args.seed)
    a_train, o_train, x_train, y_train = [], [], [], []
    r_train, d_train = [], []
    for idx, line in enumerate(train_df):
        if idx != 0:
            if args.dataset == 'adult':
                line = line.strip('\n').split(',')
                a_train.append(line[0])
                o_train.append(line[8:10])
                x_train.append(line[1:8])
                y_train.append(line[10])
            elif args.dataset == 'synthetic':
                line = line.strip('\n').split('\t')
                a_train.append(line[1])
                r_train.append(line[0])
                d_train.append([line[2]] + [line[3]])
                y_train.append(line[4])

    a_train = np.asarray(a_train, dtype=np.float32)
    a_train = np.expand_dims(a_train, axis=1)
    r_train = np.asarray(r_train, dtype=np.float32)
    r_train = np.expand_dims(r_train, axis=1)
    d_train = np.asarray(d_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_train = np.expand_dims(y_train, axis=1)

    x_train = np.asarray(x_train, dtype=np.float32)
    o_train = np.asarray(o_train, dtype=np.float32)

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = shuffle[:valid_ct]
    train_inds = shuffle[valid_ct:]

    a_valid = a_train[valid_inds]
    #x_valid = x_train[valid_inds]
    #o_valid = o_train[valid_inds]
    y_valid = y_train[valid_inds]
    r_valid = r_train[valid_inds]
    d_valid = d_train[valid_inds]

    a_train = a_train[train_inds]
    #x_train = x_train[train_inds]
    #o_train = o_train[train_inds]
    y_train = y_train[train_inds]
    r_train = r_train[train_inds]
    d_train = d_train[train_inds]

    a_test, o_test, x_test, y_test = [], [], [], []
    r_test, d_test = [], []
    for idx, line in enumerate(test_df):
        if idx != 0:
            if args.dataset == 'adult':
                line = line.strip('\n').split(',')
                a_test.append(line[0])
                o_test.append(line[8:10])
                x_test.append(line[1:8])
                y_test.append(line[10])
            elif args.dataset == 'synthetic':
                line = line.strip('\n').split('\t')
                a_test.append(line[1])
                r_test.append(line[0])
                d_test.append([line[2]] + [line[3]])
                y_test.append(line[4])

    a_test = np.asarray(a_test, dtype=np.float32)
    a_test = np.expand_dims(a_test, axis=1)
    x_test = np.asarray(x_test, dtype=np.float32)
    o_test = np.asarray(o_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    y_test = np.expand_dims(y_test, axis=1)

    r_test = np.asarray(r_test, dtype=np.float32)
    r_test = np.expand_dims(r_test, axis=1)
    d_test = np.asarray(d_test, dtype=np.float32)

    #train_set_x_tensor = torch.from_numpy(x_train)
    #train_set_o_tensor = torch.from_numpy(o_train)
    train_set_a_tensor = torch.from_numpy(a_train)
    train_set_y_tensor = torch.from_numpy(y_train)
    train_set_r_tensor = torch.from_numpy(r_train)
    train_set_d_tensor = torch.from_numpy(d_train)
    #train_set = utils.TensorDataset(train_set_x_tensor, train_set_o_tensor, train_set_a_tensor, train_set_y_tensor)
    train_set = utils.TensorDataset(train_set_r_tensor, train_set_d_tensor, train_set_a_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    #valid_set_x_tensor = torch.from_numpy(x_valid)
    #valid_set_o_tensor = torch.from_numpy(o_valid)
    valid_set_a_tensor = torch.from_numpy(a_valid)
    valid_set_y_tensor = torch.from_numpy(y_valid)
    valid_set_r_tensor = torch.from_numpy(r_valid)
    valid_set_d_tensor = torch.from_numpy(d_valid)
    #valid_set = utils.TensorDataset(valid_set_x_tensor, valid_set_o_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_set = utils.TensorDataset(valid_set_r_tensor, valid_set_d_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    #test_set_x_tensor = torch.from_numpy(x_test)
    #test_set_o_tensor = torch.from_numpy(o_test)
    test_set_a_tensor = torch.from_numpy(a_test)
    test_set_y_tensor = torch.from_numpy(y_test)
    test_set_r_tensor = torch.from_numpy(r_test)
    test_set_d_tensor = torch.from_numpy(d_test)
    #test_set = utils.TensorDataset(test_set_x_tensor, test_set_o_tensor, test_set_a_tensor, test_set_y_tensor)
    test_set = utils.TensorDataset(test_set_r_tensor, test_set_d_tensor, test_set_a_tensor, test_set_y_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    #input_dim = {'x': x_train.shape[1], 'o': o_train.shape[1], 'a': a_train.shape[1], 'y': y_train.shape[1]}
    input_dim = {'r': r_train.shape[1], 'd': d_train.shape[1], 'a': a_train.shape[1], 'y': y_train.shape[1]}
    return train_loader, valid_loader, test_loader, input_dim

def make_loader(train_df, args):
    np.random.seed(seed=args.seed)
    a_train, o_train, r_train, d_train, y_train = [], [], [], [], []
    for idx, line in enumerate(train_df):
        if idx != 0:
            line = line.strip('\n').split('\t')
            a_train.append(line[8])
            o_train.append([line[7]]+[line[10]])
            r_train.append([line[1]]+[line[7]]+[line[10]])
            d_train.append(line[2:7] + [line[9]])
            y_train.append(line[11])

    a_train = np.asarray(a_train, dtype=np.float32)
    a_train = np.expand_dims(a_train, axis=1)
    r_train = np.asarray(r_train, dtype=np.float32)
    d_train = np.asarray(d_train, dtype=np.float32)
    o_train = np.asarray(o_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_train = np.expand_dims(y_train, axis=1)

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    test_pct = 0.2
    valid_ct = int(n * valid_pct)
    test_ct = int(n * test_pct)
    valid_inds = shuffle[:valid_ct]
    test_inds = shuffle[valid_ct:valid_ct+test_ct]
    train_inds = shuffle[valid_ct+test_ct:]

    a_valid = a_train[valid_inds]
    r_valid = r_train[valid_inds]
    d_valid = d_train[valid_inds]
    o_valid = o_train[valid_inds]
    y_valid = y_train[valid_inds]

    a_test = a_train[test_inds]
    r_test = r_train[test_inds]
    d_test = d_train[test_inds]
    o_test = o_train[test_inds]
    y_test = y_train[test_inds]

    a_train = a_train[train_inds]
    r_train = r_train[train_inds]
    d_train = d_train[train_inds]
    o_train = o_train[train_inds]
    y_train = y_train[train_inds]

    train_set_r_tensor = torch.from_numpy(r_train)
    train_set_d_tensor = torch.from_numpy(d_train)
    train_set_a_tensor = torch.from_numpy(a_train)
    train_set_y_tensor = torch.from_numpy(y_train)
    train_set = utils.TensorDataset(train_set_r_tensor, train_set_d_tensor, train_set_a_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set_r_tensor = torch.from_numpy(r_valid)
    valid_set_d_tensor = torch.from_numpy(d_valid)
    valid_set_a_tensor = torch.from_numpy(a_valid)
    valid_set_y_tensor = torch.from_numpy(y_valid)
    valid_set = utils.TensorDataset(valid_set_r_tensor, valid_set_d_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    test_set_r_tensor = torch.from_numpy(r_test)
    test_set_d_tensor = torch.from_numpy(d_test)
    test_set_a_tensor = torch.from_numpy(a_test)
    test_set_y_tensor = torch.from_numpy(y_test)
    test_set = utils.TensorDataset(test_set_r_tensor, test_set_d_tensor, test_set_a_tensor, test_set_y_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    input_dim = {'r': r_train.shape[1], 'd': d_train.shape[1], 'a': a_train.shape[1], 'y': y_train.shape[1]}
    return train_loader, valid_loader, test_loader, input_dim

def over_sampling(w1, w2):
    import copy
    whole1 = copy.deepcopy(w1)
    whole2 = copy.deepcopy(w2)
    len1 = len(whole1[0])
    len2 = len(whole2[0])
    small, large = (len1, len2) if len1 < len2 else (len2, len1)
    whole_small, whole_large = (whole1, whole2) if small == len1 else (whole2, whole1)

    shuffle_list = np.random.permutation(small)
    m = int(large/small)
    q = large % small

    for list in whole_small:
        list *= m
        for i in range(q):
            shuffle = shuffle_list[i] - 1
            list.append(list[shuffle])
    assert (len(whole_small[0]) == len(whole_large[0])), 'oversampling check'

    whole = []
    for i in range(len(whole_small)):
        whole.append(whole_small[i] + whole_large[i])
    return whole

def make_balancing_loader(train_df, args):
    np.random.seed(seed=args.seed)
    a0_train, o0_train, r0_train, y0_train, d0_train = [], [], [], [], []
    a1_train, o1_train, r1_train, y1_train, d1_train = [], [], [], [], []
    for idx, line in enumerate(train_df):
        if idx != 0:
            line = line.strip('\n').split('\t')
            if line[11] == str(0):
                a0_train.append(line[8])
                o0_train.append([line[7]]+[line[10]])
                r0_train.append([line[1]] + [line[7]] + [line[10]])
                y0_train.append(line[11])
                d0_train.append(line[2:7] + [line[9]])
            else:
                a1_train.append(line[8])
                o1_train.append([line[7]] + [line[10]])
                r1_train.append([line[1]] + [line[7]] + [line[10]])
                y1_train.append(line[11])
                d1_train.append(line[2:7] + [line[9]])

    print(len(y0_train))
    print(len(y1_train))

    whole1 = [a0_train, o0_train, r0_train, y0_train, d0_train]
    whole2 = [a1_train, o1_train, r1_train, y1_train, d1_train]
    (a_train, o_train, r_train, y_train, d_train) = over_sampling(whole1, whole2)

    print(len(y0_train))
    print(len(y1_train))

    a_train = np.asarray(a_train, dtype=np.float32)
    a_train = np.expand_dims(a_train, axis=1)
    r_train = np.asarray(r_train, dtype=np.float32)
    o_train = np.asarray(o_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_train = np.expand_dims(y_train, axis=1)
    d_train = np.asarray(d_train, dtype=np.float32)

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    test_pct = 0.2
    valid_ct = int(n * valid_pct)
    test_ct = int(n * test_pct)
    valid_inds = shuffle[:valid_ct]
    test_inds = shuffle[valid_ct:valid_ct+test_ct]
    train_inds = shuffle[valid_ct+test_ct:]

    a_valid = a_train[valid_inds]
    r_valid = r_train[valid_inds]
    o_valid = o_train[valid_inds]
    y_valid = y_train[valid_inds]
    d_valid = d_train[valid_inds]

    a_test = a_train[test_inds]
    r_test = r_train[test_inds]
    o_test = o_train[test_inds]
    y_test = y_train[test_inds]
    d_test = d_train[test_inds]

    a_train = a_train[train_inds]
    r_train = r_train[train_inds]
    o_train = o_train[train_inds]
    y_train = y_train[train_inds]
    d_train = d_train[train_inds]

    train_set_r_tensor = torch.from_numpy(r_train)
    train_set_o_tensor = torch.from_numpy(o_train)
    train_set_a_tensor = torch.from_numpy(a_train)
    train_set_y_tensor = torch.from_numpy(y_train)
    train_set_d_tensor = torch.from_numpy(d_train)
    train_set = utils.TensorDataset(train_set_r_tensor, train_set_d_tensor, train_set_a_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set_r_tensor = torch.from_numpy(r_valid)
    valid_set_o_tensor = torch.from_numpy(o_valid)
    valid_set_a_tensor = torch.from_numpy(a_valid)
    valid_set_y_tensor = torch.from_numpy(y_valid)
    valid_set_d_tensor = torch.from_numpy(d_valid)
    valid_set = utils.TensorDataset(valid_set_r_tensor, valid_set_d_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    test_set_r_tensor = torch.from_numpy(r_test)
    test_set_o_tensor = torch.from_numpy(o_test)
    test_set_a_tensor = torch.from_numpy(a_test)
    test_set_y_tensor = torch.from_numpy(y_test)
    test_set_d_tensor = torch.from_numpy(d_test)
    test_set = utils.TensorDataset(test_set_r_tensor, test_set_d_tensor, test_set_a_tensor, test_set_y_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    input_dim = {'r': r_train.shape[1], 'd': d_train.shape[1], 'a': a_train.shape[1],'y': y_train.shape[1]}
    return train_loader, valid_loader, test_loader, input_dim