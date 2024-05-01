import numpy as np
import os

import torch
import torch.utils.data as utils

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd

def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b +1e-10) for (a, b) in zip(A, B)])

    return chi

def generated_dataset(datasets, args):
    factual_whole_data, counter_whole_data = None, None

    for dataset in datasets:
        npzfile = os.path.join(args.save_path, dataset + '.npz')
        dat = np.load(npzfile)

        gen_input = np.concatenate((dat['input'], dat['a']), 1)
        gen_y = dat['y']
        col = ['input' + str(i) for i in range(gen_input.shape[1])]
        col.append('y')
        factual_data = np.concatenate((gen_input, gen_y), 1)
        factual_whole_data = np.concatenate((factual_whole_data, factual_data), 0) \
            if dataset != 'train' else factual_data

        gen_input_cf = np.concatenate((dat['input_cf'], dat['a_cf']), 1)
        gen_y_cf = dat['y_cf']
        counter_data = np.concatenate((gen_input_cf, gen_y_cf), 1)
        counter_whole_data = np.concatenate((counter_whole_data, counter_data), 0) \
            if dataset != 'train' else counter_data

    factual_df = pd.DataFrame(data=factual_whole_data, columns=col)
    counter_df = pd.DataFrame(data=counter_whole_data, columns=col)
    return factual_df, counter_df, col


def original_dataset(loaders, col):
    i_all, a_all, y_all = None, None, None
    for idx1, loader in enumerate(loaders):
        for idx2, (x, o, a, y) in enumerate(loader):
            idx = idx1 + idx2
            i = torch.cat((o, x), 1)
            i_all = torch.cat((i_all, i), 0) if idx != 0 else i
            a_all = torch.cat((a_all, a), 0) if idx != 0 else a
            y_all = torch.cat((y_all, y), 0) if idx != 0 else y

    data = (i_all, a_all, y_all)
    data = torch.cat(data, 1)
    data = data.cpu().detach().numpy()
    org_df = pd.DataFrame(data=data, columns=col)
    return org_df


def baseline_classifier(train_loader, valid_loader, test_loader, args, logger):
    factual_df, counter_df, col = generated_dataset(['train'], args)
    train_df = original_dataset([train_loader, valid_loader], col)
    test_df = original_dataset([test_loader], col)

    input = train_df[col[:-1]].values
    y = train_df[col[-1]].values

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from x
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} train -->    test: {:.4f}'.format(name, acc))


def fair_whole_classifier(train_loader, valid_loader, test_loader, args, logger):
    datasets = ['train', 'valid', 'test']
    loaders = [train_loader, valid_loader, test_loader]
    factual_df, counter_df, col = generated_dataset(datasets, args)
    test_df = original_dataset(loaders, col)

    input = factual_df[col[:-1]].values
    y = factual_df[col[-1]].values

    f_data = np.asarray(factual_df[col].values).astype(int)
    original = np.asarray(test_df[col].values).astype(int)

    f_data = np.sum(f_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    f_data = f_data / np.sum(f_data)
    chi2 = chi2_distance(original, f_data)
    logger.info('(factual) chi squared distance is: {:.4f}'.format(chi2))

    cf_data = np.asarray(counter_df[col].values).astype(int)
    cf_data = np.sum(cf_data, 0)

    cf_data = cf_data / np.sum(cf_data)
    chi2 = chi2_distance(original, cf_data)
    logger.info('(counter) chi squared distance is: {:.4f}'.format(chi2))

    whole_df = [factual_df, counter_df]
    whole_df = pd.concat(whole_df)
    whole_data = np.asarray(whole_df[col].values).astype(int)
    whole_data = np.sum(whole_data, 0)

    cf_data = whole_data / np.sum(whole_data)
    chi2 = chi2_distance(original, cf_data)
    logger.info('(whole) chi squared distance is: {:.4f}'.format(chi2))

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from x
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} factual -->    real: {:.4f}'.format(name, acc))

    input = counter_df[col[:-1]].values
    y = counter_df[col[-1]].values

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} counter -->    real: {:.4f}'.format(name, acc))

    input = whole_df[col[:-1]].values
    y = whole_df[col[-1]].values

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} whole -->    real: {:.4f}'.format(name, acc))


def fair_seperate_classifier(loader, args, logger, dataset='train'):
    factual_df, counter_df, col = generated_dataset([dataset], args)
    test_df = original_dataset([loader], col)

    input = factual_df[col[:-1]].values
    y = factual_df[col[-1]].values

    cf_data = np.asarray(factual_df[col].values).astype(int)
    original = np.asarray(test_df[col].values).astype(int)

    cf_data = np.sum(cf_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    cf_data = cf_data / np.sum(cf_data)
    chi2 = chi2_distance(original, cf_data)
    logger.info('chi squared distance is: {:.4f}'.format(chi2))

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} factual -->    real: {:.4f}'.format(name, acc))

    input = counter_df[col[:-1]].values
    y = counter_df[col[-1]].values

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} counter -->    real: {:.4f}'.format(name, acc))
