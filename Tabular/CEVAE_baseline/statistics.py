import numpy as np
import os
import pandas as pd
src_path = os.path.dirname(os.path.realpath('__file__'))
train_df = open(os.path.join(src_path, '../data/cfgan/list_attr_adult.txt'))

a_train, o_train, x_train, y_train, m_train = [], [], [], [], []
for idx, line in enumerate(train_df):
    if idx != 0:
        line = line.strip('\n').split('\t')
        a_train.append(line[8])
        o_train.append([line[7]]+[line[10]])
        x_train.append(line[1:8]+line[9:11])
        y_train.append(line[11])
        m_train.append([line[7]]+[line[10]])

a_train = np.asarray(a_train, dtype=np.float32)
a_train = np.expand_dims(a_train, axis=1)
x_train = np.asarray(x_train, dtype=np.float32)
o_train = np.asarray(o_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)
y_train = np.expand_dims(y_train, axis=1)
m_train = np.asarray(m_train, dtype=np.float32)

y0 = 0
y1 = 0
n0 = 0
n1 = 0
for i in range(a_train.shape[0]):
    if a_train[i][0] == 0:
        y0 += y_train[i][0]
        n0 += 1
    else:
        y1 += y_train[i][0]
        n1 += 1
print('p(y=1|a=0) {:.4f}'.format(y0/n0))
print('p(y=1|a=1) {:.4f}'.format(y1/n1))
print('p(y=1|a=1)- p(y=1|a=0) {:.4f}'.format(y1/n1 - y0/n0))
print('\n')

o, n = [], []
for i in range(4):
    o.append([0, 0])
    n.append([0, 0])


for i in range(o_train.shape[0]):
    o_index = (o_train[i][1]+o_train[i][0]*2).astype(int)
    a_index = a_train[i][0].astype(int)
    o[o_index][a_index] += y_train[i][0]
    n[o_index][a_index] += 1

for i in range(4):
    print('{:10.4f} {:15.4f} {:25.4f}'.format(n[i][1], n[i][0], n[i][1] + n[i][0]))
print('\n')

print('p(y=1|a=1, oi) | p(y=1|a=0, oi) | p(y=1|a=1, oi) - p(y=1|a=0, oi)')
for i in range(4):
    print('{:10.4f} {:15.4f} {:25.4f}'.format(o[i][1]/n[i][1], o[i][0]/n[i][0], o[i][1]/n[i][1] - o[i][0]/n[i][0]))

