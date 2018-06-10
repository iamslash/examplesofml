from pandas_datareader import data
import pandas as pd
import fix_yahoo_finance as yf
yf.pdr_override()
import pickle, re, glob, sys, os

import tensorflow as tf
import numpy as np
#Generate images without having a window appear
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# ############################################################
# # validate arguements
# start_date = '2018-05-06' #'1996-05-06'
# stock_code = '001040.KS'
# if len(sys.argv) is 1:
#     print('Usage: t.py [stock code]')
#     print('')
#     print('Ex: t.py {}'.format(stock_code))
#     sys.exit(0)
# stock_code = sys.argv[1]

# ############################################################
# # download historical data and write to csv file.
# stock_data = data.get_data_yahoo(stock_code, start_date)
# #print(stock_data.head())
# csv_fname = './{}.csv'.format(stock_code)
# stock_data.to_csv(csv_fname)
# print('ok to save {}'.format(csv_fname))

############################################################
# train data
tf.set_random_seed(123)

seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

############################################################
# 
xy = np.loadtxt("001040.KS.csv", delimiter=',')
xy = xy[::-1] # reverse order
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]] 

############################################################
# 
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX = np.array(dataX[0:train_size])
testX = np.array(dataX[train_size:])
trainY = np.array(dataY[0:train_size])
testY = np.array(dataY[train_size:])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

############################################################
# build a LSTM
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)    
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))


