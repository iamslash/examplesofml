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
