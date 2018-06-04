from pandas_datareader import data
import pandas as pd
import fix_yahoo_finance as yf
yf.pdr_override()
import pickle, re, glob, sys

# validate arguements
start_date = '2018-05-06' #'1996-05-06'
stock_code = '001040.KS'
if len(sys.argv) is 1:
    print('Usage: t.py [stock code]')
    print('')
    print('Ex: t.py {}'.format(stock_code))
    sys.exit(0)
stock_code = sys.argv[1]

# download historical data and write to csv file.
stock_data = data.get_data_yahoo(stock_code, start_date)
#print(stock_data.head())
csv_fname = './{}.csv'.format(stock_code)
stock_data.to_csv(csv_fname)
print('ok to save {}'.format(csv_fname))

# train data

# write trained model and mark to a.dat

