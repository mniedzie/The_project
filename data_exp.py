import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import seaborn as sns

def import_data(name):

    ext = name.split('/')[-1].split('.')[-1]
    base = name.split('.')[0]
    if ext == "xlsx":
        name_csv=name.split(".")
        name_csv[-1]="csv"
        if os.path.isfile(  '.'.join(name_csv) ):
            print("reading existing csv version of xlsx file")
            df = pd.read_csv( '.'.join(name_csv), index_col=None)
        else: 
            print("reading xlsx file and saving to csv")
            df = pd.read_excel(name, engine='openpyxl', sheet_name = [ 1, 2, 3, 4] )
            for i in df.keys():
                df[i]=df[i].set_index('Client')
            merged_df = df[1]
            merged_df = merged_df.merge(df[2], how='outer', on='Client' )
            merged_df = merged_df.merge(df[3], how='outer', on='Client' )
            merged_df = merged_df.merge(df[4], how='outer', on='Client' )
            print(merged_df.head())
            merged_df = merged_df.reset_index()
            print(merged_df.head())

            merged_df.to_csv( '.'.join(name_csv), index=False )
            return merged_df
    elif ext == "csv":
        print("reading csv file")
        df = pd.read_csv(name, index_col=None)
    else:
        print("Unrecognized data type")
        exit(0)
    return df

def explore_data(data):

    # simple prints to first see what's in data
    print("\n\nReading out dataframe info \n")
#    print(data.head())
    print(data.info())
    pd.set_option('display.max_columns', None)
    print(data.describe())

    data_MF = data[data['Sale_MF'] == 1]
    data_CC = data[data['Sale_CC'] == 1]
    data_CL = data[data['Sale_CL'] == 1]
    print( 'Total revenue for MF: ', data_MF[ 'Revenue_MF' ].sum(), 'Revenue per succesfull campaign: ', data_MF[ 'Revenue_MF' ].mean() )
    print( 'Total revenue for CC: ', data_CC[ 'Revenue_CC' ].sum(), 'Revenue per succesfull campaign: ', data_CC[ 'Revenue_CC' ].mean() )
    print( 'Total revenue for CL: ', data_CL[ 'Revenue_CL' ].sum(), 'Revenue per succesfull campaign: ', data_CL[ 'Revenue_CL' ].mean() )
    attr_to_plot = [ 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ]
    f,ax = plt.subplots(2, 3, figsize = (14, 10))
    ax[0,0].hist( data_MF['Revenue_MF'], bins = 20  )
    ax[1,0].hist( data_MF['Age'], weights = data_MF['Revenue_MF'], bins = 20 )
    ax[0,1].hist( data_CC['Revenue_CC'], bins = 20  )
    ax[1,1].hist( data_CC['Age'], weights = data_CC['Revenue_CC'], bins = 20   )
    ax[0,2].hist( data_CL['Revenue_CL'], bins = 20  )
    ax[1,2].hist( data_CL['Age'], weights = data_CL['Revenue_CL'], bins = 20   )
        
    plt.show()

    # Scatter plots for chosen variables

#    attr_to_scatter = [ 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL' ]
#    attr_to_plot = [ 'ActBal_CA', 'VolumeCred_CA', 'TransactionsCred_CA', 'VolumeDeb_CA', 'TransactionsDeb_CA' ]
#    f,ax = plt.subplots(2, 3, figsize = (14, 10))
#    ax[0,0].scatter( y = data[ attr_to_plot[0] ], x = data['Count_CA'] )
#    ax[0,1].scatter( y = data[ attr_to_plot[1] ], x = data['Count_CA'] )
#    ax[0,2].scatter( y = data[ attr_to_plot[2] ], x = data['Count_CA'] )
#    ax[1,0].scatter( y = data[ attr_to_plot[3] ], x = data['Count_CA'] )
#    ax[1,1].scatter( y = data[ attr_to_plot[4] ], x = data['Count_CA'] )
#        
#    plt.show()

class CustomOperations( BaseEstimator, TransformerMixin ):
    def __init__( self, keepCustomerDateOnly = True ):
        self.keepCustomerDateOnly = keepCustomerDateOnly
    def fit( self, X, y=None ):
        return self
    def transform( self, X, y=None):
        # Format sex as -1, 0, 1 for M, NaN, F
        X['Sex']=X['Sex'].fillna('X')
        X['Sex']=X['Sex'].replace(['M','F','X'],[-1,1,0])
        ToSetToZero = [ x for x in X.columns if 'Count' in x or 'ActBal' in x or 'Volume' in x or 'Transactions' in x ]
        X[ ToSetToZero ]=X[ ToSetToZero ].fillna(0)

        return X

def add_noise( data, sigma ):
    if data.shape[1] != sigma.shape[0]:
        print( ' Wrong shape of the inputs. ' )
    modifier = np.random.normal( scale = sigma, size = ( data.shape[0], 30 ) )
    return data + modifier



def noisy_batch_gen2( data, labels, sigma, batch_size ):
    while True:
        data_noisy = add_noise( data, sigma )
        p = np.random.permutation( data.shape[0] )
        data_shuffled = data_noisy[p]
        labels_shuffled = labels[p]
        for i in range( 0, data.shape[0], batch_size ):
            yield data_shuffled[ i:i+batch_size, :], labels_shuffled[ i:i+batch_size, :]

def noisy_batch_gen( data, sigma, batch_size ):
    while True:
        data_noisy = add_noise( data, sigma )
        for i in range( 0, data.shape[0], batch_size ):
            yield data_noisy[ i:i+batch_size, :]

