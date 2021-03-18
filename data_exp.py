import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import seaborn as sns

def import_data( name ):

    root_ext = os.path.splitext( name )
    ext = root_ext[1]
    base = root_ext[0]
    if ext == ".xlsx":
        name_csv=root_ext[0]+".csv"
        if os.path.isfile(  name_csv ):
            print("reading existing csv version of xlsx file")
            df = pd.read_csv( name_csv, index_col=None)
        else: 
            print("reading xlsx file and saving to csv")
            # first sheet contains feature/label description, so i do not keep it
            df = pd.read_excel(name, engine='openpyxl', sheet_name = [ 1, 2, 3, 4] )
            for i in df.keys():
                df[i]=df[i].set_index('Client')
            merged_df = df[1]
            for i in range( 2, 5 ):
                merged_df = merged_df.merge(df[i], how='outer', on='Client' )
            merged_df = merged_df.reset_index()

            merged_df.to_csv( name_csv, index=False )
            return merged_df
    elif ext == ".csv":
        print("reading csv file")
        df = pd.read_csv(name, index_col=None)
    else:
        print("Unrecognized data type")
        exit(0)
    return df

class CustomOperations( BaseEstimator, TransformerMixin ):
    # custom transformer, to process 'Sex' feature and to set zero 
    # to missing values (could be done with SimpleImputer)
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
    modifier = np.random.normal( scale = sigma, size = ( data.shape[0], sigma.shape[0] ) )
    return data + modifier


def noisy_batch_gen( data, labels, sigma, batch_size ):
    while True:
        data_noisy = add_noise( data, sigma )
        p = np.random.permutation( data.shape[0] )
        data_shuffled = data_noisy[p]
        labels_shuffled = labels[p]
        for i in range( 0, data.shape[0], batch_size ):
            yield data_shuffled[ i:i+batch_size, :], labels_shuffled[ i:i+batch_size, :]


def noisy_batch_gen2( data, sigma, batch_size ):
    while True:
        data_noisy = add_noise( data, sigma )
        for i in range( 0, data.shape[0], batch_size ):
            yield data_noisy[ i:i+batch_size, :]






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

    data_MF =       data[ data['Sale_MF'] == 1]
    print( data_MF.describe() )



    # scatter plots of campaign succes, led me to the numbers below

    attr_to_scatter = [ 'Sale_MF',  'Sale_CC',  'Sale_CL' ]
    pd.plotting.scatter_matrix( data[ attr_to_scatter ], figsize = ( 12, 8) )
    plt.show()

    # checking how often more than one campaign works

    dataMF = data[ data['Sale_MF'] == 1]
    dataCC = data[ data['Sale_CC'] == 1]
    dataCL = data[ data['Sale_CL'] == 1]

    dataMFCL = dataMF[ dataMF['Sale_CL'] == 1]
    dataMFCC = dataMF[ dataMF['Sale_CC'] == 1]
    dataCLCC = dataCL[ dataCL['Sale_CC'] == 1]

    dataMFCLCC = dataMFCL[ dataMFCL['Sale_CC'] == 1]
    print( 'MF', dataMF['Sale_MF'].sum() )
    print( 'CC', dataCC['Sale_CC'].sum() )
    print( 'CL', dataCL['Sale_CL'].sum() )
                  
    print( 'MFCL', dataMFCL['Sale_MF'].sum() )
    print( 'MFCC', dataMFCC['Sale_CC'].sum() )
    print( 'CLCC', dataCLCC['Sale_CL'].sum() )
                  
    print( 'MFCLCC', dataMFCLCC['Sale_CL'].sum() )

    # checking how well each revenue is sampled into train/test sets

    train_set, test_set = train_test_split( data, test_size=0.2, random_state=42)

    f,ax = plt.subplots(3, 4, figsize = (14, 10))
    ax[0,0].hist( data['Revenue_MF'], bins = 20, log=True  , label = 'whole MF rev' )
    ax[0,0].legend()
    ax[0,1].hist( data['Revenue_CC'], bins = 20, log=True, label = 'whole CC rev'   )
    ax[0,1].legend()
    ax[0,2].hist( data['Revenue_CL'], bins = 20, log=True, label = 'whole CL rev'   )
    ax[0,2].legend()
    ax[0,3].hist( data['Revenue_max'], bins = 50, log=True, label = 'whole max rev'   )
    ax[0,3].legend()
    ax[1,0].hist( train_set['Revenue_MF'], bins = 20, log=True, label = 'train MF rev'   )
    ax[1,0].legend()
    ax[1,1].hist( train_set['Revenue_CC'], bins = 20, log=True, label = 'train CC rev'   )
    ax[1,1].legend()
    ax[1,2].hist( train_set['Revenue_CL'], bins = 20, log=True, label = 'train CL rev'   )
    ax[1,2].legend()
    ax[1,3].hist( train_set['Revenue_max'], bins = 20, log=True, label = 'train max rev'   )
    ax[1,3].legend()
    ax[2,0].hist( test_set['Revenue_MF'], bins = 20, log=True, label = 'test MF rev'   )
    ax[2,0].legend()
    ax[2,1].hist( test_set['Revenue_CC'], bins = 20, log=True, label = 'test CC rev'   )
    ax[2,1].legend()
    ax[2,2].hist( test_set['Revenue_CL'], bins = 20, log=True, label = 'test CL rev'   )
    ax[2,2].legend()
    ax[2,3].hist( test_set['Revenue_max'], bins = 20, log=True, label = 'test max rev'   )
    ax[2,3].legend()
    plt.show()

    data_VolDebNaN.hist( bins=50, figsize=(20,15), log=True )
    plt.show()

    scaler_sc   = full_pipe.named_transformers_['num'].named_steps.std_scaler.scale_;
    scaler_mean = full_pipe.named_transformers_['num'].named_steps.std_scaler.mean_;

def set_color(row):
    color = 'black'
    if row[ 'Sale_MF' ] == 1 and row[ 'Sale_CC' ] == 1 and row[ 'Sale_CL' ] == 1:
        color = 'gold'
    elif row[ 'Sale_MF' ] == 1 and row[ 'Sale_CC' ] == 1:
        color = 'orange'
    elif row[ 'Sale_MF' ] == 1 and row[ 'Sale_CL' ] == 1:
        color = 'fuchsia'
    elif row[ 'Sale_CL' ] == 1 and row[ 'Sale_CC' ] == 1:
        color = 'turquoise'
    elif row[ 'Sale_CL' ]:
        color = 'royalblue'
    elif row[ 'Sale_CC' ]:
        color = 'forestgreen'
    elif row[ 'Sale_MF' ]:
        color = 'red'
    return color

def set_colorMF(row):
    color = 'black'
    if row[ 'Sale_MF' ] == 1:
        color = 'fuchsia'
    return color

def set_colorCC(row):
    color = 'black'
    if row[ 'Sale_CC' ] == 1:
        color = 'fuchsia'
    return color

def set_colorCL(row):
    color = 'black'
    if row[ 'Sale_CL' ] == 1:
        color = 'fuchsia'
    return color

def set_weight(row):
    weight = 6
    if row[ 'Sale_MF' ] == 1 and row[ 'Sale_CC' ] == 1 and row[ 'Sale_CL' ] == 1:
        weight = row[ 'Revenue_MF' ] + row[ 'Revenue_CC' ] + row[ 'Revenue_CL' ]
    elif row[ 'Sale_MF' ] == 1 and row[ 'Sale_CC' ] == 1:
        weight = row[ 'Revenue_MF' ] + row[ 'Revenue_CC' ] 
    elif row[ 'Sale_MF' ] == 1 and row[ 'Sale_CL' ] == 1:
        weight = row[ 'Revenue_MF' ] + row[ 'Revenue_CL' ]
    elif row[ 'Sale_CL' ] == 1 and row[ 'Sale_CC' ] == 1:
        weight = row[ 'Revenue_CC' ] + row[ 'Revenue_CL' ]
    elif row[ 'Sale_CL' ]:
        weight = row[ 'Revenue_CL' ]
    elif row[ 'Sale_CC' ]:
        weight = row[ 'Revenue_CC' ]
    elif row[ 'Sale_MF' ]:
        weight = row[ 'Revenue_MF' ]
    return weight

def set_weightMF(row):
    weight = 1
    if row[ 'Sale_MF' ] == 1:
        weight = row[ 'Revenue_MF' ]
    return weight*5

def set_weightCC(row):
    weight = 1
    if row[ 'Sale_CC' ] == 1:
        weight = row[ 'Revenue_CC' ]
    return weight*5

def set_weightCL(row):
    weight = 1
    if row[ 'Sale_CL' ] == 1:
        weight = row[ 'Revenue_CL' ]
    return weight*5


