from data_exp import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt



if __name__ == '__main__' :

# load data, split into test and train set, look at general properties of data

    data = import_data( 'data/Task_Data_Scientist_Dataset.xlsx' )

# remove non-adults as I expect them not to be targets of the campaign, though they do seem to purchase the products sometimes.

    data = data[ data[ 'Age' ] > 17 ]

    data_VolDebNaN = data[ data['VolumeDeb'].isna() ]
    data_predict = data[data['Sale_MF'].isna()]
    data = data[data['Sale_MF'].notna()]

    data_VolDebNaN.hist( bins=50, figsize=(20,15), log=True )
    plt.show()
    
#    data.hist( bins=50, figsize=(20,15), log=True )
#    plt.show()

    train_set, test_set = train_test_split( data, test_size=0.2, random_state=42)

    test_set_labels  =  test_set[ [ 'Client', 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ] ]
    train_set_labels = train_set[ [ 'Client', 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ] ]

    test_set  =  test_set.drop( [ 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ], axis = 1 )
    train_set = train_set.drop( [ 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ], axis = 1 )
    
    scaled_items = [ x for x in train_set.columns.values.tolist() if 'Client' not in x]

    num_pipeline = Pipeline([ ( 'att_mod', CustomOperations() ),
                              ( 'imputer', SimpleImputer( strategy='mean' ) ),
                              ( 'std_scaler', StandardScaler() ),
    ])

    id_pipeline = Pipeline([ ( 'imputer', SimpleImputer( strategy='mean' ) ),
    ])

    full_pipe = ColumnTransformer([
        ( 'id', id_pipeline, [ 'Client' ] ), 
#        ( 'num', num_pipeline, train_set.columns.values.tolist() ), 
        ( 'num', num_pipeline, scaled_items ), 
#        ( 'cat', OneHotEncoder(), [ 'Sex' ] )
    ])

    train_set_tr = full_pipe.fit_transform( train_set )
    test_set_tr = full_pipe.transform( test_set )

    scaler_sc   = full_pipe.named_transformers_['num'].named_steps.std_scaler.scale_;
    scaler_mean = full_pipe.named_transformers_['num'].named_steps.std_scaler.mean_;

    scaler_sc   = np.concatenate( [ [0.], scaler_sc ] ) 

#    print( train_set_tr.shape )
#    print( train_set_tr )
#    print( train_set_labels.to_numpy() )
#    print( type( train_set_labels.to_numpy() ) )
#    print( scaler_sc.shape  )


#    print(test_set_labels.head())
    train_set_labels_arr = train_set_labels.to_numpy()
#    print( train_set_labels_arr[0] )
#    print( train_set_tr[0] )

#    print( train_set_tr[0] )
#    print( train_set_labels_arr[0] )

#    for i,j in noisy_batch_gen( train_set_tr, scaler_sc, 20 ):
#        print(i.shape, j.shape)

#    for i,j in noisy_batch_gen2( train_set_tr, train_set_labels_arr, scaler_sc, 200 ):
#        print(i[0,0], j[0,0])










#    print( train_set_tr )
#    print( train_set_num.describe() )
#    print( train_set_num.columns.values.tolist() )
#    explore_data( train_set_num[ train_set_num[ 'VolumeCred' ].isna()] )
#    explore_data( train_set_num[ train_set_num[ 'VolumeCred' ]>100000] )
#    explore_data( data )
