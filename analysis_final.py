from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import cv

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression

from data_exp import *
from nn import *
from result_functions import *

if __name__ == '__main__' :

    # load data, split into test and train set, look at general properties of data
    data = import_data( 'data/Task_Data_Scientist_Dataset.xlsx' )

    # remove non-adults as I expect them not to be targets of the campaign, though they do seem to purchase the products sometimes.
    # remove also the clients with very high campaign revenue (removes 3 points)
    print( data['Client'].size )
    data = data[ data[ 'Age' ] > 17 ]
    print( data['Client'].size )
    data_predict = data[data['Sale_MF'].isna()]
    data = data[data['Sale_MF'].notna()]
    print( data['Client'].size )
    data = data[ data[ 'Revenue_MF' ] < 150 ]
    print( data['Client'].size )
    data = data[ data[ 'Revenue_CC' ] < 250 ]
    print( data['Client'].size )
    data = data[ data[ 'Revenue_CL' ] < 50 ]
    print( data['Client'].size )

    # split into the set I'll be working with and the set to provide predition for

    train_set = data.copy( deep=True )
    
    # separate labels and features
    label_names = [ 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ]

    train_set_labels = train_set[ label_names ]

    train_set = train_set.drop( [ 'Client', 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ], axis = 1 )
    predict_IDs = data_predict[ 'Client' ]
    predict_set = data_predict.drop( [ 'Client', 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ], axis = 1 )

    # I standardize campaign revenues by hand 
    MF_mean = train_set_labels['Revenue_MF'].mean()
    MF_std = train_set_labels['Revenue_MF'].std()
    CC_mean = train_set_labels['Revenue_CC'].mean()
    CC_std = train_set_labels['Revenue_CC'].std()
    CL_mean = train_set_labels['Revenue_CL'].mean()
    CL_std = train_set_labels['Revenue_CL'].std()

    train_set_labels[ 'Revenue_MF' ] = ( train_set_labels[ 'Revenue_MF' ] - MF_mean ) / MF_std
    train_set_labels[ 'Revenue_CC' ] = ( train_set_labels[ 'Revenue_CC' ] - CC_mean ) / CC_std
    train_set_labels[ 'Revenue_CL' ] = ( train_set_labels[ 'Revenue_CL' ] - CL_mean ) / CL_std

    # process input features through the pipelines
    num_pipeline = Pipeline([ ( 'att_mod', CustomOperations() ),
                              ( 'std_scaler', StandardScaler() ),
    ])

    full_pipe = ColumnTransformer([
        ( 'num', num_pipeline, train_set.columns.values.tolist() ), 
    ])

    train_set_tr = full_pipe.fit_transform( train_set )
    predict_set_tr = full_pipe.transform( predict_set )

    train_set_labels_arr = train_set_labels.to_numpy()
    predict_IDs_arr = predict_IDs.to_numpy()

    # I have all the data processed, I can start analysing it. 

    trainings = 20
    nepochs = 300

    counts = []       
    pred_revs = []    
    true_revs = []    
    targeteds = []   
                      
    counts_xgb = []   
    pred_revs_xgb = []
    true_revs_xgb = []

    counts_classic = []   
    pred_revs_classic = []
    true_revs_classic = []

    input_shape = train_set_tr.shape[1:]

    # I will simultaneously fit categorization and classification, I split the labels into those
    train_labels_class = train_set_labels_arr[:,:3]
    train_labels_reg = train_set_labels_arr[:,3:]

    all_scores = []
    all_pred = []
    train_losses = []
    val_losses = []
    predictions = np.zeros( ( predict_set_tr.shape[0], 6) )

    # and I train the model defined amount of times
    for j in range(trainings):
        print( '\n\nTraining NN model number {}\n\n'.format( j+1 ) )
        # I define and compile my NN
        model = buildRegressor(input_shape, 2, 32, 0.5 )
        compileModel( model )
        training_history=model.fit(
            train_set_tr,
            [train_labels_class, train_labels_reg],
            #sample_weight=train_weights,
            epochs= nepochs,
            verbose=2,
            batch_size=64,
        )
        if j == 0:
            tf.keras.utils.plot_model(model,
                                      show_shapes=True,
                                      to_file='model.pdf')

        train_loss = training_history.history['loss']
        train_losses.append( train_loss )

        class_prediction = model.predict( predict_set_tr )[0]
        regre_prediction = model.predict( predict_set_tr )[1]

        output = np.concatenate( [ class_prediction, regre_prediction ], axis = 1 )
        predictions += output

#######################################################
#    I summarize the results here
#######################################################
    # Normalize prediction sum by # of trainings, to get the average, scale the revenues back
    prediction = predictions/trainings
    scaled_prediction = np.ndarray.copy(prediction)
    scaled_prediction[:,3] = scaled_prediction[:,3] * MF_std + MF_mean
    scaled_prediction[:,4] = scaled_prediction[:,4] * CC_std + CC_mean
    scaled_prediction[:,5] = scaled_prediction[:,5] * CL_std + CL_mean
    # get the count of the clients got right, and the revenue values and save it for future

    targets, method, pred_rev = get_predictions_IDs( predict_IDs_arr, scaled_prediction, 4 )
    unique, frequency = np.unique(method,
                              return_counts = True) 
    print( unique )
    print( frequency )
    collected_results = np.vstack(( targets, method, pred_rev )).T
    results_df = pd.DataFrame( collected_results, columns=[ 'Client', 'Campaign', 'pred_rev' ] )
    results_df['Campaign']=results_df['Campaign'].replace([0,1,2],['MF','CC','CL'])
    results_df.to_csv( "results.csv", index=True )


