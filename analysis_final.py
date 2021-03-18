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
    data = data[ data[ 'Age' ] > 17 ]
    data_predict = data[data['Sale_MF'].isna()]
    data = data[data['Sale_MF'].notna()]
    data = data[ data[ 'Revenue_MF' ] < 150 ]
    data = data[ data[ 'Revenue_CC' ] < 250 ]
    data = data[ data[ 'Revenue_CL' ] < 50 ]

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

    trainings = 2
    nepochs = 30

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

        train_loss = training_history.history['loss']
        train_losses.append( train_loss )

        class_prediction = model.predict( predict_set_tr )[0]
        regre_prediction = model.predict( predict_set_tr )[1]

        output = np.concatenate( [ class_prediction, regre_prediction ], axis = 1 )
        predictions += output

    # Normalize prediction sum by # of trainings, to get the average, scale the revenues back
    prediction = predictions/trainings
    scaled_prediction = np.ndarray.copy(prediction)
    scaled_prediction[:,3] = scaled_prediction[:,3] * MF_std + MF_mean
    scaled_prediction[:,4] = scaled_prediction[:,4] * CC_std + CC_mean
    scaled_prediction[:,5] = scaled_prediction[:,5] * CL_std + CL_mean
    # get the count of the clients got right, and the revenue values and save it for future

    targets, method, pred_rev = get_predictions_IDs( predict_IDs_arr, prediction, 1 )
    for i in range( len( targets ) ):
        print(targets[i], method[i], pred_rev[i])
    unique, frequency = np.unique(method,
                              return_counts = True) 
    print( unique )
    print( frequency )

#######################################################
#    I summarize the results here
#######################################################

#    collected_results = np.concatenate( ( np.array( targeteds ), 
#                                          np.array( counts ),    
#                                          np.array( true_revs ),  
#                                          ), axis=1 )
#    results_df = pd.DataFrame( collected_results, columns=[ 'targeted', 'NN_1',      'NN_2',      'NN_3',      'NN_4',      'NN_5',
#                                                                        'NN_pred1',  'NN_pred2',  'NN_pred3',  'NN_pred4',  'NN_pred5'] )
#    results_df.loc['mean'] = results_df.mean()
#    results_df.loc['std'] = results_df.std()
#    results_df.loc['sum'] = results_df.sum()
#    results_df = results_df.round(2)
#    results_df.to_csv( "results_NN_XGB.csv", index=True )
#
#    tex_table( results_df, ['NN_1',      'NN_2',      'NN_3',      'NN_4',      'NN_5'], "report/tables/NN_counts.tex",
#                            "Clients who bought the marketed product, results from NN using sort and selection methods 1-5. " )
#    tex_table( results_df, ['xgb_1',      'xgb_2',      'xgb_3',      'xgb_4',      'xgb_5'], "report/tables/xgb_counts.tex",
#                            "Clients who bought the marketed product, results from XGBoost using sort and selection methods 1-5. " )
#    tex_table( results_df, ['cls_1',      'cls_2',      'cls_3',      'cls_4',      'cls_5'], "report/tables/cls_counts.tex",
#                            "Clients who bought the marketed product, results from cls using sort and selection methods 1-5. " )
#
#    tex_table( results_df, ['NN_pred1',      'NN_pred2',      'NN_pred3',      'NN_pred4',      'NN_pred5'], "report/tables/NN_predRev.tex",
#                        "Predicted revenue achieved, results from NN using sort and selection methods 1-5. " )
#    tex_table( results_df, ['xgb_pred1',      'xgb_pred2',      'xgb_pred3',      'xgb_pred4',      'xgb_pred5'], "report/tables/xgb_predRev.tex",
#                        "Predicted revenue achieved, results from XGBboost using sort and selection methods 1-5. " )
#    tex_table( results_df, ['cls_pred1',      'cls_pred2',      'cls_pred3',      'cls_pred4',      'cls_pred5'], "report/tables/cls_predRev.tex",
#                        "Predicted revenue achieved, results from SGDclassification and LinRegression using sort and selection methods 1-5. " )
