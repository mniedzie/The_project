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
    data = data[ data[ 'Revenue_MF' ] < 150 ]
    data = data[ data[ 'Revenue_CC' ] < 250 ]
    data = data[ data[ 'Revenue_CL' ] < 50 ]
#    data = data[ data['Sale_MF'] == 1]

    # split into the set i'll be working with and the set to provide predition for
    data_predict = data[data['Sale_MF'].isna()]
    data = data[data['Sale_MF'].notna()]

    # will use this feature for stratification
    data[ 'Revenue_max' ] = data[ [ 'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ] ].max( axis = 1 )
    data[ 'Revenue_max' ] = pd.cut( data[ 'Revenue_max' ],
                                    bins = [-0.1, 10., 20., 45., 100., 500.],
                                    labels = [ 1, 2, 3, 4, 5 ])

    # split the data while maintaining the max revenue shape
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_ix, test_ix in split.split( data, data[ 'Revenue_max' ] ):
        train_set = data.iloc[ train_ix ]
        test_set  = data.iloc[ test_ix ]
    data = data.drop( 'Revenue_max', axis = 1 )
    train_set = train_set.drop( 'Revenue_max', axis = 1 )
    test_set  = test_set .drop( 'Revenue_max', axis = 1 )
#    train_set, test_set = train_test_split( data, test_size=0.2, random_state=42)
    
    # separate labels and features
    label_names = [ 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ]
    #label_names = [ 'Revenue_MF' ]

    test_set_labels  =  test_set[ label_names ]
    train_set_labels = train_set[ label_names ]

    test_set  = test_set.drop( [ 'Client', 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ], axis = 1 )
    train_set = train_set.drop( [ 'Client', 'Sale_MF',  'Sale_CC',  'Sale_CL',  'Revenue_MF',  'Revenue_CC',  'Revenue_CL' ], axis = 1 )

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

    test_set_labels[ 'Revenue_MF' ] = ( test_set_labels[ 'Revenue_MF' ] - MF_mean ) / MF_std
    test_set_labels[ 'Revenue_CC' ] = ( test_set_labels[ 'Revenue_CC' ] - CC_mean ) / CC_std
    test_set_labels[ 'Revenue_CL' ] = ( test_set_labels[ 'Revenue_CL' ] - CL_mean ) / CL_std

    # process input features through the pipelines
    num_pipeline = Pipeline([ ( 'att_mod', CustomOperations() ),
                              ( 'std_scaler', StandardScaler() ),
    ])

    full_pipe = ColumnTransformer([
        ( 'num', num_pipeline, train_set.columns.values.tolist() ), 
#        ( 'cat', OneHotEncoder(), [ 'Sex' ] )
    ])

    train_set_tr = full_pipe.fit_transform( train_set )
    test_set_tr = full_pipe.transform( test_set )


#    train_set_tr, val_set_tr, train_set_labels, val_set_labels = train_test_split( train_set_tr, train_set_labels, test_size=0.2, random_state=42)
#
    train_set_labels_arr = train_set_labels.to_numpy()
    test_set_labels_arr = test_set_labels.to_numpy()
#    val_set_labels_arr = val_set_labels.to_numpy()

    # I have all the data processed, I can start analysing it. 

    k = 5
    num_samples = len(train_set_tr) // k
    trainings = 10

    counts = []
    pred_revs = []
    true_revs = []
    targeteds = []

    counts_xgb = []
    pred_revs_xgb = []
    true_revs_xgb = []

    for i in range(k):
        val_set_tr     = train_set_tr[ i * num_samples: (i + 1) * num_samples ]
        val_set_labels = train_set_labels_arr[ i * num_samples: (i + 1) * num_samples ]

        train_set_fold = np.concatenate(
          [ train_set_tr[ : i * num_samples],
            train_set_tr[ (i + 1) * num_samples : ] ],
            axis = 0
        )

        train_set_labels_fold = np.concatenate(
          [ train_set_labels[ : i * num_samples],
            train_set_labels[ (i + 1) * num_samples : ] ],
            axis = 0
        )

        input_shape = train_set_fold.shape[1:]


        # I will simultaneously fit categorization and classification, I split the labels into those
        train_labels_class = train_set_labels_fold[:,:3]
        train_labels_reg = train_set_labels_fold[:,3:]

        val_labels_class = val_set_labels[:,:3]
        val_labels_reg = val_set_labels[:,3:]

        all_scores = []
        all_pred = []
        train_losses = []
        val_losses = []
        predictions = np.zeros( val_set_labels.shape )
        truth       = val_set_labels

        # and I train the model defined amount of times
        for _ in range(trainings):
            # I define and compile my NN
            model = buildRegressor(input_shape, 2, 32, 0.5 )
            compileModel( model )
            training_history=model.fit(
                train_set_fold,
                [train_labels_class, train_labels_reg],
                #sample_weight=train_weights,
                epochs=300,
                verbose=2,
                batch_size=64,
                validation_data=( val_set_tr, [val_labels_class, val_labels_reg] )
            )

            train_loss = training_history.history['loss']
            val_loss = training_history.history['val_loss']
            train_losses.append( train_loss )
            val_losses.append( val_loss )
            

            class_prediction = model.predict( val_set_tr )[0]
            regre_prediction = model.predict( val_set_tr )[1]
            regre_prediction[:,0] = regre_prediction[:,0] * MF_std + MF_mean
            regre_prediction[:,1] = regre_prediction[:,1] * CC_std + CC_mean
            regre_prediction[:,2] = regre_prediction[:,2] * CL_std + CL_mean


            output = np.concatenate( [ class_prediction, regre_prediction ], axis = 1 )
            predictions += output

        # here we dont take into account that we can sell multiple items to the same person
        # naive check here tries to sell the 1 best option to the top outputs

        # Normalize prediction sum by # of trainings, to get the average, duh
        prediction = predictions/trainings
        truth[:,3] = truth[:,3] * MF_std + MF_mean
        truth[:,4] = truth[:,4] * CC_std + CC_mean
        truth[:,5] = truth[:,5] * CL_std + CL_mean
        # get the count of the clients got right, and the revenue values and save it for future
        count, pred_rev, true_rev = get_predictions( truth, prediction )
        counts.append( count )
        pred_revs.append( pred_rev )
        true_revs.append( true_rev )
        targeteds.append( int( len(prediction) * 0.15 ) )

        # I train bdts for classification and regression

        train_labels_xgb = []
        val_labels_xgb = []
        data_matrices = []
        val_matrices = []
        for i in range(6):
            train_labels_xgb.append( train_set_labels_fold[:,i] )
            val_labels_xgb.append( val_set_labels[:,i] )
            data_matrices.append( xgb.DMatrix( data = train_set_fold, label = train_labels_xgb[i], feature_names =  train_set.columns.values.tolist()  ) )
            val_matrices.append( xgb.DMatrix( data = val_set_tr, label = val_labels_xgb[i], feature_names =  train_set.columns.values.tolist() ) )

        param_cla = {
                'objective':'binary:logistic',
                'max_depth': 5,
                'learning_rate': 0.1,
        }
        param_reg = {
                'objective':'reg:squarederror',
                'max_depth': 5,
                'learning_rate': 0.1,
        }
    
    
        pred_xgb = []
        for i in range(3): 
            xgb_r = xgb.train( params = param_cla, dtrain = data_matrices[ i ] )
            pred = xgb_r.predict( val_matrices[ i ] )
            pred_xgb.append(pred)
        for i in range(3): 
            xgb_r = xgb.train( params = param_reg, dtrain = data_matrices[ 3 + i ] )
            pred = xgb_r.predict( val_matrices[ 3 + i ] )
            pred_xgb.append( pred )
        pred_xgb[3] = pred_xgb[3] * MF_std + MF_mean
        pred_xgb[4] = pred_xgb[4] * CC_std + CC_mean
        pred_xgb[5] = pred_xgb[5] * CL_std + CL_mean

        prediction_xgb = np.array( [ [ pred_xgb[0][i], pred_xgb[1][i], pred_xgb[2][i], pred_xgb[3][i], pred_xgb[4][i], pred_xgb[5][i] ] for i in range( len( pred_xgb[5] ) )  ] )
        count, pred_rev, true_rev = get_predictions( truth, prediction_xgb )
        counts_xgb.append( count )
        pred_revs_xgb.append( pred_rev )
        true_revs_xgb.append( true_rev )
#        xgb.plot_importance(xgb_r)
#        plt.figure(figsize = (16, 12))
#        plt.show()

    for i in range( k ):
        print( 'Out of top {}, {} are nonzero'.format(targeteds[i], counts[i]) )
        print( 'Out of top {}, {} are nonzero from xgb'.format(targeteds[i], counts_xgb[i]) )
        print( 'Pred revenue is {} truth value is {}'.format( pred_revs[i], true_revs[i] ) )
        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb[i], true_revs_xgb[i] ) )
    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs ), sum( true_revs ) ) )
    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb), sum( true_revs_xgb ) ) )

#            epochs = range(1, len(train_loss) + 1)
#            plt.plot(epochs, train_loss, 'b', label = 'training loss')
#            plt.plot(epochs, val_loss, 'r', label = 'validation loss')
#            plt.xlabel("Epochs")
#            plt.ylabel("Loss")
#            plt.legend(numpoints = 1)
#            plt.savefig("nn_loss.pdf")
#    MF_prediction = prediction[:,3] * MF_std + MF_mean
#    MF_truth      = truth[:,3] * MF_std + MF_mean
#    CC_prediction = prediction[:,4] * CC_std + CC_mean
#    CC_truth      = truth[:,4] * CC_std + CC_mean
#    CL_prediction = prediction[:,4] * CL_std + CL_mean
#    CL_truth      = truth[:,4] * CL_std + CL_mean
#    
#    print(MF_prediction.shape)
#    plt.scatter( y = MF_prediction, x = MF_truth, label = 'MF revenue', cmap=plt.get_cmap("jet"), alpha=0.2 )
#    plt.scatter( y = CC_prediction, x = CC_truth, label = 'MF revenue', cmap=plt.get_cmap("jet"), alpha=0.2 )
#    plt.scatter( y = CL_prediction, x = CL_truth, label = 'MF revenue', cmap=plt.get_cmap("jet"), alpha=0.2 )
#    plt.xlabel("true revenue")
#    plt.ylabel("predicted")
#    plt.legend()
#    plt.savefig("rev_scatter.pdf")


#    print( val_set_labels_arr[ val_set_labels_arr > 0 ] )
#    print( model.predict( val_set_tr )[ val_set_labels_arr > 0 ] )
#    print( model.predict( val_set_tr )[ val_set_labels_arr <= 0 ] )
#
#    print( model.predict( val_set_tr )[ val_set_labels_arr > 0 ].mean() )
#    print( model.predict( val_set_tr )[ val_set_labels_arr <= 0 ].mean() )
#
#
#    MF_prediction = model.predict( val_set_tr )[1][:,0] * MF_std  + MF_mean
#    MF_truth      = val_set_labels_arr[:,3] * MF_std + MF_mean
#    print( MF_prediction )
#    print( MF_truth )






















#    data_dmatrix = xgb.DMatrix( data = train_set_tr, label = train_set_labels_arr, feature_names =  train_set.columns.values.tolist()  )
#    test_dmatrix = xgb.DMatrix( data = test_set_tr, label = test_set_labels_arr, feature_names =  train_set.columns.values.tolist()  )
#    val_dmatrix = xgb.DMatrix( data = val_set_tr, label = val_set_labels_arr, feature_names =  train_set.columns.values.tolist()  )
#
#    param = {
##            'booster':'gblinear',
#            'objective':'reg:squarederror',
#            'max_depth': 5,
#            'learning_rate': 0.1,
#    }
#
#    xgb_clf = XGBClassifier(**param)
#    scores = cross_val_score(xgb_clf, train_set_tr, train_set_labels_arr,
#                             scoring="neg_mean_squared_error", cv=10)
#    tree_rmse_scores = np.sqrt(-scores)
#    print( tree_rmse_scores )


#    xgb_r = xgb.train( params = param, dtrain = data_dmatrix )
#    pred = xgb_r.predict( val_dmatrix )
##    print( pred )
##    print( val_set_labels_arr.flatten() )
#    rmse = np.sqrt(MSE( val_set_labels_arr, pred))
##    print( 'Validation rms error: ', rmse )
#    train_pred = xgb_r.predict( data_dmatrix )
#    train_rmse = np.sqrt(MSE( train_set_labels_arr, train_pred))
##    print( 'Train rms error: ', train_rmse )
#
##    xgb.plot_importance(xgb_r)
##    plt.figure(figsize = (16, 12))
##    plt.show()
#
#
#    lin_reg = LinearRegression()
#    #lin_reg.fit( train_set_tr, train_set_labels_arr)
##    lin_pred = lin_reg.predict( val_set_tr )
##    print( lin_pred )
##    print( val_set_labels_arr.flatten() )
##    lin_rmse = np.sqrt(MSE( val_set_labels_arr, lin_pred))
##    print( 'Lin reg val set error: ', lin_rmse )
##
##    lin_train_pred = lin_reg.predict( train_set_tr )
##    lin_train_rmse = np.sqrt(MSE( train_set_labels_arr, lin_train_pred))
##    print( 'Lin reg train set error: ', lin_train_rmse )

'''
'''










#    print( train_set_tr.shape )
#    print( train_set_tr )
#    print( train_set_labels.to_numpy() )
#    print( type( train_set_labels.to_numpy() ) )
#    print( scaler_sc.shape  )


#    print(test_set_labels.head())
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
