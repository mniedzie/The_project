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
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)#, random_state=42)
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


    for i in range(k):
        if k == 1:
            print(' One fold chosen, will use random split with 20% validation ')
            train_set_fold, val_set_tr, train_set_labels_fold, val_set_labels = train_test_split( train_set_tr, train_set_labels, test_size=0.2, random_state=42)
        else:
            print(' Using k-fold sen, currently in fold {} '.format(i+1) )
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
        for j in range(trainings):
            print( '\n\nTraining NN model number {} in fold {}\n\n'.format(j+1, i+1) )
            # I define and compile my NN
            model = buildRegressor(input_shape, 2, 32, 0.5 )
            compileModel( model )
            training_history=model.fit(
                train_set_fold,
                [train_labels_class, train_labels_reg],
                #sample_weight=train_weights,
                epochs= nepochs,
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


            output = np.concatenate( [ class_prediction, regre_prediction ], axis = 1 )
            predictions += output

        # here we dont take into account that we can sell multiple items to the same person
        # naive check here tries to sell the 1 best option to the top outputs

        # Normalize prediction sum by # of trainings, to get the average, scale the revenues back
        prediction = predictions/trainings
        scaled_prediction = np.ndarray.copy(prediction)
        scaled_prediction[:,3] = scaled_prediction[:,3] * MF_std + MF_mean
        scaled_prediction[:,4] = scaled_prediction[:,4] * CC_std + CC_mean
        scaled_prediction[:,5] = scaled_prediction[:,5] * CL_std + CL_mean
        scaled_truth = np.ndarray.copy(truth)
        scaled_truth[:,3] = scaled_truth[:,3] * MF_std + MF_mean
        scaled_truth[:,4] = scaled_truth[:,4] * CC_std + CC_mean
        scaled_truth[:,5] = scaled_truth[:,5] * CL_std + CL_mean
        # get the count of the clients got right, and the revenue values and save it for future

        count1, pred_rev1, true_rev1 = get_predictions( scaled_truth, scaled_prediction, 1 )
        count2, pred_rev2, true_rev2 = get_predictions( scaled_truth, scaled_prediction, 2 )
        count3, pred_rev3, true_rev3 = get_predictions( scaled_truth, scaled_prediction, 3 )
        count4, pred_rev4, true_rev4 = get_predictions( scaled_truth, scaled_prediction, 4 )
        count5, pred_rev5, true_rev5 = get_predictions( scaled_truth, scaled_prediction, 5 )

        counts.append( [ count1, count2, count3, count4, count5]  )
        pred_revs.append( [ pred_rev1, pred_rev2, pred_rev3, pred_rev4, pred_rev5]  )
        true_revs.append( [ true_rev1, true_rev2, true_rev3, true_rev4, true_rev5]  )

        targeteds.append( [int( len(prediction) * 0.15 )] )
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
            #xgb.plot_importance(xgb_r)
            #plt.figure(figsize = (16, 12))
            #plt.show()
        for i in range(3): 
            xgb_r = xgb.train( params = param_reg, dtrain = data_matrices[ 3 + i ] )
            pred = xgb_r.predict( val_matrices[ 3 + i ] )
            pred_xgb.append( pred )

        prediction_xgb = np.array( [ [ pred_xgb[0][i], pred_xgb[1][i], pred_xgb[2][i], pred_xgb[3][i], pred_xgb[4][i], pred_xgb[5][i] ] for i in range( len( pred_xgb[5] ) )  ] )
        scaled_prediction_xgb = np.ndarray.copy(prediction_xgb)

        scaled_prediction_xgb[3] = scaled_prediction_xgb[3] * MF_std + MF_mean
        scaled_prediction_xgb[4] = scaled_prediction_xgb[4] * CC_std + CC_mean
        scaled_prediction_xgb[5] = scaled_prediction_xgb[5] * CL_std + CL_mean

        count1, pred_rev1, true_rev1 = get_predictions( scaled_truth, scaled_prediction_xgb, 1 )
        count2, pred_rev2, true_rev2 = get_predictions( scaled_truth, scaled_prediction_xgb, 2 )
        count3, pred_rev3, true_rev3 = get_predictions( scaled_truth, scaled_prediction_xgb, 3 )
        count4, pred_rev4, true_rev4 = get_predictions( scaled_truth, scaled_prediction_xgb, 4 )
        count5, pred_rev5, true_rev5 = get_predictions( scaled_truth, scaled_prediction_xgb, 5 )

        counts_xgb.append( [ count1, count2, count3, count4, count5]  )
        pred_revs_xgb.append( [ pred_rev1, pred_rev2, pred_rev3, pred_rev4, pred_rev5]  )
        true_revs_xgb.append( [ true_rev1, true_rev2, true_rev3, true_rev4, true_rev5]  )

        # I will train logistic regression with SGDClassifier and some simple linear regression for revenues 

        train_labels_class1 = train_set_labels_fold[:,0]
        train_labels_class2 = train_set_labels_fold[:,1]
        train_labels_class3 = train_set_labels_fold[:,2]

        train_labels_reg1 = train_set_labels_fold[:,3]
        train_labels_reg2 = train_set_labels_fold[:,4]
        train_labels_reg3 = train_set_labels_fold[:,5]

        # Here I chose the features that had highest importance in xgb
        train_set_fold1 = train_set_fold[ :, ( 1, 2, 5, 9, 11, 16, 22, 25, 27 ) ]
        train_set_fold2 = train_set_fold[ :, ( 1, 2, 9, 10, 19, 22, 23, 26, 27 ) ]
        train_set_fold3 = train_set_fold[ :, ( 1, 2, 9, 15, 16, 17, 18, 19, 21, 22 ) ]

        val_set1 = val_set_tr[ :, ( 1, 2, 5, 9, 11, 16, 22, 25, 27 ) ]
        val_set2 = val_set_tr[ :, ( 1, 2, 9, 10, 19, 22, 23, 26, 27 ) ]
        val_set3 = val_set_tr[ :, ( 1, 2, 9, 15, 16, 17, 18, 19, 21, 22 ) ]

        sgd_class1 = SGDClassifier( random_state=42, loss='log' )
        sgd_class2 = SGDClassifier( random_state=42, loss='log' )
        sgd_class3 = SGDClassifier( random_state=42, loss='log' )

        sgd_class1.fit( train_set_fold1, train_labels_class1 ) 
        sgd_class2.fit( train_set_fold2, train_labels_class2 )
        sgd_class3.fit( train_set_fold3, train_labels_class3 )

        sgd_pred_class1 = sgd_class1.decision_function( val_set1 ) 
        sgd_pred_class2 = sgd_class2.decision_function( val_set2 )
        sgd_pred_class3 = sgd_class3.decision_function( val_set3 )

        lin_reg1 = LinearRegression( )
        lin_reg2 = LinearRegression( )
        lin_reg3 = LinearRegression( )

        lin_reg1.fit( train_set_fold1, train_labels_reg1 ) 
        lin_reg2.fit( train_set_fold2, train_labels_reg2 )
        lin_reg3.fit( train_set_fold3, train_labels_reg3 )

        sgd_pred_reg1 = lin_reg1.predict( val_set1 ) 
        sgd_pred_reg2 = lin_reg2.predict( val_set2 )
        sgd_pred_reg3 = lin_reg3.predict( val_set3 )
        prediction_classic = np.array( [ [ sgd_pred_class1[i], 
                                           sgd_pred_class2[i], 
                                           sgd_pred_class3[i], 
                                           sgd_pred_reg1[i], 
                                           sgd_pred_reg2[i], 
                                           sgd_pred_reg3[i] ] for i in range( len( sgd_pred_reg1 ) )  ] )
        scaled_prediction_classic = np.ndarray.copy(prediction_classic)

        scaled_prediction_classic[3] = scaled_prediction_classic[3] * MF_std + MF_mean
        scaled_prediction_classic[4] = scaled_prediction_classic[4] * CC_std + CC_mean
        scaled_prediction_classic[5] = scaled_prediction_classic[5] * CL_std + CL_mean

        count_cla1, pred_rev_cla1, true_rev_cla1 = get_predictions( scaled_truth, scaled_prediction_classic, 1 )
        count_cla2, pred_rev_cla2, true_rev_cla2 = get_predictions( scaled_truth, scaled_prediction_classic, 2 )
        count_cla3, pred_rev_cla3, true_rev_cla3 = get_predictions( scaled_truth, scaled_prediction_classic, 3 )
        count_cla4, pred_rev_cla4, true_rev_cla4 = get_predictions( scaled_truth, scaled_prediction_classic, 4 )
        count_cla5, pred_rev_cla5, true_rev_cla5 = get_predictions( scaled_truth, scaled_prediction_classic, 5 )

        counts_classic.append( [ count_cla1, count_cla2, count_cla3, count_cla4, count_cla5]  )
        pred_revs_classic.append( [ pred_rev_cla1, pred_rev_cla2, pred_rev_cla3, pred_rev_cla4, pred_rev_cla5]  )
        true_revs_classic.append( [ true_rev_cla1, true_rev_cla2, true_rev_cla3, true_rev_cla4, true_rev_cla5]  )
        #lin_reg.fit( train_set_tr, train_set_labels_arr)
    #    lin_pred = lin_reg.predict( val_set_tr )
    #    print( lin_pred )
    #    print( val_set_labels_arr.flatten() )
    #    lin_rmse = np.sqrt(MSE( val_set_labels_arr, lin_pred))
    #    print( 'Lin reg val set error: ', lin_rmse )
    #
    #    lin_train_pred = lin_reg.predict( train_set_tr )
    #    lin_train_rmse = np.sqrt(MSE( train_set_labels_arr, lin_train_pred))
    #    print( 'Lin reg train set error: ', lin_train_rmse )

#######################################################
#    I summarize the results here
#######################################################

    collected_results = np.concatenate( ( np.array( targeteds ), 
                                          np.array( counts ),    np.array( counts_xgb ),     np.array( counts_classic ), 
                                          np.array( true_revs ), np.array( true_revs_xgb ),  np.array( true_revs_classic ), 
                                          np.array( pred_revs ), np.array( pred_revs_xgb ),  np.array( pred_revs_classic )
                                          ), axis=1 )
    results_df = pd.DataFrame( collected_results, columns=[ 'targeted', 'NN_1',      'NN_2',      'NN_3',      'NN_4',      'NN_5',
                                                                        'xgb_1',     'xgb_2',     'xgb_3',     'xgb_4',     'xgb_5',
                                                                        'cls_1',     'cls_2',     'cls_3',     'cls_4',     'cls_5',
                                                                        'NN_true1',  'NN_true2',  'NN_true3',  'NN_true4',  'NN_true5',
                                                                        'xgb_true1', 'xgb_true2', 'xgb_true3', 'xgb_true4', 'xgb_true5',
                                                                        'cls_true1', 'cls_true2', 'cls_true3', 'cls_true4', 'cls_true5',
                                                                        'NN_pred1',  'NN_pred2',  'NN_pred3',  'NN_pred4',  'NN_pred5',
                                                                        'xgb_pred1', 'xgb_pred2', 'xgb_pred3', 'xgb_pred4', 'xgb_pred5',
                                                                        'cls_pred1', 'cls_pred2', 'cls_pred3', 'cls_pred4', 'cls_pred5'] )
    results_df.loc['mean'] = results_df.mean()
    results_df.loc['std'] = results_df.std()
    results_df.loc['sum'] = results_df.sum()
    results_df = results_df.round(2)
    results_df.to_csv( "results_NN_XGB.csv", index=True )


#    print_results( k, targeteds, counts1, counts_xgb1, pred_revs1, true_revs1, pred_revs_xgb1, true_revs_xgb1, 
#                                counts2, counts_xgb2, pred_revs2, true_revs2, pred_revs_xgb2, true_revs_xgb2, 
#                                counts3, counts_xgb3, pred_revs3, true_revs3, pred_revs_xgb3, true_revs_xgb3, 
#                                counts4, counts_xgb4, pred_revs4, true_revs4, pred_revs_xgb4, true_revs_xgb4, 
#                                counts5, counts_xgb5, pred_revs5, true_revs5, pred_revs_xgb5, true_revs_xgb5, )
#    for i in range( k ):
#        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts1[i]) )
#        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb1[i]) )
#        print( 'Pred revenue is {} truth value is {}'.format( pred_revs1[i], true_revs1[i] ) )
#        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb1[i], true_revs_xgb1[i] ) )
#    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs1 ), sum( true_revs1 ) ) )
#    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb1), sum( true_revs_xgb1 ) ) )
#    print( '\n\n' )
#    for i in range( k ):
#        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts2[i]) )
#        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb2[i]) )
#        print( 'Pred revenue is {} truth value is {}'.format( pred_revs2[i], true_revs2[i] ) )
#        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb2[i], true_revs_xgb2[i] ) )
#    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs2 ), sum( true_revs2 ) ) )
#    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb2), sum( true_revs_xgb2 ) ) )
#    print( '\n\n' )
#    for i in range( k ):
#        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts3[i]) )
#        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb3[i]) )
#        print( 'Pred revenue is {} truth value is {}'.format( pred_revs3[i], true_revs3[i] ) )
#        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb3[i], true_revs_xgb3[i] ) )
#    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs3 ), sum( true_revs3 ) ) )
#    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb3), sum( true_revs_xgb3 ) ) )

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
