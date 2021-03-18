import operator
import numpy as np
import sys


def sort_key_cla( label_pred ):
    label_list, pred_list = label_pred
    class_list = pred_list[:3]
    return max( class_list)
def sort_key_rev( label_pred ):
    label_list, pred_list = label_pred
    class_list = pred_list[3:]
    return max( class_list)
def sort_key_product( label_pred ):
    label_list, pred_list = label_pred
    cla_list = np.array( pred_list[ :3 ])
    reg_list = np.array( pred_list[ 3: ])
    final_list = cla_list*reg_list
    return max( final_list.tolist() )


# the option here chooses on which output to sort and choose: 
# 1 = revenue, 
# 2 = classification score, 
# 3 = product of those, 
# 4 = sort by rev but choose by classification
# 5 = sort by classification choose by rev
def get_predictions( truth, prediction, option = 1 ):

    label_pred = [ ( list(l), list(p) ) for l, p in zip( truth, prediction ) ]
    if option == 1:
        label_pred = sorted( label_pred, key = sort_key_rev, reverse = True )
    elif option == 2:
        label_pred = sorted( label_pred, key = sort_key_cla, reverse = True )
    elif option == 3:
        label_pred = sorted( label_pred, key = sort_key_product, reverse = True )
    elif option == 4:
        label_pred = sorted( label_pred, key = sort_key_rev, reverse = True )
    elif option == 5:
        label_pred = sorted( label_pred, key = sort_key_cla, reverse = True )
    else:
        sys.exit(' Incorrect sorting option chosen ')
    count = 0
    pred_rev = 0
    true_rev = 0
    targeted = int( len(label_pred) * 0.15 )

    for label_list, pred_list in label_pred[:targeted]:
        if option == 1:
            index, value = max(enumerate(pred_list [3:] ), key=operator.itemgetter(1))
        elif option == 2:
            index, value = max(enumerate(pred_list [:3] ), key=operator.itemgetter(1))
        elif option == 3:
            cla_list = np.array( pred_list[ :3 ])
            reg_list = np.array( pred_list[ 3: ])
            final_list = cla_list*reg_list
            index, value = max(enumerate(final_list.tolist() ), key=operator.itemgetter(1))
        elif option == 4:
            index, value = max(enumerate(pred_list [:3] ), key=operator.itemgetter(1))
        elif option == 5:
            index, value = max(enumerate(pred_list [3:] ), key=operator.itemgetter(1))
        if label_list[index] > 0:
            count += 1
        pred_rev += pred_list[ index + 3 ]
        true_rev += label_list[ index + 3 ]
    return count, pred_rev, true_rev


def get_predictions_IDs( IDs, prediction, option = 1 ):

    label_pred = [ ( l, list(p) ) for l, p in zip( IDs, prediction ) ]
    if option == 1:
        label_pred = sorted( label_pred, key = sort_key_rev, reverse = True )
    elif option == 2:
        label_pred = sorted( label_pred, key = sort_key_cla, reverse = True )
    elif option == 3:
        label_pred = sorted( label_pred, key = sort_key_product, reverse = True )
    elif option == 4:
        label_pred = sorted( label_pred, key = sort_key_rev, reverse = True )
    elif option == 5:
        label_pred = sorted( label_pred, key = sort_key_cla, reverse = True )
    else:
        sys.exit(' Incorrect sorting option chosen ')
    pred_rev = 0
    targeted = int( len(label_pred) * 0.15 )
    targets = []
    method = []
    pred_rev = []
    for ID, pred_list in label_pred[:targeted]:
        if option == 1:
            index, value = max(enumerate(pred_list [3:] ), key=operator.itemgetter(1))
        elif option == 2:
            index, value = max(enumerate(pred_list [:3] ), key=operator.itemgetter(1))
        elif option == 3:
            cla_list = np.array( pred_list[ :3 ])
            reg_list = np.array( pred_list[ 3: ])
            final_list = cla_list*reg_list
            index, value = max(enumerate(final_list.tolist() ), key=operator.itemgetter(1))
        elif option == 4:
            index, value = max(enumerate(pred_list [:3] ), key=operator.itemgetter(1))
        elif option == 5:
            index, value = max(enumerate(pred_list [3:6] ), key=operator.itemgetter(1))
        targets.append( ID )
        method.append( index )
        pred_rev.append( pred_list[ index + 3 ])
    return targets, method, pred_rev


def tex_table( df, labels, name, caption ):
    f = open( name, "w" )
    f.write("   \\begin{table}[h] \n") 
    f.write("   \\centering \n") 
    f.write("   \\caption{ ")
    f.write( caption )
    f.write("} \n")
    f.write("      \\begin{tabular}{c|c|c|c|c|c}\n" )
    f.write("        fold & m1 & m2 & m3 & m4 & m5 \\\\ \n" )
    for i in range(df.shape[0]):
        f.write( "        {} & {} & {} & {} & {} & {}  \\\\ \n".format( df.index[i], df[ labels[0] ].iloc[i], df[ labels[1] ].iloc[i], df[ labels[2] ].iloc[i], df[ labels[3] ].iloc[i], df[ labels[4] ].iloc[i] ) )
    f.write("      \\end{tabular} \n" )
    f.write("   \\end{table} \n")

    f.close()



# horrible spaghetti
def print_results( k, targeteds, counts1, counts_xgb1, pred_revs1, true_revs1, pred_revs_xgb1, true_revs_xgb1, 
                                 counts2, counts_xgb2, pred_revs2, true_revs2, pred_revs_xgb2, true_revs_xgb2, 
                                 counts3, counts_xgb3, pred_revs3, true_revs3, pred_revs_xgb3, true_revs_xgb3, 
                                 counts4, counts_xgb4, pred_revs4, true_revs4, pred_revs_xgb4, true_revs_xgb4, 
                                 counts5, counts_xgb5, pred_revs5, true_revs5, pred_revs_xgb5, true_revs_xgb5, ):
    for i in range( k ):
        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts1[i]) )
        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb1[i]) )
        print( 'Pred revenue is {} truth value is {}'.format( pred_revs1[i], true_revs1[i] ) )
        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb1[i], true_revs_xgb1[i] ) )
    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs1 ), sum( true_revs1 ) ) )
    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb1), sum( true_revs_xgb1 ) ) )
    print( '\n\n' )
    for i in range( k ):
        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts2[i]) )
        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb2[i]) )
        print( 'Pred revenue is {} truth value is {}'.format( pred_revs2[i], true_revs2[i] ) )
        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb2[i], true_revs_xgb2[i] ) )
    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs2 ), sum( true_revs2 ) ) )
    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb2), sum( true_revs_xgb2 ) ) )
    print( '\n\n' )
    for i in range( k ):
        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts3[i]) )
        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb3[i]) )
        print( 'Pred revenue is {} truth value is {}'.format( pred_revs3[i], true_revs3[i] ) )
        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb3[i], true_revs_xgb3[i] ) )
    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs3 ), sum( true_revs3 ) ) )
    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb3), sum( true_revs_xgb3 ) ) )
    print( '\n\n' )
    for i in range( k ):
        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts4[i]) )
        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb4[i]) )
        print( 'Pred revenue is {} truth value is {}'.format( pred_revs4[i], true_revs4[i] ) )
        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb4[i], true_revs_xgb4[i] ) )
    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs4 ), sum( true_revs4 ) ) )
    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb4), sum( true_revs_xgb4 ) ) )
    print( '\n\n' )
    for i in range( k ):
        print( 'Out of top {}, {} are nonzero'.format( targeteds[i], counts5[i]) )
        print( 'Out of top {}, {} are nonzero from xgb'.format( targeteds[i], counts_xgb5[i]) )
        print( 'Pred revenue is {} truth value is {}'.format( pred_revs5[i], true_revs5[i] ) )
        print( 'Pred revenue is {} truth value is {} using xgb'.format( pred_revs_xgb5[i], true_revs_xgb5[i] ) )
    print( 'SUM: Pred revenue is {} truth value is {}'.format( sum( pred_revs5 ), sum( true_revs5 ) ) )
    print( 'SUM: Pred revenue is {} truth value is {} using xgb'.format( sum( pred_revs_xgb5), sum( true_revs_xgb5 ) ) )

