import operator


def sort_key_cla( label_pred ):
    label_list, pred_list = label_pred
    class_list = pred_list[:3]
    return max( class_list)
def sort_key_rev( label_pred ):
    label_list, pred_list = label_pred
    class_list = pred_list[3:]
    return max( class_list)



def get_predictions( truth, prediction ):

    label_pred = [ ( list(l), list(p) ) for l, p in zip( truth, prediction ) ]
    label_pred = sorted( label_pred, key = sort_key_rev, reverse = True )
    count = 0
    pred_rev = 0
    true_rev = 0
    targeted = int( len(label_pred) * 0.15 )

    for label_list, pred_list in label_pred[:targeted]:
        index, value = max(enumerate(pred_list [:3] ), key=operator.itemgetter(1))
        if label_list[index] > 0:
            count += 1
        pred_rev += pred_list[ index + 3 ]
        true_rev += label_list[ index + 3 ]
    return count, pred_rev, true_rev
