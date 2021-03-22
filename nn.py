import tensorflow as tf
#from keras import models
#from keras import layers
#from keras import optimizers
#from keras import callbacks
#from keras import backend as K
#from keras import Input
#from keras import Model
#from keras import regularizers

import matplotlib.pyplot as plt

def buildRegressor( input_shape, num_hidden_layers, nodes_per_layer, dropout_rate, l2_weight = 0. ):
    
    # define input layer
    sample_input = tf.keras.Input( shape = input_shape, name='Input' )
    print( 'input_shape', input_shape )

    intermed = sample_input
    # define hidden layers
    for _ in range(num_hidden_layers):
        # regularized with l2 weights, linear activation here beacues whatever other activation should be applied 
        # only after batchnormalization, each layer has same number of nodes
        intermed = tf.keras.layers.Dense( nodes_per_layer, activation = 'linear', kernel_regularizer=tf.keras.regularizers.l2( l2_weight ), name='Hidden_{}'.format(_) )( intermed )
        intermed = tf.keras.layers.BatchNormalization( name='Normalization_{}'.format(_))( intermed )
        intermed = tf.keras.layers.PReLU( name='Activation_{}'.format(_))( intermed )
        if dropout_rate > 0:
            intermed = tf.keras.layers.Dropout( dropout_rate, name='Dropout_{}'.format(_) )( intermed )

    # classification layers (separate for each category)
    classification_output = tf.keras.layers.Dense(3, activation = 'sigmoid', name='Classifier' )(intermed)

    # regression layers (separate for each revenue)
    regression_output = tf.keras.layers.Dense( 3, activation = 'linear', name='Regressor' )(intermed)
    
    model = tf.keras.Model( sample_input, [classification_output, regression_output] )
    return model

def compileModel( model ):
    model.compile(
        optimizer = tf.keras.optimizers.Nadam(),
        loss = [ tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanSquaredError() ],
        # to make sure that regression loss does not dominate the learning
        # was necessary before i normalized the revenues.
#        loss_weights = [ 200., 1.],
    )


