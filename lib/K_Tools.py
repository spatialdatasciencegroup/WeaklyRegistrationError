import os, sys, tempfile

import numpy as np

import keras 
import tensorflow as tf
import keras.metrics as kmetrics

from keras.layers import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.models import Model, save_model, load_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import keras.backend as K

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Model imports
import lib.models.UNET as unet  
import lib.models.FCN as fcn  
import lib.models.SegNet as segnet  
import lib.models.DeepLab as deeplab

# Local Module
import lib.GeoTools as gt
import lib.Tiling as tile

"""
Tools for the Keras module handling model preparation and evaluation.
"""

### Global Parameters
WINDOW_SHAPE = (224,224, 7)
ODD_LEARNING_RATE  = 1 ## Multiplied by provided learning rate in oscillator
EVEN_LEARNING_RATE = 0.5

def Get_Model(key, dropout=0.2):
    """ Select prepared model for evaluation. """

    print("Preparing {} model...".format(key))

    # Switch model key
    if 'UNET' in key:
        ## Unet selected, no parameters required.  
        model = unet.UNET_7_224(dropout_val=dropout)

    elif 'SegNet' in key:
        ## SegNet selected, need specific shape data.
        model = segnet.SegNet()

    elif 'DeepLab' in key:
        ## Deeplab selected, need specific shape data.
        model = deeplab.Deeplabv3()

    elif 'FCN' in key:
        ## Fully Convolutional Network selected, need specific shape data.
        model = fcn.FCN_Vgg16_32s()

    else:
        ## Invalid key passed; print err and exit
        print("ERROR for Model Selection: Invalid key '{}', please use valid model key:".format(key))
        print("'UNET', 'SegNet', 'FCN', 'DeepLab'")
        sys.exit(0)

    print("{} model built successfully.".format(key))
    return model

### Learning Rate Schedulers
############################

def Oscillate_LR(lr, em_step, em_target):
    """ Get an oscillating Learning Rate by EM step index. """
    if em_step % 2 == 0:
        return lr*EVEN_LEARNING_RATE
    return lr*ODD_LEARNING_RATE

def Linear_Reduce_LR(lr, em_step, em_target):
    """ Reduce Learning Rate based on EM step index. """
    epsilon = 0.0001 # Minimum value, temporary patch for inevitable 0.0 LR
    return (lr - (em_step * lr / em_target)) + epsilon



### Model Preparation
#####################

def SetCallbacks(weights_out=None, es_patience = 20,rlop_factor = 0.5, tensorboard_path = None):
    """ Prepare a Keras model's callbacks as a list. """

    # Reduce Learning Rate on Plateau
    rlop_monitor = 'val_loss'
    
    rlop_patience = 5
    rlop_min_lr = 1e-5
    rlop_min_delt = 0.00001
    rlop_verbosity = 0
    rlop_mode = 'min'

    # Early Stopping
    es_monitor = 'val_loss'
    #es_patience = 20
    es_verbosity = 0

    # Model Checkpoint
    mc_monitor = 'val_loss'
    mc_sbo = True
    mc_verbosity = 0
    if weights_out == None:
        # Backup output path 
        weights_out = './weights.h5'
        
    # Tensorboard
    if not tensorboard_path:
        tensorboard_path = './backup_tensorboard_logs'
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    
    callbacks = [ 
            ReduceLROnPlateau(monitor=rlop_monitor, factor=rlop_factor, patience=rlop_patience, min_lr=rlop_min_lr, min_deta=rlop_min_delt, verbose=rlop_verbosity, mode=rlop_mode),
            EarlyStopping(monitor=es_monitor, patience=es_patience, verbose=es_verbosity),
            ModelCheckpoint(weights_out, monitor=mc_monitor, save_best_only=mc_sbo, verbose=mc_verbosity),
            TensorBoard(log_dir=tensorboard_path)
        ]

    return callbacks




### Model Metrics
#################

def f1_score(y_true, y_pred): #taken from old keras source code
    """ F1 Score measuring using hard labels. """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def precision(y_true, y_pred):
    """ Get precision in tensors. """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """ Get recall in tensors. """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def dice_coef(y_true, y_pred):
    """ Soft label dice coefficient measure. """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_loss(y_true, y_pred):
    """ Points to dice_coef function to create loss evaluation with soft labels. """
    return -dice_coef(y_true, y_pred)

def weighted_dice_coef(y_true, y_pred):
    """ Soft label dice coefficient measure. """
    # weights are stored in labels, extract
    y_true = np.moveaxis(y_true, 3, 1)
    y_true_data = y_true[:, 0]
    y_weights = y_true[:, 1]
    
    # Apply Weight matrix
    y_true_f = K.flatten(y_true_data * y_weights)
    y_pred_f = K.flatten(y_pred * y_weights)
 
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)
    

def weighted_dice_coef_loss(y_true, y_pred):
    return -weighted_dice_coef(y_true, y_pred)
    
def false_positives(y_true, y_pred):
    """ Get number of false positives in tensors. """
    fp = kmetrics.FalsePositives()
    fp.update_state(y_true, y_pred)
    return fp.result().numpy()

def false_negatives(y_true, y_pred):
    """ Get number of false positives in tensors. """
    fn = kmetrics.FalseNegatives()
    fn.update_state(y_true, y_pred)
    return fn.result().numpy()

def keras_precision(y_true, y_pred):
    """ Get precision in tensors. """
    fn = kmetrics.Precision()
    fn.update_state(y_true, y_pred)
    return fn.result().numpy()

def keras_recall(y_true, y_pred):
    """ Get recall in tensors. """
    fn = kmetrics.Recall()
    fn.update_state(y_true, y_pred)
    return fn.result().numpy()

### Model Evaluation
####################

def SaveHistory(history, out_dir, tag=''):
    """ Takes Keras History object storing F1, Dice Coef, loss, and accuracy in a list of values. 
    
    Saves plots as PNG.
    """

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    dice_list = [s for s in history.history.keys() if 'dice' in s and 'val' not in s]
    val_dice_list = [s for s in history.history.keys() if 'dice' in s and 'val' in s]
    f1_list = [s for s in history.history.keys() if 'f1' in s and 'val' not in s]
    val_f1_list = [s for s in history.history.keys() if 'f1' in s and 'val' in s]
        
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss{}'.format(tag))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_dir + '/loss{}.png'.format(tag))

    ## Dice Coefficient
    plt.figure(2)
    for l in dice_list:
        plt.plot(epochs, history.history[l], 'b', label='Training Dice Coefficient (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_dice_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation Dice Coefficient (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('DiceCoef{}'.format(tag))
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.savefig(out_dir + '/dcoef{}.png'.format(tag))

    ## F1 Score
    plt.figure(4)
    for l in f1_list:
        plt.plot(epochs, history.history[l], 'b', label='Training F1 Score (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_f1_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation F1 Score (' + str(format(history.history[l][-1],'.5f'))+')')


    plt.title('F1_Score{}'.format(tag))
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(out_dir + '/f1{}.png'.format(tag))

    ## Accuracy
    plt.figure(3)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')


    plt.title('Accuracy{}'.format(tag))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out_dir + '/acc{}.png'.format(tag))


def Get_Metric(history, metric):
    """ Return the final value of a specific metric in a Keras history object. """
    return history.history[metric][len(history.history[metric])-1]


def ModelReport(X, Y, model, report_type, index=0, print_report=False, report_md=None):
    """ Get current model preformance, save to dict and print. 
    
    Args:
        X (np.ndarray): Image tensor.
        Y (np.ndarray): Y_true label tensor.
        report_type (str): 'Training', 'Testing', or 'Validation', used to print dict.
        index (int): Index from EM test, used to label output dict. 
        print_report (bool [True]): Optionally print dict. 
    
    Returns:
        dict: dict with keyed metrics.
        - F1_Score (float): Raw F1 Score (<1).
        - Dice_Score (float): Raw Dice Coef Score (<1).
        - False_Positives (float [int]): Total false Positives. 
        - False_Negatives (float [int]): Total false Negatives.
    """

    y_pred=np.copy(model.predict(X))
    y_true=np.copy(Y)
    
    f1 = f1_score(tf.cast(y_true, tf.float32),tf.cast(y_pred, tf.float32))
    dice = dice_coef(tf.cast(y_true, tf.float32),tf.cast(y_pred, tf.float32))
    
    y_pred.shape = y_pred.shape[0]* y_pred.shape[1]*y_pred.shape[2]* y_pred.shape[3]
    y_true.shape = y_true.shape[0]* y_true.shape[1]*y_true.shape[2]* y_true.shape[3]

    report = {
        'F1_Score': f1.numpy(),
        'Dice_Score': dice.numpy(),
        'False_Positives': false_positives(y_true, y_pred),
        'False_Negatives': false_negatives(y_true, y_pred),
        'Precision': precision(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)),
        'Recall': recall(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)),
        'Keras_Precision': keras_precision(y_true, y_pred),
        'Keras_Recall': keras_recall(y_true, y_pred),
    }

    # Write to markdown if passed
    if report_md:
        report_md.write('{} Report {:02}\n'.format(report_type, index))
        for key, item in report.items():
            if key == 'Confusion Matrix':
                report_md.write(' - {}: {}\n'.format( key, item ))
            else:    
                report_md.write(' - {}: {:.3f}\n'.format( key, (item*100) ))
        report_md.write('\n\n')
        
    # print report if passed
    if print_report:
        print('{} Report {:02}'.format(report_type, index))
        for key, item in report.items():
            if key == 'Confusion Matrix':
                print(' - {}: {}\n'.format( key, item ))
            else:    
                print(' - {}: {:.3f}'.format( key, (item*100) ))
    return report 

def PrintReport(model_report: dict, spaces: int = 4):
    """ Prints a model report from the above ModelReport Function. """
    for key, item in model_report.items():
        if key == 'Confusion Matrix':
            continue
        else:   
            print(f"{(' '*spaces)}- {key}: {np.round((item*100), 3)}")


### Model Tools
###############


def intermediate_model(model, layer_index: int = -2):
    """ 
    Get an intermediate-layer prediction model from a keras.Model class.
    
    Note:
        Changed from 'layer_name' to 'layer_index' since names are not consistent
        over multiple compilations in the same tensorflow environment. This was 
        causing issues in the EM_iteration pipeline.  
    
    Args:
        model (keras.Model):    Model to extract intermediate layer from.
        layer_index (int [-2]): Index of layer to set as output. 
                                Defaults to -2, representing the last layer before activation.
    
    Returns:
        keras.Model:            Model with identical inputs and output of defined feature.
        """
    # Return model up to layer index. 
    return tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=layer_index).output)

def Get_Pmap(source_raster, pmodel, pmap_fp):
    """ 
    Use a trained Keras model to generate a probability output map covering the entire raster. 
    Tiles input raster, runs predictor on tiled windows, then appends the output windows, covering nearly the entire raster.

    Args:
        source_raster (DatasetReader): 7D Input raster to run prediction on. 
        out_path (string): Directory to save pmap.
        pmap_name (string): Filename to save pmap as
        pmodel (keras.model): Model to run predictions with.

    Returns:
        DatasetReader: 1D Raster representing full probability output map.
    """
    (win_height, win_width, _) = WINDOW_SHAPE

    inputs = tile.Tile(source_raster, win_width, win_height)
    cl_inputs = np.rollaxis(inputs, 1, 4)

    predictions = pmodel.predict(cl_inputs)

    cols = int((source_raster.width / win_width))
    rows = int((source_raster.height / win_height))

    p_rows = []
    k = 0 # iterator
    for i in range(rows):
        this_row = []
        for j in range(cols):
            # Where the U_Net predictor is called
            this_row.append(np.squeeze(predictions[k])) # Output array Shape == (win_height, win_width) could add [0] to avoid dim1 wrapper (1, win_height, win_width)
            k += 1
        p_rows.append(np.hstack(this_row)) # stack row pixel arrays 
    pmap = np.vstack(p_rows) # Shape == (src_height, src_width) ONLY, expand dims done later
    
    return gt.ND_Raster(arr=pmap, source_raster=source_raster, out_path=pmap_fp)