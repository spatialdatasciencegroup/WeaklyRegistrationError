import os, tempfile

import keras 
import tensorflow as tf
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.random import set_seed
import keras.backend as K

INPUT_CHANNELS = 7
OUTPUT_MASK_CHANNELS = 1


def SetCallbacks(weights_out=None):
    # Reduce Learning Rate on Plateau
    rlop_monitor = 'val_loss'
    rlop_factor = 0.5
    rlop_patience = 5
    rlop_min_lr = 1e-9
    rlop_min_delt = 0.00001
    rlop_verbosity = 1
    rlop_mode = 'min'

    # Early Stopping
    es_monitor = 'val_loss'
    es_patience = 20
    es_verbosity = 0

    # Model Checkpoint
    mc_monitor = 'val_loss'
    mc_sbo = True
    mc_verbosity = 0
    if weights_out == None:
        weights_out = './weights.h5'
    callbacks = [ 
            ReduceLROnPlateau(monitor=rlop_monitor, factor=rlop_factor, patience=rlop_patience, min_lr=rlop_min_lr, min_deta=rlop_min_delt, verbose=rlop_verbosity, mode=rlop_mode),
            EarlyStopping(monitor=es_monitor, patience=es_patience, verbose=es_verbosity),
            ModelCheckpoint(weights_out, monitor=mc_monitor, save_best_only=mc_sbo, verbose=mc_verbosity),
        ]

    return callbacks



# Learning Rate Scheduler

def Step_LR_Schedule(lr, em_step, em_target, const=10):
    """ Uses EM index to determine LR. 
    
    Parameters:
    em_step (int): Current EM step index.
    em_target (int): total number of em steps to be run.
    const (num): Constant for division (optional)
    """
    # distributes change evenly over all steps.
    return (lr * ((em_target - em_step) / const))



#* metrics


# Precision

# Recall




# F1 Score with hard labels
def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def f1_score_loss(y_true, y_pred):
    return -f1_score(y_true, y_pred)

# Dice Coefficient with Hard labels
def hard_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    y_true_f = K.round(K.clip(y_true, 0, 1))
    y_pred_f = K.round(K.clip(y_pred, 0, 1))
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def hard_dice_coef_loss(y_true, y_pred):
    return -hard_dice_coef(y_true, y_pred)



# Dice Coefficient
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


#* model
def double_conv_layer(x, size, std_init, dropout=0.0, batch_norm=True):
    if K.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same', kernel_initializer=std_init)(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same', kernel_initializer=std_init)(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


# Declare U_Net
def UNET_7_224(dropout_val=0.2, std_init=None):
    if K.image_data_format() == 'channels_first':
        inputs = Input((INPUT_CHANNELS, 224, 224))
        axis = 1
    else:
        inputs = Input((224, 224, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters, std_init, dropout_val)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, std_init, dropout_val)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, std_init, dropout_val)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, std_init, dropout_val)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, std_init, dropout_val)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters, std_init, dropout_val)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters, std_init, dropout_val)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, std_init, dropout_val)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, std_init, dropout_val)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, std_init, dropout_val)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, std_init, dropout_val)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1), kernel_initializer=std_init)(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="UNET_7_224")

    return model


def SaveHistory(history, out_dir, tag):
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

def Get_Metric_Data(history, metric):
    return history.history[metric]

def Get_Metric(history, metric):
    return history.history[metric][len(history.history[metric])-1]

