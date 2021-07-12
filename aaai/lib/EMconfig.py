from lib.ModuleTools import *
from lib.UnetTools import *
from lib.internal import _raster
"""
Static EM parameters for testing.

Filepaths set up for DLW.

The parameters in this file MUST reflect their configuration in the Baseline Notebook. 
"""

src_dir = '/data/GeometricErrors/aaai_data'


# -- Enviroment --
env = {} 

tf_seed = wrap(2001, env) # Tensorflow
np_seed = wrap(2001, env) # Numpy
py_seed = wrap(2001, env) # Python
# -----------------------------------------



# -- Tensor Shape and Config --
TensorConfig = {}

# Input Window Shape
WinShape = wrap( (224,224) , TensorConfig)
# Buffer to apply to new annotation when preparing input tensors
labelBuffer = wrap( 2 , TensorConfig)

# - Upsampling Options (Must be same as Baseline) -
# upsample training tiles by horizontal flip
train_h_flip = wrap( True, TensorConfig)
# upsample training tiles by verticle flip
train_v_flip = wrap( True, TensorConfig)
# upsample training tiles by rotation
train_rotate = wrap ( True, TensorConfig)

# upsample validation tiles by horizontal flip
val_h_flip = wrap( True, TensorConfig)
# upsample validation tiles by verticle flip
val_v_flip = wrap( True, TensorConfig)
# upsample validation tiles by rotation
val_rotate = wrap ( True, TensorConfig)
# -----------------------------------------




# -- Annotations -- 
Annotations = {}
# Max length for two linestrings to be repaired under emt.Fix_Lines
maxRepairDist = wrap( 20 , Annotations)
# Buffer used when considering optimal candidates imperfect labels
weightBuffer = wrap( 10 , Annotations)
# Buffered used when rating the optimal frame quality 
precisionBuffer = wrap( 6 , Annotations)
# -----------------------------------------




# -- Model Config -- 
unetcfg = {}
# Learning Rate
learningRate = wrap( 0.1 , unetcfg)
# Training Metrics
metrics = wrap( [dice_coef,'accuracy', f1_score] , unetcfg)
# Batch size 
batchSize = wrap( 32 , unetcfg)
# Shuffle Data
useShuffle = wrap( True , unetcfg)
# Epoch Count
epochs = wrap( 50 , unetcfg)
# Dropout Value
dropval = wrap( 0.2 , unetcfg)
# -----------------------------------------




# - Data from Baseline -
inputDirs = {}
# Dir holds candidate shapefiles for every segment (Must generate from CreateCandidates.ipynb
CandidateDirectory = wrap(src_dir + '/segments', inputDirs)

# Holds Pre-Trained UNet weights (.h5)
preTrainedPath = wrap(src_dir + '/preweights.h5', inputDirs)

# Offsets for training Tiles to re-sample tensors (as csv)
train_offsets_fp = wrap(src_dir + '/train_offsets.csv', inputDirs)
# Offsets for validation Tiles to re-sample tensors (as csv)
val_offsets_fp = wrap(src_dir + '/val_offsets.csv', inputDirs)





# -- Source Data (FROM BOX) --
sourceData = {}

# We call the naip raster for it's nodata value, which was assigned in arcgis
naip = _raster('/data/GeometricErrors/Raw/Set01/crop_1m.tif')
# Ground Truth Label Path
gt_labels = wrap_fp(src_dir + '/GroundTruth.shp', sourceData)
# Imperfect Label Path
imp_labels = wrap_fp(src_dir + '/imperfectLines.shp', sourceData)

# Raster for testing area
test_raster = wrap_fp(src_dir + '/testRaster.tif', sourceData)
# Raster for training area
train_raster = wrap_fp(src_dir + '/trainRaster.tif', sourceData)

# Pre-Trained Model's output pmap (Needed for generating candidate windows when weighting segments)
initial_pmap = wrap_fp(src_dir + '/initial_pmap.tif', sourceData)




# - Input Arrays -
InputArrays = {}
# Training Image Tensor
X_train = wrap_fp(src_dir+'/X_train.npy', InputArrays)
# Baseline Training Label Tensor (for reference)
Y_train = wrap_fp(src_dir+'/Y_train.npy', InputArrays)

# Validation Image Tensor
X_val = wrap_fp(src_dir+'/X_val.npy', InputArrays)
# Baseline Validation Label Tensor (for reference)
Y_val = wrap_fp(src_dir+'/Y_val.npy', InputArrays)

# Testing Tensors (Ground Truth Annotation)
X_test = wrap_fp(src_dir+'/X_test.npy', InputArrays)
Y_test = wrap_fp(src_dir+'/Y_test.npy', InputArrays)


