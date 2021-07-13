"""
Preprocessing Script
====================

Parses raw downloaded data from dropbox into training-ready tensors.

The following files are required for preprocessing script:
- 'merged_raster.tif': Merged Image raster
- 'imperfect_labels.shp': Imperfect stream labels 
- 'refined_labels.shp': Hand-refined stream labels

Outputs parsed tensors into the tensor folder defined by TENSOR_DIR in config.py.

Tasks:
    Save polygon representation of windows to results folder
    Add option for output.
"""
import os, sys
import numpy as np
import rasterio as rio
import geopandas as gpd
import lib.raster_parse as rp
import lib.Tiling as t
from lib import gettime

### Hyperparameters 
# Buffers for labels (in meters)
LABEL_BUFFER = 4
# Target number of tiles for each set
TEST_TARGET = 200
TRAIN_TARGET = 500
VAL_TARGET = 150
# Tile upsampling options
AUGMENT_OPTIONS = dict(
    horizontal_flip = False,
    vertical_flip = True,
    rotate = True
)
### ---------------

# Use '-v' for verbose
VERBOSE = ('-v' in sys.argv)
# Verbosity printing
def printv(*args):
    if VERBOSE:
        print(*args)
    else:
        return

from config import INPUT_DATA_DIR, TENSOR_DIR
if not os.path.exists(INPUT_DATA_DIR):
    raise RuntimeError(f"Missing required path for raw input data: '{INPUT_DATA_DIR}'. This can be configured in config.py.")
if not os.path.exists(TENSOR_DIR):
    raise RuntimeError(f"Missing required path for tensor output: '{TENSOR_DIR}'. This can be configured in config.py.")

### Load Input Data
test_raster, train_raster = None, None
# Check if training/testing rasters have already been seperated
if "train_raster.tif" in os.listdir(INPUT_DATA_DIR):
    train_raster_fp = os.path.join(INPUT_DATA_DIR, "train_raster.tif")
    train_raster = rio.open(train_raster_fp)
if "test_raster.tif" in os.listdir(INPUT_DATA_DIR):
    test_raster_fp = os.path.join(INPUT_DATA_DIR, "test_raster.tif")
    test_raster = rio.open(test_raster_fp)

if ((not test_raster) or (not train_raster)):
    # Split rasters from full scene if not loaded
    merged_fp = os.path.join(INPUT_DATA_DIR, 'merged_imagery.tif')
    merged_raster = rio.open(merged_fp)
    test_raster, train_raster = rp.RasterSplit(merged_raster, split=0.5, out_dir=INPUT_DATA_DIR)

# Load label shapefiles
imperfect_label_fp = os.path.join(INPUT_DATA_DIR, 'imperfect_labels.shp')
imperfect_labels = gpd.read_file(imperfect_label_fp)
refined_label_fp = os.path.join(INPUT_DATA_DIR, 'refined_labels.shp')
refined_labels = gpd.read_file(refined_label_fp)

printv("> Successfully loaded raw data.")

# Sample testing tensors
(X_test, Y_test) = t.SampleTestTiles(raster=test_raster, 
                                      labels=refined_labels,
                                      label_buffer=LABEL_BUFFER, 
                                      target=TEST_TARGET,
                                      out_poly_dir=None,
                                      verbose=VERBOSE)
printv("> Created Testing tensors.")

(X_train, Y_train, train_offsets), (X_val, Y_val, val_offsets) = t.SampleTiles(
    raster=train_raster,      
    labels=imperfect_labels,  
    label_buffer=LABEL_BUFFER,
    train_target=TRAIN_TARGET,
    val_target=VAL_TARGET,             
    out_poly_dir=None,
    verbose=VERBOSE
)
printv("> Created Training and validation tensors.")

# Upsample Training Tensors
X_train = t.AugmentImages(X_train, **AUGMENT_OPTIONS)
Y_train = t.AugmentImages(Y_train, **AUGMENT_OPTIONS)

# Upsample Validation Tensors
X_val = t.AugmentImages(X_val, **AUGMENT_OPTIONS)
Y_val = t.AugmentImages(Y_val, **AUGMENT_OPTIONS)
printv("> Upsampled Input Data.")

# Save Offsets to csv files so the same windows may be resampled during iteration
train_offsets_fp = os.path.join(TENSOR_DIR, 'train_offsets.csv')
t.WriteOffsets('Training', train_offsets, train_offsets_fp)
val_offsets_fp = os.path.join(TENSOR_DIR, 'val_offsets.csv')
t.WriteOffsets('Validation', val_offsets, val_offsets_fp)
printv("> Wrote offset csv files.")

# Save all tensors 
np.save(os.path.join(TENSOR_DIR, 'X_train'), X_train)
np.save(os.path.join(TENSOR_DIR, 'Y_train'), Y_train)
np.save(os.path.join(TENSOR_DIR, 'X_val'), X_val)
np.save(os.path.join(TENSOR_DIR, 'Y_val'), Y_val)
np.save(os.path.join(TENSOR_DIR, 'X_test'), X_test)
np.save(os.path.join(TENSOR_DIR, 'Y_test'), Y_test)
printv("> Saved tensors to folder.")


### Write test data to markdown
markdown_fp = os.path.join(TENSOR_DIR, 'tensor_info.md')
with open(markdown_fp, 'w+') as md:

    # Header / Notes
    md.write("# Tensor Info\n\n")
    md.write(f"### Created on {gettime('%D at %I:%M:%S%p')}\n")
    md.write('\n---\n\n')

    md.write('### Parameters:\n')
    md.write(f'- Buffer: {LABEL_BUFFER}\n')
    md.write(f'- Testing Target: {TEST_TARGET}\n')
    md.write(f'- Training Target: {TRAIN_TARGET}\n')
    md.write(f'- Validation Target: {VAL_TARGET}\n')
    md.write('- Upsampling Options:\n')
    for k, v in AUGMENT_OPTIONS.items():
        md.write(f'  - {k}: {v}\n')
    md.write('\n---\n\n')

    md.write('### Tensors:\n')
    md.write('Training:\n')
    md.write('- X_Train: {}\n'.format(X_train.shape))
    md.write('- Y_Train: {}\n'.format(Y_train.shape))
    md.write('Validation:\n')
    md.write('- X_Validation: {}\n'.format(X_val.shape))
    md.write('- Y_Validation: {}\n'.format(Y_val.shape))
    md.write('Testing:\n')
    md.write('- X_Test: {}\n'.format(X_test.shape))
    md.write('- Y_Test: {}\n'.format(X_test.shape))
    md.close()