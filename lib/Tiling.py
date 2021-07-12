import os, sys, time, csv, random, math

import numpy as np
import rasterio as rio 
import geopandas as gpd
import rasterio.windows as win
import shapely.geometry as shp


import lib.GeoTools as gt
""" 
For tiling rasters. 
"""

def SampleTestTiles(raster, labels, label_buffer=0.0, target=200, out_poly_dir=None):
    """ Sample Testing Tensor. """

    # Get small buffered label raster
    if label_buffer > 0:
        smBuffLabels = gt.GDF_Buffer(labels, label_buffer, True)
    else:
        smBuffLabels = labels
    label_raster = gt.GDF_Rasterize(smBuffLabels, raster)
    
    # Get large buffered label raster
    wide_buffer = 112 + label_buffer
    lgBuffLabels = gt.GDF_Buffer(labels, wide_buffer, True)
    lgLabel_raster = gt.GDF_Rasterize(lgBuffLabels, raster, out_path=None)
    lgLabel_data = np.squeeze(lgLabel_raster.read())
    print("Labels Rasterized")

    
    # Reading Pixel Data
    data = label_raster.read(1)
    # Get Row-col pixel index on np array
    pixel_index = np.where(data==1)
    # assuming row-wise scanning in returned where vector
    pixel_data=[[pixel_i, pixel_j, 1] for pixel_i, pixel_j in zip(pixel_index[0],pixel_index[1])]
    print("Read in {} pixels.".format(len(pixel_data)))
                
    # Sample Filled Tiles
    filledWindows, filled_boxes = [], []
    X_tiles, Y_tiles = [], []
    while len(filledWindows) < target:
        # Select Pixel
        focus_pixel = random.choice(pixel_data)
        # Create Window
        newWin = win.Window(col_off=focus_pixel[1]-112, row_off=focus_pixel[0]-112, width=224, height=224)
        # Read data from window
        image_win = raster.read(window=newWin)
        label_win = label_raster.read(window=newWin)
        # Verify Image Shape and nodata
        if image_win.shape == (raster.count,224,224) and np.amin(image_win) >= 0.0:
            filledWindows.append(newWin)
            X_tiles.append(image_win)
            Y_tiles.append(label_win)
            (MinX, MinY, MaxX, MaxY) = win.bounds(newWin, raster.transform)
            filled_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))
    print("Sampled Filled Windows")

    # Sample EMpty Windows
    emptyWindows, empty_boxes = [], []
    while len(emptyWindows) < target:
        empty_row = random.randint(0, raster.height-112)
        empty_col = random.randint(0, raster.width-112)
        if lgLabel_data[empty_row][empty_col] == 0:
            newWin = win.Window(col_off=empty_col-112, row_off=empty_row-112, width=224, height=224)
            # Read data from window
            image_win = raster.read(window=newWin)
            label_win = label_raster.read(window=newWin)
            if image_win.shape == (raster.count,224,224) and np.amin(image_win) >= 0.0:
                emptyWindows.append(newWin)
                X_tiles.append(image_win)
                Y_tiles.append(label_win)
                (MinX, MinY, MaxX, MaxY) = win.bounds(newWin, raster.transform)
                empty_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))
                
    print("Sampled Empty Windows")
    
    if out_poly_dir:
        if not os.path.exists(out_poly_dir): 
            os.mkdir(out_poly_dir)
        FtestBoxFrame = gpd.GeoDataFrame(geometry=filled_boxes, crs=raster.crs)
        FtestBoxFrame.to_file('{}/FilledTestingTiles.shp'.format(out_poly_dir))
        EtestBoxFrame = gpd.GeoDataFrame(geometry=empty_boxes, crs=raster.crs)
        EtestBoxFrame.to_file('{}/EmptyTestingTiles.shp'.format(out_poly_dir))

    # Convert to arrays
    test_img = np.array(X_tiles)
    test_lbl = np.array(Y_tiles)
    # Channels Last
    test_img = np.rollaxis(test_img, 1, 4)
    test_lbl = np.rollaxis(test_lbl, 1, 4)
    # Force Datatype
    test_img = test_img.astype('float32')
    test_lbl = test_lbl.astype('float32')
    return (test_img, test_lbl)



def SampleTiles(raster, labels, label_buffer=0.0, nodata=0.0, target=200, val_target=80, out_poly_dir=None):
    """ Sample Training and Validation Tensors. """
    train_tile_offsets = []
    val_tile_offsets = []


    # NOTE: 
    
    # Get small buffered label raster
    if label_buffer > 0.0:
        smBuffLabels = gt.GDF_Buffer(labels, label_buffer, True)
    else:
        smBuffLabels = labels
    
    label_raster = gt.GDF_Rasterize(smBuffLabels, raster)
    
    # Get large buffered label raster
    wide_buffer = 112+label_buffer
    lgBuffLabels = gt.GDF_Buffer(labels, wide_buffer, True)
    lgLabel_raster = gt.GDF_Rasterize(lgBuffLabels, raster)
    lgLabel_data = np.squeeze(lgLabel_raster.read())
    print("Labels Rasterized")

    
    # Reading Pixel Data
    data = label_raster.read(1)
    # Get row-col pixel locations as nested list. 
    pixel_index = np.where(data==1)
    # assuming row-wise scanning in returned where vector
    pixel_data=[[pixel_i, pixel_j, 1] for pixel_i, pixel_j in zip(pixel_index[0],pixel_index[1])]
    
     # Select Pixel
    focus_pixel_set = np.random.choice(len(pixel_data), size=len(pixel_data), replace=False)

    
    idx = 0
    # Sample Filled Tiles
    filledWindows = []
    X_filled_tiles, Y_filled_tiles = [], []
    for fp_idx in focus_pixel_set: 
        
        focus_pixel = pixel_data[fp_idx] 
        
        
        # Create Window
        newWin = win.Window(col_off=focus_pixel[1]-112, row_off=focus_pixel[0]-112, width=224, height=224)
        # Read data from window
        image_win = raster.read(window=newWin)
        label_win = label_raster.read(window=newWin)
           
            
        # Verify Image Shape and nodata
        if image_win.shape == (raster.count,224,224): # and np.amin(image_win) >= 0.0
            filledWindows.append(newWin)
            X_filled_tiles.append(image_win)
            Y_filled_tiles.append(label_win)

        if len(filledWindows) >= target:
            break
            
    print("Sampled {} Filled Windows".format(len(filledWindows)))

    # Sample Empty Windows
    emptyWindows = []
    X_empty_tiles, Y_empty_tiles = [], []
    while len(emptyWindows) < len(X_filled_tiles):
        empty_row = random.randint(0, raster.height-112)
        empty_col = random.randint(0, raster.width-112)
        if lgLabel_data[empty_row][empty_col] == 0:
            newWin = win.Window(col_off=empty_col-112, row_off=empty_row-112, width=224, height=224)
            # Read data from window
            image_win = raster.read(window=newWin)
            label_win = label_raster.read(window=newWin)
            # Confirm Valid Window
            if image_win.shape == (raster.count,224,224) and np.amin(image_win) >= 0.0:
                emptyWindows.append(newWin)
                X_empty_tiles.append(image_win)
                Y_empty_tiles.append(label_win)
    print("Sampled Empty Windows")
    
    # Select Filled Validation Tiles
    filled_void_boxes, empty_void_boxes = [], []
    filled_train_boxes, empty_train_boxes = [], []
    filled_validation_boxes, empty_validation_boxes = [], []
    train_img, train_lbl = [],[]
    val_img, val_lbl = [],[] 
    filled_blacklist = []
    # Save Filled Validation Windows
    while len(val_img) < math.floor(val_target/2):
        selector = random.randint(0, (len(filledWindows)-1))
        validation_win = filledWindows[selector]
        for idx, fwindow in enumerate(filledWindows):
            if idx == selector or win.intersect(validation_win, fwindow):
                if idx not in filled_blacklist:
                    filled_blacklist.append(idx)
                    if idx != selector:
                        (MinX, MinY, MaxX, MaxY) = win.bounds(fwindow, raster.transform)
                        filled_void_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))
        val_img.append(X_filled_tiles[selector])
        val_lbl.append(Y_filled_tiles[selector])
        val_tile_offsets.append((filledWindows[selector].col_off, filledWindows[selector].row_off))
        
        (MinX, MinY, MaxX, MaxY) = win.bounds(validation_win, raster.transform)
        filled_validation_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))

    # Save Filled Training Windows
    for idx, window in enumerate(filledWindows):
        if idx not in filled_blacklist:
            train_img.append(X_filled_tiles[idx])
            train_lbl.append(Y_filled_tiles[idx])
            train_tile_offsets.append((filledWindows[idx].col_off, filledWindows[idx].row_off))

            (MinX, MinY, MaxX, MaxY) = win.bounds(window, raster.transform)
            filled_train_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))

    print("Filled Validation Tiles:", len(val_img))
    print("Filled Training Tiles:", len(train_img))
    
    # Save Empty Validation Windows
    empty_blacklist = []
    while len(val_img) < val_target:
        selector = random.randint(0, (len(emptyWindows)-1))
        validation_win = emptyWindows[selector]
        for idx, ewindow in enumerate(emptyWindows):
            if idx == selector or win.intersect(validation_win, ewindow):
                if idx not in empty_blacklist:
                    empty_blacklist.append(idx)
                    if idx != selector:
                        (MinX, MinY, MaxX, MaxY) = win.bounds(ewindow, raster.transform)
                        empty_void_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))
        val_img.append(X_empty_tiles[selector])
        val_lbl.append(Y_empty_tiles[selector])
        val_tile_offsets.append((emptyWindows[selector].col_off, emptyWindows[selector].row_off))

        (MinX, MinY, MaxX, MaxY) = win.bounds(validation_win, raster.transform)
        empty_validation_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))
    
    maxEmpty = math.floor(len(train_img)*2)
    # Save Empty Training Windows
    for idx, window in enumerate(emptyWindows):
        if idx not in empty_blacklist:
            if len(train_img) >= maxEmpty:
                print("Maximum number of empty training tiles reached.")
                break
            train_img.append(X_empty_tiles[idx])
            train_lbl.append(Y_empty_tiles[idx])
            train_tile_offsets.append((emptyWindows[idx].col_off, emptyWindows[idx].row_off))

            (MinX, MinY, MaxX, MaxY) = win.bounds(window, raster.transform)
            empty_train_boxes.append(shp.box(MinX, MinY, MaxX, MaxY))
    
    if out_poly_dir:
        if not os.path.exists(out_poly_dir): 
            os.mkdir(out_poly_dir)
        # Save Training Tile Polygons    
        FtrainBoxFrame = gpd.GeoDataFrame(geometry=filled_train_boxes, crs=raster.crs)
        FtrainBoxFrame.to_file('{}/FilledTrainingTiles.shp'.format(out_poly_dir))
        EtrainBoxFrame = gpd.GeoDataFrame(geometry=empty_train_boxes, crs=raster.crs)
        EtrainBoxFrame.to_file('{}/EmptyTrainingTiles.shp'.format(out_poly_dir))
        # Save Validation Tile Polygons
        FvalBoxFrame = gpd.GeoDataFrame(geometry=filled_validation_boxes, crs=raster.crs)
        FvalBoxFrame.to_file('{}/FilledValidationTiles.shp'.format(out_poly_dir))
        EvalBoxFrame = gpd.GeoDataFrame(geometry=empty_validation_boxes, crs=raster.crs)
        EvalBoxFrame.to_file('{}/EmptyValidationTiles.shp'.format(out_poly_dir))
        # Save Deleted Tile Polygons
        EvoidBoxFrame = gpd.GeoDataFrame(geometry=empty_void_boxes, crs=raster.crs)
        EvoidBoxFrame.to_file('{}/EmptyDeletedTiles.shp'.format(out_poly_dir))
        FvoidBoxFrame = gpd.GeoDataFrame(geometry=filled_void_boxes, crs=raster.crs)
        FvoidBoxFrame.to_file('{}/FilledDeletedTiles.shp'.format(out_poly_dir))

    # Convert to arrays
    train_img = np.array(train_img)
    train_lbl = np.array(train_lbl)
    val_img = np.array(val_img)
    val_lbl = np.array(val_lbl)
    # Channels Last
    train_img = np.rollaxis(train_img, 1, 4)
    train_lbl = np.rollaxis(train_lbl, 1, 4)
    val_img = np.rollaxis(val_img, 1, 4)
    val_lbl = np.rollaxis(val_lbl, 1, 4)
    # Force Datatype
    train_img = train_img.astype('float32')
    train_lbl = train_lbl.astype('float32')
    val_img = val_img.astype('float32')
    val_lbl = val_lbl.astype('float32')
    return (train_img, train_lbl, train_tile_offsets), (val_img, val_lbl, val_tile_offsets)

def WriteOffsets(name, offsets, CSVpath):
    """ Write Offset list to csv. """
    with open(CSVpath, 'w+', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for (col_off, row_off) in offsets:
            csvwriter.writerow([col_off, row_off])


def ResampleTiles(label_raster, offsets_fp):
    """ Reads (col_off, row_off) from csv, using them to create windows for label sampling."""
    
    if os.path.splitext(offsets_fp)[1] != '.csv':
        print("Error (ResampleTiles): Input offsets_fp must be path to csv.")
        sys.exit(0)

    tiles = []
    with open(offsets_fp, 'r+', newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            (col_off, row_off) = int(row[0]), int(row[1])
            TileWin = win.Window(col_off=col_off, row_off=row_off, width=224, height=224)
            tiles.append(label_raster.read(window=TileWin))

    tiles = np.array(tiles)
    tiles = np.rollaxis(tiles, 1, 4)
    tiles = tiles.astype('float32')
    return tiles

def Tile(raster, winWidth, winHeight):
    """ Tile a single raster. """

    # Calculate constants for window formatting
    COLS = int((raster.width / winWidth))
    ROWS = int((raster.height / winHeight))
    AREA = COLS * ROWS
    BANDS = raster.count # Could be INPUT_CHANNELS

    # inputArr stores the image data on which our model will be trained, but without raster crs
    inputArr = np.ndarray(shape=(AREA,BANDS,winHeight,winWidth))

    for i in range(ROWS):
        for j in range(COLS):

            index = ((i*COLS)+(j)) # To ensure indexes are counted correctly
            # Define the window we will be working in for the current chunk
            newWin = win.Window(col_off = (j * winWidth), row_off = (i * winHeight), width=winWidth, height=winHeight)
            # Returns ndarray of pixels on raster in newWin
            inputArr[index] = raster.read(window=newWin)

    return inputArr



import numpy as np
from scipy.ndimage import rotate as scirotate

def AugmentImages(tensor, h_flip=True, v_flip=True, rotate=True):
   
    # Get new tensor shape
    tensor_shp = tensor.shape
    num_windows = tensor_shp[0]
   
    # Parse options
    num_selected = sum([int(option) for option in [h_flip, v_flip, rotate]])
    if num_selected == 0: 
        # No upsampling
        return tensor
    else: 
        # determine shape by uspampling options 
        aug_shp = (num_windows*(num_selected+1),) + tensor_shp[1:4]
    
    
    # Create augmented tensor 
    aug_tensor = np.ones(aug_shp)
    aug_tensor[0:num_windows,...] = tensor
    
    
    idx = 1 # cycle through portions of array.
    if h_flip:
        # Horizontal Flip
        aug_tensor[num_windows:num_windows*2,::1,...] = tensor[:,::-1,...]
        idx += 1
    if v_flip:
        # Verticle Flip
        aug_tensor[num_windows*idx:num_windows*(idx+1),:,::1,...] = tensor[:,:,::-1,...]
        idx += 1
    if rotate:
        # Rotate images 
        aug_tensor[num_windows*idx:] = scirotate(tensor, angle=90, axes=(1,2))

    return aug_tensor