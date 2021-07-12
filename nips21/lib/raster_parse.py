import os, sys
import rasterio as rio
import numpy as np

import rasterio.windows as rwin

import lib.GeoTools as gt

"""
Tools for parsing/pre-processing raster objects.
"""

def RasterSplit(raster, split=0.5, out_dir=None):
    """ Split a raster into two new, non-overlapping rasters.
    
    Args:
        raster (rio.DatasetReader or .tif path): Source Raster to be divided.
        split (float): What portions should be split.
        vertical (bool): Optionally split the data vertically.
    
    Returns:
        tuple(rio.DatasetReader): Two Rasters, post-split. The first will contain the portion of split, the second containing the remainding slice.
    """
       
    # Get area for first split
    (winWidth1,winHeight1) = (raster.width, int(raster.height*split))

    # Get dimensions for remaining area.  
    split2 = 1-split
    (winWidth2,winHeight2) = (raster.width, int(raster.height*split2))

    # get win2 offsets
    offx = 0
    offy = winHeight1
    
    # Window representations of each split area.
    window1 = rwin.Window(0,0, winWidth1, winHeight1)
    window2 = rwin.Window(offx, offy, winWidth2, winHeight2)

    # Get raster data as numpy arrays.
    arr1 = raster.read(window=window1)
    arr2 = raster.read(window=window2)

    # Initialize output paths (none if no output directory passed.
    outfp1=None
    outfp2=None
        
    if out_dir:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        outfp1 = os.path.join(out_dir, 'test_raster.tif')
        outfp2 = os.path.join(out_dir, 'train_raster.tif')
    
    raster1 = gt.ND_Raster(arr1, source_raster=raster, out_path=outfp1, transform=rwin.transform(window1, raster.transform), nodata=raster.nodata)
    raster2 = gt.ND_Raster(arr2, source_raster=raster, out_path=outfp2, transform=rwin.transform(window2, raster.transform), nodata=raster.nodata)

    return raster1, raster2


def merge(rasters, nodata=-1e-04, out_path: str = None):
    """ 
    Merge raster bands from list of rasters with the same shape.
    Outputs all bands to a numpy float32 datatype.
    
    Args:
        rasters ([rio.DatasetReader]): List of rasterio DatasetReaders as rasters to be merged
        out_path (filepath): Path to save merged raster.

    Returns:
        rio.DatasetReader: Merged raster. Order defined by order of input Copy1rasters.
    """
    
    try:
        raster_iter = iter(rasters)
    except TypeError as te:
        raise RuntimeError('raster_parse.merge: Input rasters must be iterable.')
    
    # Store all band data as list of numpy arrays
    all_bands = []
    
    for idx, raster in enumerate(rasters):
        
        # Check raster validity
        if not isinstance(raster, rio.DatasetReader):
            raise TypeError("raster_parse.merge: Invalid input raster type on index {}, expected rio.DatasetReader, got '{}'.".format(idx, type(rio.DatasetReader).__name__))
        
        # Check raster shape
        if idx == 0:
            # Save expected shape for all rasters
            std_shape = raster.shape
        else:
            # Check equivalent shape
            if raster.shape != std_shape:
                raise RuntimeError("raster_parse.merge: Invalid shape for raster {}, expected: {}, got: {}.".format(idx, std_shape, raster.shape))
        
        # Read raster band data and add to band data list
        all_bands.extend([raster.read(band, out_dtype=np.float32) for band in raster.indexes])
   
    # Stack band data (channels-first)
    merged_data = np.stack(all_bands, axis=0)
    
    # Return as raster (saves if out_path passed)
    return gt.ND_Raster(merged_data, source_raster=rasters[0], nodata=nodata, out_path=out_path)

def cut_shapes(raster, shapes, out_path):
    """
    Cuts shapes from raster
    """
    masked_data, trans = rmask.mask(merged, shapes=shapes, all_touched=True, invert=True, nodata=raster.nodata, filled=True)
    return gt.ND_Raster(masked_data, merged, nodata=raster.nodata, out_path=out_path)


import numpy as np
import numpy.ma as ma

def raster_norm(band, min_nodata, new_nodata=-1e-04):
    
    masked = ma.masked_where(band <= min_nodata, band)
    
    min_val = np.amin(masked)
    max_val = np.amax(masked)
    
    masked_normal = (masked-min_val) / (max_val - min_val)
    
    return ma.filled(masked_normal, fill_value=new_nodata).astype('float32')


def preconfiged_merge():
    """ NOTE: Does not work because these layers need casting into the correct scene. """
    
    # Re-merge to add slope
    import os, sys
    import numpy as np
    import numpy.ma as ma
    import rasterio as rio
    import geopandas as gpd 

    naip_layers = rio.open('/data/GeometricErrors/GeometricErrorStudyArea2/naip.tif')
    dem_layer   = rio.open('/data/GeometricErrors/GeometricErrorStudyArea2/DEM.tif')
    slope_layer = rio.open('/data/GeometricErrors/GeometricErrorStudyArea2/slope_1m_full.tif')
    tpi_layer   = rio.open('/data/GeometricErrors/GeometricErrorStudyArea2/tpi_9_1m.tif')

    print(naip_layers.nodata)
    print(dem_layer.nodata)
    print(slope_layer.nodata)
    print(tpi_layer.nodata)


    normal_bands = []

    all_bands = [naip_layers.read(idx) for idx in naip_layers.indexes]
    for band in all_bands:
        normal_bands.append(raster_norm(band, 0.0))


    dem_data = np.squeeze(dem_layer.read(1))
    normal_bands.append(raster_norm(dem_data, -10000))

    slope_data = np.squeeze(slope_layer.read(1))
    normal_bands.append(raster_norm(slope_data, -9998))

    tpi_data = np.squeeze(tpi_layer.read(1))
    normal_bands.append(raster_norm(tpi_data, -10000))
    
    return normal_bands

