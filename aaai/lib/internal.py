import os
import sys


import rasterio as rio
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon

#* Misc Shortcut Tools

def _iterable(obj):
    """Returns True if object can be iterated over."""
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def _cpath(path, ext, **kwargs):
    """Ensure passed file path has extension ext with error printing."""
    (_, fpExt) = os.path.splitext(path)
    if (fpExt != ext):
        err_args(**kwargs)
        return False 
    return True 

def _ctype(obj, dtype, **kwargs):
    """Ensure passed obj is of dtype with error printing."""
    if not isinstance(obj, dtype):
        if 'fname' in kwargs:
            print("Error ({}): Bad type. Expected {}, recieved {}.".format(kwargs['fname'], dtype, type(obj)))
        if 'err_msg' in kwargs:
            print(kwargs['err_msg'])
        if 'exit' in kwargs:
            sys.exit(0)
        return False
    return True

def _isgeom(obj, err_msg=None):
    """ Returns true for accepted shapely geometry types. """
    for geom_type in [shp.Point, shp.LineString, shp.Polygon]:
        if isinstance(obj, geom_type):
            return True
    if err_msg != None:
        print(err_msg)
    return False

def check_data(*args):
    """ Checks args for equivalence. """
    if len(args) == 1:
        return bool(args[0])
    for a in args[1:]:
        if a != last: return False
        last = a
    return True

def err_args(**kwargs):
    if 'err_msg' in kwargs:
        print(kwargs['err_msg'])
    if 'exit' in kwargs:
        print("Exiting.")
        sys.exit(0)
    return None
        
# Coming Soon


def _raster(obj, funcname='_raster'):
    """ From Raster or '.tif' filepath, return rio.DatasetReader. """
    if isinstance(obj, rio.DatasetReader):
        return obj
    elif isinstance(obj, str):
        head, ext = os.path.splitext(obj)
        if '.tif' in ext:
            path = head + ext
            obj = rio.open(path)
        else:
            path1 = head + '.tif'
            if os.path.exists(path1):
                obj = rio.open(path1)
            else:
                path2 = head + '.tiff' 
                if os.path.exists(path2):
                    obj = rio.open(path2)
                else:
                    print("Error ({}): Invalid Path for input raster: {}.\nAttempted: '{}'  '{}'.".format(funcname, obj, path1, path2))
                    sys.exit(0)
        return obj
    else:
        print("Error ({}): Invalid type for input raster {}.".format(funcname, type(obj).__name__))
        sys.exit(0)

def _shapes(obj, funcname='_shapes'):
    """ From GeoDataFrame or '.shp' filepath, return GeoDataFrame. """
    if isinstance(obj, gpd.GeoDataFrame):
        return obj
    elif isinstance(obj, str):
        head, ext = os.path.splitext(obj)
        path = head + '.shp'
        try:
            obj = gpd.read_file(path)
        except FileNotFoundError:
            print("Error ({}): Couldn't locate shapefile '{}' for input shapes.".format(funcname, path))
            sys.exit(0)
        return obj
    else:
        print("Error ({}): Invalid type for input shapes {}. Try '.shp' filepath or GeoDataFrame.".format(funcname, type(obj).__name__))
        sys.exit(0)