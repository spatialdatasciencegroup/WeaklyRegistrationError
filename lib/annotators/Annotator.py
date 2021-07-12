import os, sys, time, math, random

import numpy as np
import numpy.ma as ma

import geopandas as gpd
import rasterio as rio

import rasterio.features as rfeat
import rasterio.windows as rwin

import shapely.geometry as shp


"""
Parent Annotator Class to hold methods and attributes used across both methods.

"""

anno_kwargs_key = {
    'interval': ("(int [10])", "Interval at which coordinates are updated over each annotaion."),
    'crs': ("(proj.CRS [None])", "Coordinate Reference System for annotation system. NOT IMPLEMENTED."),
    'coordinate_precision': ("(float [1e-4])", "Float used to compare coordinate position."),
    'verbosity': ("(int [0])", "Enables index printing. \n\t- 1: print indices.\n\t- 0: Silent") 
}


class Annotator:

    def __init__(self, **kwargs) -> None:
        """ Initalize Annotator with the interval at which annotations will be updated,
        coordinate reference system, and coordinate evalutation precision. 
        Kwargs available. 
        
        kwargs:
            interval (int): Interval in the system's CRS that annotation coordiantes will be updated. (Default: 10) 
            crs (CRS): Coordinate Reference System used by annotator.
            coordiante_precision (float): Minimum value that coordinates must be equal to. (Default: 0.0001)
            verbosity (int): Int defining annotator printing.
        """
        
        # Default Attributes
        default_attr = dict(
            interval = 10,
            crs = None,
            coordinate_precision = 1e-4,    # Float precision for coordinate comparison 
            verbosity = 0,                  # Verbosity, 1: prints indices.
            kwargs_key = anno_kwargs_key    # Key word argument key for printing
        )

        # Allowed Attributes
        allowed_attr = list(default_attr.keys())

        # Update with passed arguments
        default_attr.update(kwargs)        
        
        # Update kwargs
        self.__dict__.update((k,v) for k,v in default_attr.items() if k in allowed_attr)


    ### Annotator Configuration Printing
    def show_attributes(self):
        """ Prints attribute information, kwargs may be less obvious. """
        print("\n\nArguments and attributes key for the {}:\n----------------------------".format(type(self).__name__))
        for key, item in self.kwargs_key.items():
            
            print("{} {}:\n\t{}\n".format(key, item[0], item[1]))

    ### General Geometry Tools

    def same_coords(self, coords_a, coords_b):
        """ Compare coordinate tuples with internal float coordinate_precision, return bool. 
        
        Args:
            coords_a, coords_b (tuple(float)): coordinates to compare.
        
        Returns: 
            bool: If the coordinates are deemed identical by the evaluation of
                    the internal precision float, returns True. 
                    Otherwise returns False
        """
        if abs(coords_a[0] - coords_b[0]) > self.coordinate_precision:
            return False
        if abs(coords_a[1] - coords_b[1]) > self.coordinate_precision:
            return False
        return True


    ### Geometry / Raster interaction

    def set_crop_window(self, raster):
        """ Creates a frame to crop geometries to the bounds of a raster. 
        
        Args:
            raster (rio.DatasetReader): Raster to crop geometries. 

        Returns:
            shp.Polygon: Polygon to use when comparing geometries to raster's bounds in same CRS as raster.
        """
        
        # Get bounding box in coordinate ref 
        (mx,my,Mx,My) = raster.bounds
        crop_window = shp.box(mx,my,Mx,My)
    
        self.crop_window = crop_window
        return crop_window

    
    ### Coordinate Position Tools

    def get_offsets(self, point_a, point_b):
        """ Generate perpendicularly offset values to apply to coordinate points. 
        
        Notes:
            Tested this and confirmed that these offsets produce an identical 
            euclidean offset up to 4 decimals' precision. 

        Args:
            point_a (shp.Point): starting point
            point_b (shp.Point): ending point
        
        Returns:
            tuple(float): offset weights (x,y).
        """
        
        delta_x = (point_b.x-point_a.x)
        delta_y = (point_b.y-point_a.y)

        # Weight offsets with euclidean ?sum?
        offset_x = (delta_y/math.sqrt(pow(delta_y, 2)+pow(delta_x, 2)))
        offset_y = (delta_x/math.sqrt(pow(delta_y, 2)+pow(delta_x, 2)))

        return offset_x, offset_y

    