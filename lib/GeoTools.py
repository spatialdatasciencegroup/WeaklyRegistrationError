import os, tempfile

import numpy as np
import geopandas as gpd
import rasterio as rio

import shapely
import shapely.geometry as shp

import rasterio.features as rfeat

""" 
Functions for general geometry and raster manipulation.
"""

def Make_Labels(gdf, raster, buff_dist=2, out_path=None) -> rio.DatasetReader:
    """ 
    Combines the buffering and rasterization process into 
    a single call for easy conversion from GDF to label raster. 
    """
    gdf = gdf.to_crs(raster.crs)
    buffed_gdf = gdf_buffer(gdf, buff_dist=buff_dist)
    return gdf_rasterize(gdf=buffed_gdf, raster=raster, out_path=out_path)


def GDF_Buffer(gdf, buff_dist=2, flatten=False):
    """ Alias for gdf_buffer."""
    return gdf_buffer(gdf, buff_dist=buff_dist, flatten=flatten)



def gdf_buffer(gdf, buff_dist=2, flatten=False):
    """ Buffer a GeoDataFrame, I wrote this a long time ago so may need editing or testing
    """
    buff_frame = gpd.GeoDataFrame(geometry=[geom.buffer(buff_dist) for geom in gdf.geometry], crs=gdf.crs)

    if flatten:
        union_frame = gpd.GeoDataFrame(geometry=[buff_frame.geometry.unary_union], crs=gdf.crs) # Prevent re-recording overlapping values.
        buff_frame = union_frame.explode().reset_index(drop=True)

    return buff_frame



def variable_buffer(gdf: gpd.GeoDataFrame, default_buffer: float, index_buffers: dict):
    """
    Applies variable buffer to geodataframe
    
    Args:
        gdf (GeoDataFrame)
        defualt_buffer (float): Applied to all indexes not specified in list.
        index_buffers (dict): keyed buffers to apply to any number of indices.
            {idx: buffer, idx_2: buffer}

    Returns:
        gpd.GeoDataFrame: Selectively buffered lines, in the same orientation as before
    """
    
    all_buffed = []
    
    # Iterate over all geoms
    for idx, geom in enumerate(gdf.geometry):
        # Check for specified buffer
        if idx in index_buffers.keys():
            # Use different buffer
            buff_val = index_buffers[idx]
        else:
            # Use default
            buff_val = default_buffer
        
        # buffer and append
        all_buffed.append(geom.buffer(buff_val))

    # Return in same type
    return gpd.GeoDataFrame(geometry=all_buffed, crs=gdf.crs)



def ND_Raster(arr, source_raster, out_path=None, transform=None, nodata=None):
    """ Save a numpy ndarray as a raster to out_path or a temporary file. 
    
    Args:
        arr (np.ndarray):                   Image data to save into the new raster.
        source_raster (rio.DatasetReader):  Raster with new raster's config.
        out_path (string):                  optional filepath for raster, must end in (.tif).
        transform (affine.Affine):          optional transformation matrix for new raster.

    Returns:
        rio.DatasetReader: New raster with data from arr and config from source_raster.
    """

    if out_path == None:
        temp_dir = tempfile.TemporaryDirectory()
        out_path = os.path.join(temp_dir.name, 'temp_raster.tif')
    if transform == None:
        transform = source_raster.transform
    if len(arr.shape) < 3:
        arr = np.expand_dims(arr, 0)
    with rio.open(out_path, "w", driver=source_raster.driver, width=arr.shape[2], height=arr.shape[1], count=arr.shape[0], dtype=arr.dtype, crs=source_raster.crs, transform=transform, nodata=nodata) as dest:
        dest.write(arr)
    result = rio.open(out_path)
    return result


def list_obj(a):
    """ Ensures objects are iterable. """
    while True:
        try: 
            _ = iter(a)
            break
        except TypeError as err:
            a = [a]
    return list(a)
            
def gdf_iou(gdf_a, gdf_b, buffer):
    """ Gets the intersection over union of two geodataframes with the same CRS.

    Args:
        gdf_a (gpd.GeoDataFrame): First frame to be compared.
        gdf_b (gpd.GeoDataFrame): Second frame to be compared.
            Note: Frames must have the same coordinate reference system.

        buffer (float): Buffer to apply before comparing IoU value.
    
    Returns:
        float (0-1): Intersection / Union of geometries after buffering.
    """

    if gdf_a.crs != gdf_b.crs:
        print("ERROR (gdf_iou): Coordinate Reference Systems must be the same for inputs.")

    # Buffer Geometries in both frames
    buff_a = gpd.GeoDataFrame(geometry=[geom.buffer(buffer) for geom in gdf_a.geometry], crs=gdf_a.crs)
    buff_b = gpd.GeoDataFrame(geometry=[geom.buffer(buffer) for geom in gdf_b.geometry], crs=gdf_a.crs)

    # Merge Potential Overlap
    polys_a = shapely.ops.unary_union(buff_a.geometry.tolist())
    polys_b = shapely.ops.unary_union(buff_b.geometry.tolist())

    # Ensure the polygons are iterable
    polys_a = list_obj(polys_a)
    polys_b = list_obj(polys_b)
    
    # Convert Multi-Polygons to GeoDataFrame
    polys_a = gpd.GeoDataFrame(geometry=list(polys_a), crs=gdf_a.crs)
    polys_b = gpd.GeoDataFrame(geometry=list(polys_b), crs=gdf_a.crs)
    
    # Get Intersection, Union as GeoDataFrames
    intersection_frame = gpd.overlay(polys_a, polys_b, how='intersection')
    union_frame = gpd.overlay(polys_a, polys_b, how='union')

    # Sum Intersection Area
    total_intersect = 0
    for geom in intersection_frame.geometry:
        total_intersect += geom.area
    
    # Sum Union Area
    total_union = 0
    for geom in union_frame.geometry:
        total_union += geom.area

    return (total_intersect / total_union)


#* Geometry Rasterization

# With GeoDataframe and rio.DatasetReader, and out_dir, rasterize shapes into 1-band raster image with (0,1) values, (1 = labeled, 0=unlabeled)
def GDF_Rasterize(gdf, raster, out_path=None): 
    """ 
    Rasterizes a GeoDataFrame (gdf) using the metadata of raster. out_path optional.
    Input gdf and raster must have the same CRS.

    Args:
        gdf (gpd.GeoDataFrame):     Geometries to be rasterized
        raster (rio.DatasetReader): Raster to define metadata of rasterized geometries
        
    Returns:
        rio.DatasetReader: Rasterized geometries, inside geoms == 1, outside == 0.
    """

    gdf_arr = rfeat.rasterize(shapes=gdf.geometry, out_shape=raster.shape, fill=0, transform=raster.transform, all_touched=True, default_value=1)

    return ND_Raster(gdf_arr, raster, out_path=out_path)


def gdf_rasterize(gdf, raster, out_path=None):
    """ Alias for GDF_Rasterize. """
    return GDF_Rasterize(gdf, raster, out_path)