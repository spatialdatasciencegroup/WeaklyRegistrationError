import os, sys, time
import numpy as np
import rasterio as rio
import geopandas as gpd
from rasterio import crs
from rasterio import features 
import rasterio.windows as win
import matplotlib.pyplot as plt
import shapely.geometry as shapes
from pyproj import Proj, Transformer, transform
from shapely.geometry import Point, LineString, Polygon, box

# Custom Modules
import lib.GeoTools as gt
from lib.internal import _cpath, _ctype, _isgeom, _shapes, _raster
from lib.fixclip import clip as clp
import lib.GeoTools as gt


# To circumnavigate geopandas shapeTile issue, some non-critical errors have to be silenced.
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)


#* Shape Tools: uses shapely and geopandas to parse and manipulate geometries.
#* Module uses GeoDataFrame as standard data structure. 


#* Geometry Parsing


# Split Returns two lines split from the original line at the given distance from the line's origin
def Split(line, distance):
    # This is taken from shapely manual, linked by user eguaio on stack overflow
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line), None]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


# Parse_by_Dist takes a line and splits it at fixed intervals respective to dist.
def Parse_by_Dist(line, distance, crs=None):
    _ctype(line, LineString, err_msg="Error (Parse_by_Dist): Input line must be a shapely Linestring.", exit=True)
    if distance <= 0.0 or distance > line.length or (line.length - distance) < 1:
        if crs==None:
            return [line]
        else:
            return gpd.GeoDataFrame(geometry=[line], crs=crs)
    current_line = line
    lines = []
    for i in range((int(line.length / distance)) + 1):
        save_line, current_line = Split(current_line, distance)
        lines.append(save_line)

    if crs == None:
        return lines

    out_frame = gpd.GeoDataFrame(geometry=lines, crs=crs)
    return out_frame


#* Multi-Geometry Parsing

# Turn a GDF into segmented gdf, parsed by distance
def Segment_GDF(gdf, segment_length):
    
    _ctype(gdf, gpd.GeoDataFrame, err_msg="Error (Segment_GDF): Input Frame must be a gpd.GeoDataFrame.", exit=True)

    all_segments = []
    for idx, line in enumerate(gdf.geometry):
        # Get list of segments for this line
        line_seg_list = Parse_by_Dist(line=line, distance=segment_length)

        # Append created segments to master list without increasing dimensionality
        for segment in line_seg_list:
            all_segments.append(segment)

    out_frame = gpd.GeoDataFrame(geometry=all_segments, crs=gdf.crs)
    return out_frame

def GDF_Buffer(gdf, buff_dist, flatten=False):

    gdf = _shapes(gdf, "GDF_Buffer")

    b_shapes = [geom.buffer(buff_dist) for geom in gdf.geometry]
    buff_frame = gpd.GeoDataFrame(geometry=b_shapes, crs=gdf.crs)

    if flatten:
        union_frame = gpd.GeoDataFrame(geometry=[buff_frame.geometry.unary_union], crs=gdf.crs) # Prevent re-recording overlapping values.
        buff_frame = union_frame.explode().reset_index(drop=True)

    return buff_frame
    
#* Geometry Rasterization

# With GeoDataframe and rio.DatasetReader, and out_dir, rasterize shapes into 1-band raster image with (0,1) values, (1 = labeled, 0=unlabeled)
def GDF_Rasterize(gdf, raster, out_path=None): 
    """ Rasterizes a GeoDataFrame (gdf) using the metadata of raster. out_path optional. """

    _ctype(gdf, gpd.GeoDataFrame, err_msg="Error (GDF_Rasterize): Input Frame must be a gpd.GeoDataFrame.", exit=True)
    _ctype(raster, rio.DatasetReader, err_msg="Error (GDF_Rasterize): Input raster must be a rio.DatasetReader.", exit=True)

    copy_frame = gdf.to_crs(raster.crs.data) 
    gdf_arr = features.rasterize(shapes=copy_frame.geometry, out_shape=raster.shape, fill=0, transform=raster.transform, all_touched=True, default_value=1)

    return gt.ND_Raster(gdf_arr, raster, out_path=out_path)


#* Multi-Geometry Normalization

# Convert a GeoDataFrame with messy linestrings into a clean one
def Flatten_Frame(gdf, newCrs=None):
    """ Convert MultiLinestrings in a GeoDataFrame into seperate Linestrings.""" 
    _ctype(gdf, gpd.GeoDataFrame, err_msg="Error (Flatten_Frame): Input Frame (gdf) must be a gpd.GeoDataFrame.", exit=True)
    
    out_shapes = []
    for idx, shape in enumerate(gdf.geometry):
        if isinstance(shape, shapes.LineString):
            # Reduces 3d coords to 2d coords
            out_shapes.append(LineString([xy[0:2] for xy in list(shape.coords)]))
        if isinstance(shape, shapes.MultiLineString):
            for line in shape:
                out_shapes.append(LineString([xy[0:2] for xy in list(line.coords)]))

    # If passed, assign crs
    if newCrs == None:
        out_frame = gpd.GeoDataFrame(geometry=out_shapes, crs=gdf.crs)
    else:
        out_frame = gpd.GeoDataFrame(geometry=out_shapes, crs=newCrs)
    return out_frame


#* Geometry Coordinate Manipulation 

def align_gdf(source_frame, fix_frame, crs_transformer=None):
    """ Use the pyproj transformer to align the geometries of two GeoDataFrames.

    Parameters:
    source_frame (GeoDataFrame): Holds target information.
    fix_frame (GeoDataFrame): Geometries to be aligned.
    crs_transformer (pyproj.Transformer): Optional transformer to greatly increase efficiency 

    Returns:
    GeoDataFrame: With fix_frame geometries, but source_frame coordinates.
    """
    source_frame = _shapes(source_frame, "align_gdf")
    fix_frame = _shapes(fix_frame, "align_gdf")
    
    # Create Coordinate Transformer
    if crs_transformer == None:
        trans = Transformer.from_crs(fix_frame.crs, source_frame.crs, always_xy=True)
    else:
        trans = crs_transformer

    outLines = []
    for line in fix_frame.geometry:        
        outLines.append(LineString([Point(trans.transform(crd[0], crd[1])) for crd in line.coords]))

    out_gdf = gpd.GeoDataFrame(geometry=outLines, crs=source_frame.crs)
    return out_gdf

def align_geom(source_frame, fix_shape, fix_crs):
    """ Use the pyproj transformer to align the geometries of a Linestring to a new CRS.

    Parameters:
    source_frame (GeoDataFrame): Holds target information.
    fix_shape (Linestring): Linestring to be aligned.
    fix_crs (CRS object): CRS to be converted from. 

    Returns:
    Geometry: With fix_shape geometry, but source_frame coordinates.
    """
    _ctype(source_frame, gpd.GeoDataFrame, err_msg="Error (align_geom): Input source_frame must be a gpd.GeoDataFrame.", exit=True)

    _ctype(fix_frame, LineString, err_msg="Error (align_geom): Input fix_shape must be a shapely Linestring.", exit=True)

    trans = Transformer.from_crs(fix_crs, source_frame.crs)

    points = []
    for crds in fix_shape.coords:
        xVal, yVal = crds
        y1, x1 = trans.transform(xVal, yVal)
        points.append(Point(x1,y1))
        
    return LineString(points)


#* Frame Analysis

def GDF_Area(gdf):
    """ Get the area of a GeoDataFrame containing Polygons."""
    _ctype(gdf, gpd.GeoDataFrame, err_msg="Error (GDF_Area): Input gdf must be a gpd.GeoDataFrame.", exit=True)
    area = 0
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            area += geom.area
    return area

def GDF_Intersect(gdf_a, gdf_b, buff=None):
    """ Create a frame holding the flattened intersect of two GeoDataFrames.
    
    Parameters:
    gdf_a (GeoDataFrame): GeoDataFrame holding Polygons or Linestrings.
    gdf_b (GeoDataFrame): GeoDataFrame holding Polygons or Linestrings.
    buff (float/int): Buffer to apply to both GeoDataFrames. (required if using Linestrings)

    Returns:
    GeoDataFrame: Contains Polygons containing the intersection of both GeoDataFrames.
    """
    _ctype(gdf_a, gpd.GeoDataFrame, err_msg="Error (GDF_Intersect): Input gdf_a must be a gpd.GeoDataFrame.", exit=True)
    _ctype(gdf_b, gpd.GeoDataFrame, err_msg="Error (GDF_Intersect): Input gdf_b must be a gpd.GeoDataFrame.", exit=True)
    if gdf_a.crs != gdf_b.crs:
        print("Error (GDF_Intersect): Input Frames do not have equal crs.")
        sys.exit(0)
    if buff != None:
        gdf_a = GDF_Buffer(gdf_a, buff)
        gdf_b = GDF_Buffer(gdf_b, buff)    
    
    all_shapes = []
    for geom in gdf_a.geometry:
        all_shapes.append(geom)
    for geom in gdf_b.geometry:
        all_shapes.append(geom)

    intersect_frame = gpd.overlay(gdf_a, gdf_b, how='intersection')
    intersect_geoms = intersect_frame.geometry.unary_union
    intersect_frame = gpd.GeoDataFrame(geometry=[intersect_geoms], crs=gdf_b.crs)
    intersect_frame.explode().reset_index(drop=True)
    
    return intersect_frame

def GDF_Union(gdf_a, gdf_b, buff=None):
    """Take two GeoDataFrames with Linestrings, return a GeoDataFrame with the union of the two frames after buffering.
    
    Parameters:
    gdf_a (GeoDataFrame): GeoDataFrame holding Polygons or Linestrings.
    gdf_b (GeoDataFrame): GeoDataFrame holding Polygons or Linestrings.
    buff (float/int): Buffer to apply to both GeoDataFrames. (required if using Linestrings)

    Returns:
    GeoDataFrame: Contains Polygons containing the area of both GeoDataFrames.
    """
    _ctype(gdf_a, gpd.GeoDataFrame, err_msg="Error (GDF_Union): Input gdf_a must be a gpd.GeoDataFrame.", exit=True)
    _ctype(gdf_b, gpd.GeoDataFrame, err_msg="Error (GDF_Union): Input gdf_b must be a gpd.GeoDataFrame.", exit=True)
    if gdf_a.crs != gdf_b.crs:
        print("Error (GDF_Union): Input Frames do not have equal crs.")
        sys.exit(0)
    if buff != None:
        gdf_a = GDF_Buffer(gdf_a, buff)
        gdf_b = GDF_Buffer(gdf_b, buff)

    all_shapes = []
    for geom in gdf_a.geometry:
        all_shapes.append(geom)
    for geom in gdf_b.geometry:
        all_shapes.append(geom)
    
    gdf = gpd.GeoDataFrame(geometry=all_shapes, crs=gdf_b.crs)

    union_geoms = gdf.geometry.unary_union # Prevent re-recording overlapping values.
    union_frame = gpd.GeoDataFrame(geometry=[union_geoms], crs=gdf.crs)
    union_frame.explode().reset_index(drop=True)

    return union_frame

def CropGDF(gdf, raster, strict=True):
    """ Takes gdf, removes values that are not on raster.

    Parameters:
    gdf (GeoDataFrame or .shp path): Geometries to be cropped to the extent of the raster.
    raster (rio.DatasetReader): Raster defining target extent.
    strict (bool): Defines whether shapes must exist entirely within the bounds of the raster. (default=True)

    Returns:
    gpd.GeoDataFrame: Containing only geometries that lie within the raster's extent.
    """
    gdf = _shapes(gdf, 'CropGDF')
    raster = _raster(raster, 'CropGDF')

    # Get bounding box in coordinate ref
    (mx,my,Mx,My) = raster.bounds
    boundingPoly = box(mx,my,Mx,My)

    # Convert Polygon Coordinates
    trans = Transformer.from_crs(raster.crs, gdf.crs)
    points = []
    for x, y in boundingPoly.exterior.coords:
        y1, x1 = trans.transform(x,y)
        points.append(Point(x1, y1))
    window = Polygon(points)

    # Save Frame
    metadata = {key: [] for key in gdf.columns}
    for idx, geom in enumerate(gdf.geometry):
        if strict == True:
            if window.contains(geom):
                for key, data in gdf.items():
                    metadata[key].append(data[idx])
        else:
            if window.intersects(geom):
                for key, data in gdf.items():
                    metadata[key].append(data[idx])
    if len(metadata['geometry']) == 0:
        print("Error (CropGDF): None of the target geometries exist on the raster. Checked {}.".format(len(gdf.geometry)))
        sys.exit(0)

    return gpd.GeoDataFrame(metadata, crs=gdf.crs)
