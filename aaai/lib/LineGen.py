import os, sys
import numpy as np
import pandas as pd
import random as rand
import rasterio as rio
import geopandas as gpd
import rasterio.windows as win
import matplotlib.pyplot as plt
from shapely.ops import substring
import shapely.geometry as shapes
from pyproj import Proj, transform
from shapely.affinity import translate
from shapely.geometry import Point, LineString, Polygon, box

import lib.GeoTools as gt
import lib.ShapeTools as st
from lib.fixclip import clip as clp
from lib.internal import _ctype

# To circumnavigate geopandas shapeTile issue, some non-critical errors have to be silenced.
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

# Module to analyze existing annotations and generate candidate lines based on their metadata.

#* Components / Line Analysis

def Get_Endpoints(line, asPoints=False):
    """ Get the coordinate endpoints of shapely Linestring. """
    _ctype(line, LineString, err_msg='Error (Get_Endpoints): Input line must be a shapely Linestring.', exit=True)
    if asPoints:
        return (Point(line.coords[0]), Point(line.coords[len(line.coords)-1]))
    else:
        return (line.coords[0], line.coords[len(line.coords)-1])


def Get_Slope(line, asLine=False):
    """ Get the slope of a shapely Linestring. """
    _ctype(line, LineString, err_msg='Error (Get_Slope): Input line must be a shapely Linestring.', exit=True)
    
    (x1, y1), (x2, y2) = Get_Endpoints(line) 
    line_slope = (y1 - y2)/(x1 - x2)

    if asLine:
        return LineString(Get_Endpoints(line, asPoints=True))

    return line_slope

def Trailing_Length(gdf, dist):
    """ Find the excess length on a GeoDataFrame of Linestrings. """
    _ctype(gdf, gpd.GeoDataFrame, err_msg="Error (Trailing_Length): Input gdf must be a gpd.GeoDataFrame.", exit=True)

    trailing_length = 0
    for idx, line in enumerate(gdf.geometry):
        if not _ctype(gdf, LineString, "Error (Trailing_Length): Input gdf must be a gpd.GeoDataFrame with only Linestrings."):
            sys.exit(0)
        if line.length < dist:
            trailing_length += line.length
        else:
            trailing_length += (line.length % dist)
    return trailing_length




#* Offset Coordinate Tuple Generation 

# Weight dist float/int based on bounding box dims
def Weight_Dist(line, dist):

    # Get bounding box coordinates, regardless of crs 
    minx, miny, maxx, maxy = line.bounds
    
    # Get bounding box dimensions    
    width = (maxx - minx)
    height = maxy - miny

    # Calculate Weights
    x_weight = height/width
    y_weight = width/height

    # normalize values, preventing them from throwing dist too far off
    xVal = x_weight / (y_weight + x_weight)
    yVal = y_weight / (y_weight + x_weight)

    return ((dist*xVal),(dist*yVal))

# Generate properly parrallel candidate Lines, without checking if the segment is linear, then determining offset tuple based on bounding box dimensions
def Off_Coords(line, n, dist, rmax, exp=False, use_rand=False):

    _ctype(line, LineString, err_msg="Error (Off_Coords): Input line must be a shapely Linestring.", exit=True)
    
    # random iterator always increases offet.
    while rmax < 1: 
        rmax += 1


    offsets = []
    if Get_Slope(line) > 0:
        for i in range(n):
            (xVal, yVal) = Weight_Dist(line=line, dist=dist) # Update Weights for every pair
            offsets.append((-xVal, yVal))
            offsets.append((xVal, -yVal))   
            if use_rand:              
                if exp:    
                    dist *= rand.uniform(1, rmax) 
                else: 
                    dist += rand.uniform(1, rmax) 
            else:
                if exp:    
                    dist *= rmax 
                else: 
                    dist += rmax
    else:

        for i in range(n):

            (xVal, yVal) = Weight_Dist(line=line, dist=dist)
            offsets.append((xVal, yVal))
            offsets.append((-xVal, -yVal))                 
            if use_rand:
                if exp:    
                    dist *= rand.uniform(1, rmax) 
                else: 
                    dist += rand.uniform(1, rmax) 
            else:
                if exp:    
                    dist *= rmax 
                else: 
                    dist += rmax
    return offsets


#* Geometry Transposing


# Takes Tuples and generaetes candidate lines
def Shift_Line(line, dirs, crs):
    """ Export as a geoseries filled with candidate line proposals. """
    # Validate source line dtype
    _ctype(line, LineString, err_msg="Error (Shift_Line): Input line must be a shapely Linestring.", exit=True)

    # Initialize list to hold this segment's candidates
    candidates = []

    # Include source line 
    candidates.append(line)

    for (x,y) in dirs:
        nextLine = translate(line, xoff=x, yoff=y)
        candidates.append(nextLine)

    out_frame = gpd.GeoDataFrame(geometry=candidates, crs=crs)

    return out_frame


#* Candidate Line Generation


# Generate candidates for a GeoDataframe filled with line segments, output as a list of GDFs organized by candidate index
def Get_Candidates(seg_gdf, n, dist, rmax, exp_increase=False, use_rand=False):
    
    _ctype(seg_gdf, gpd.GeoDataFrame, err_msg="Error (Trailing_Length): Input seg_gdf must be a gpd.GeoDataFrame.", exit=True)
    # Will store one GeoDataframe for each segment, holding that segment's candidates
    seg_frames = []

    # Iterate over linestring segments in seg_gdf
    for line in seg_gdf.geometry:

        # Validate segment Geometry Type
        if not isinstance(line, LineString):
            print("Get_Candidates Error: Invalid Datatype for Source line. Need shapely.Linestring.")
            print("Passed Type:", type(line))
            return line

        # Generate a list of offsets for this segment
        seg_offsets = Off_Coords(line=line, n=n, dist=dist, rmax=rmax, exp=exp_increase, use_rand=use_rand)

        # GDF of candidates for this segment 
        seg_candidates = Shift_Line(line, seg_offsets, crs=seg_gdf.crs)

        # Master list of candidate gdframes. 
        seg_frames.append(seg_candidates) # Length = (N - number of segments)

    return seg_frames


#* Formatting, Training


# Takes a list of geodataframes, each filled with candidates for a single segment, returns a list of geodataframes, each with a candidate for every segment
def Group_Candidates_by_Index(input_frames):
    """ Takes Candidates sorted by segment, returns candidates sorted by candidate index.
    # input_frames list - 
    # -- frame_count :     (N - number of segments)
    # -- geoms_per_frame : (C - canididates per segment)

    # output_frames list - 
    # -- frame_count :     (C - canididates per segment)
    # -- geoms_per_frame : (N - number of segments)
    """

    new_geoms = []
    
    for idx, frame in enumerate(input_frames):
        _ctype(frame, gpd.GeoDataFrame, err_msg="Error (Group_Candidates_by_Index): Input list must contain only gpd.GeoDataFrame.", exit=True)
        for sidx, shape in enumerate(frame.geometry):
            if idx == 0:
                new_geoms.append([]) # on first iteration, must create empty lists
            new_geoms[sidx].append(shape) # add shape to shape index list in new_geoms list

    output_frames = []

    for geoms in new_geoms:
        output_frames.append(gpd.GeoDataFrame(geometry=geoms, crs=input_frames[0].crs))

    return output_frames



def LoadCandidates(CandidateDirectory):
    """ Load Candidates by segment from a candidate directory. """
    candidateFrames = []
    for fname in os.listdir(CandidateDirectory):
        _, ext = os.path.splitext(fname)
        if ext == '.shp':
            candidateFrames.append(gpd.read_file(os.path.join(CandidateDirectory, fname)))
    return candidateFrames