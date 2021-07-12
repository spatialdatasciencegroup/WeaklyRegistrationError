import os
import time
import keras
import numpy as np
import pandas as pd
import random as rand
import rasterio as rio
import geopandas as gpd
import rasterio.mask as msk
import rasterio.windows as win
import matplotlib.pyplot as plt
from shapely.ops import substring
import shapely.geometry as shapes
from pyproj import Proj, Transformer, transform
from shapely.affinity import translate
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import linemerge
import numpy.ma as ma
from lib.internal import _shapes, _raster
from rasterio.errors import WindowError

# custom modules
from lib.internal import _ctype
import lib.GeoTools as gt
import lib.ShapeTools as st
import lib.LineGen as lg
from lib.fixclip import clip as clp


#* Package to prepare and apply the EM algortithm to candidate lines
#* Functions ordered by usage sequence.

#? Preparing Data

def Get_Pmap(source_raster, win_shape, pmodel, pmap_fp):
    """ 
    Use a trained U_Net model to generate a probability output map covering the entire raster. 
    Tiles input raster, runs predictor on tiled windows, then appends the output windows, covering nearly the entire raster.

    Parameters:
    source_raster (DatasetReader): 7D Input raster to run prediction on. 
    win_shape (tuple): Pixel (height, width) of the inputs for predictor model.
    out_path (string): Directory to save pmap.
    pmap_name (string): Filename to save pmap as
    pmodel (keras.model): Model to run predictions with.

    Returns:
    DatasetReader: 1D Raster representing full probability output map.
    """
    (win_height, win_width) = win_shape

    inputs = gt.Tile(source_raster, win_width, win_height)
    cl_inputs = np.rollaxis(inputs, 1, 4)

    predictions = pmodel.predict(cl_inputs)

    cols = int((source_raster.width / win_width))
    rows = int((source_raster.height / win_height))

    p_rows = []
    k = 0 # iterator
    for i in range(rows):
        this_row = []
        for j in range(cols):
            # Where the U_Net predictor is called
            this_row.append(np.squeeze(predictions[k])) # Output array Shape == (win_height, win_width) could add [0] to avoid dim1 wrapper (1, win_height, win_width)
            k += 1
        p_rows.append(np.hstack(this_row)) # stack row pixel arrays 
    pmap = np.vstack(p_rows) # Shape == (src_height, src_width) ONLY, expand dims done later
    
    return gt.ND_Raster(arr=pmap, source_raster=source_raster, out_path=pmap_fp)


#? Weighting Candidate Segments



def CandidateWindows(raster, candidate_gdf, buff, convert_crs=True):
    """ Rasterize Candidate Segments, output referencable windows for each segment. """
    
   
    raster = _raster(raster)
    candidate_gdf = _shapes(candidate_gdf)

    # modify crs without corrupting original geometry.
    if convert_crs:
        candidate_gdf = candidate_gdf.to_crs(raster.crs)
    copy_gdf = st.GDF_Buffer(candidate_gdf, buff)

    masks = []  # Stores numpy mask from buffered label.
    windows = [] # Stores rasterio windows for each candidate, or void for candidates outside the raster.
    for idx, geom in enumerate(copy_gdf.geometry):

        geo_mask, _, geo_win = msk.raster_geometry_mask(raster, [geom], all_touched=True, crop=True)
        masks.append(geo_mask)
        windows.append(geo_win)
    
    metadata = {'mask': masks, 'window': windows}
    geo_frame = gpd.GeoDataFrame(metadata, geometry=candidate_gdf.geometry, crs=candidate_gdf.crs) 

    return geo_frame

def WeightCandidates(pmap, candidate_gdf):
    """ Reads Linestring data from raster using window and mask metadata. """
    pmap = _raster(pmap, "WeightCandidates")
    candidate_gdf = _shapes(candidate_gdf, "WeightCandidates")

    weights = []
    for idx in range(len(candidate_gdf.geometry)):
        # Read pmap for pixel values in linestring bounds
        pmap_data = np.squeeze(pmap.read(window=candidate_gdf['window'][idx]))
        
        # mask pmap array with geometry mask
        masked_data = ma.array(pmap_data, mask=candidate_gdf['mask'][idx])
        
        # Read non-masked values, compute mean
        px_values = [val for val in masked_data.flatten() if not isinstance(val, np.ma.core.MaskedConstant)]
        weights.append(sum(px_values)/len(px_values))
    
    df = pd.DataFrame({
        'weight': weights,
        'window': candidate_gdf['window'],
        'mask': candidate_gdf['mask']
    })
    return gpd.GeoDataFrame(df, geometry=candidate_gdf.geometry, crs=candidate_gdf.crs)

#? Select and append Optimal Candidate Lines

def Top_Candidate(weighted_frame, meta_key, default_random=True):
    """ Select the candidate with the highest weight from weighted_frame.

    Parameters:
        weighted_frame (gpd.GeoDataFrame): Has float/int-labeled geometries.
        meta_key (str): Name of the collumn storing the weight to evaluate.
        default_random (bool): Optionally select a random candidate in zero-weighted areas of the predicted class map. 

    Returns:
    shp.geometry object: The top weighted candidate. If none of the candidates have weight, return the first geometry.
    """
    
    # Check for empty weights
    if sum(weighted_frame[meta_key]) <= 1e-4:
        if default_random:
            # Select random candidate in empty predicted class map
            return np.random.choice(weighted_frame.geometry)
        elif False: # Saving for later implementation, never called
            # default to original
            return weighted_frame.geometry[0]
    
    """ AAAI Setup """
    # sort by weight
    weighted_frame = weighted_frame.sort_values(meta_key).reset_index()
    return weighted_frame.geometry[len(weighted_frame)-1]

    
def Select_Candidate(weighted_frame, meta_key, weight_limit=0, default_random=True):
    """ Select a 'likely' candidate from weighted_frame, the higher the candidate's weight, the more likely it will be selected. Note that input frame must be sorted by meta_key collumn.

    Parameters:
    weighted_frame (gpd.GeoDataFrame): Has float/int-labeled geometries, sorted by weight (ascending).
    meta_key (string): Name of the collumn storing the weight to evaluate.
    limit (int):

    Returns:
    gpd.GeoDataFrame: Contains a likely candidate for every segment.
    """
   
    
    # Check for empty weights
    if sum(weighted_frame[meta_key]) <= 1e-4:
        if default_random:
            # Select random candidate in empty predicted class map
            return np.random.choice(weighted_frame.geometry)
        elif False: # Saving for later implementation, never called
            # default to original
            return weighted_frame.geometry[0]

       
    """ AAAI Setup """
    # sort by weight
    weighted_frame = weighted_frame.sort_values(meta_key)
    
    # remove the lowest weights if needed
    if weight_limit != 0:
        weighted_frame = weighted_frame[-weight_limit:] 
    
    # Get total weight and a random float within the range
    total_weight = sum(weighted_frame[meta_key])
    n = rand.uniform(0, total_weight)
    
    # Select Linestring
    for weight, geom in zip(weighted_frame[meta_key], weighted_frame.geometry):
        if n < weight:
            return geom
        n = n - weight
    
    # if none selected, return the top weight
    return geom



def Mean_Precisions(weighted_frames, meta_key):
    """ Takes GeoDataFrames with weight-labeled geometries, compares the target metadata, finds the top candidate for each segment, returns a single frame.

    Parameters:
    weighted_frames (gpd.GeoDataFrame): Has float/int - labeled geoms.
    meta_key (string): Name of the collumn storing the weight to evaluate.

    Returns:
    list of floats: Contains the average weight for candidates in that segment.
    """
    seg_avgs = []
    for idx, frame in enumerate(weighted_frames):
        total_weight = 0
        for weight in frame[meta_key]:
            total_weight += weight 
        seg_avgs.append(total_weight / len(frame[meta_key]))
    return seg_avgs


#? Other Tools

def Connect_Lines(gdf, max_dist): 
    """ Creates a geodataframe of clean annotations from a geodataframe of broken line segments, 
    
    Parameters:
    gdf (GeoDataframe): Holds disconnected linestring segments.
    max_dist (int/float): Represents the furthest 'connectable' distance between two linestrings.

    Returns:
    GeoDataframe: with a smaller number of longer linestrings that represent the same annotation.
    """

    # Initialize holder of all lines, add first entry, start iterator (number of connected segments)
    new_lines = []
    new_lines.append([])
    k = 0
    for idx, geom in enumerate(gdf.geometry):

        # If line_n is the last in the frame, add it to the last list and end the loop.
        if idx+1 == len(gdf.geometry):
            new_lines[k].append(geom)
            break

        line_n = geom 
        line_m = gdf.geometry[idx+1]
        dist = line_n.distance(line_m)

        if (dist < max_dist):                
            new_crds = []
            n_last_crd = Point(line_n.coords[len(line_n.coords)-1])
            n_second_last_crd = Point(line_n.coords[len(line_n.coords)-2])
            m_first_crd = Point(line_m.coords[0])

            if (n_last_crd.distance(m_first_crd) > n_second_last_crd.distance(m_first_crd)) and (len(line_n.coords) > 2):
                for idx, crd in enumerate(line_n.coords):
                    if idx+2 < len(line_n.coords):
                        new_crds.append(crd)                
            else:
                for idx, crd in enumerate(line_n.coords):
                    if idx+1 != len(line_n.coords):
                        new_crds.append(crd)

            new_crds.append(line_m.coords[0])
            line_n = LineString(new_crds)
            new_lines[k].append(line_n)
        else:
            new_lines[k].append(line_n)
            new_lines.append([])
            k += 1

    new_line_set = []
    for line_list in new_lines:
        all_coords = []
        for line in line_list:
            for crd in line.coords:
                all_coords.append(crd)

        new_line_set.append(LineString(all_coords))


    fixed_frame = gpd.GeoDataFrame(geometry=new_line_set, crs=gdf.crs)

    return fixed_frame



def GDF_Precision(ref_frame, focus_frame, buff):
    """Takes a frame of Linestrings to compare to the ground truth, returning the precision (overlay/gt_area).

    Parameters:
    ref_frame (GeoDataFrame): Ground Truth Annotation.
    focus_frame (GeoDataFrame): Candidate Annotation to be evaluated.
    buff (float): Buffer in arc degrees to apply when considering overlay.

    Returns:
    float: 0-1 precision value (portion of the ground_truth annotation covered by the ref_frame).
    """
    _ctype(ref_frame, gpd.GeoDataFrame, err_msg="Error (emt.GDF_Precision): Input ref_frame must be a gpd.GeoDataFrame.", exit=True)
    _ctype(focus_frame, gpd.GeoDataFrame, err_msg="Error (emt.GDF_Precision): Input focus_frame must be a gpd.GeoDataFrame.", exit=True)
    
    gt_buff = st.GDF_Buffer(ref_frame, buff, flatten=True)

    aligned_fframe = st.align_gdf(ref_frame, focus_frame)
    buff_fframe = st.GDF_Buffer(aligned_fframe, buff, flatten=True)

    intersection = 0
    for idx,buff_line in enumerate(buff_fframe.geometry):
        intersect_area = 0
        for gt_poly in gt_buff.geometry:
            intersect_poly = buff_line.intersection(gt_poly)
            if intersect_poly.area > intersect_area:
                intersect_area = intersect_poly.area
        if intersect_area > 0:
            intersection += intersect_area
    gt_area = st.GDF_Area(gdf=gt_buff)
    return (intersection/gt_area)


def LabelFramePrecision(ref_frame, focus_frame, buff, crs_transformer=None):
    """Takes a GeoDataFrame to compare to the ground truth, returns a new GeoDataframe with each geometry's precision based on overlay with the ground truth.

    Parameters:
    ref_frame (GeoDataFrame): Ground Truth Annotation .
    focus_frame (GeoDataFrame): Frame with Linestrings whose quality will be assessed.
    buff (float): Buffer in arc degrees to apply when considering overlay. 
    crs_transformer (pyproj.Transformer): Optional transformer to greatly increase efficiency 

    Returns:
    GeoDataFrame: Contains two collumns:
        geometry: original segment geometries in original crs
        quality: 0-1 value representing what portion of the (buffed) shape lies withing the (buffed) ground_truth annotation.
    """

    _ctype(ref_frame, gpd.GeoDataFrame, err_msg="Error (emt.LabelFramePrecision): Input ref_frame must be a gpd.GeoDataFrame.", exit=True)
    _ctype(focus_frame, gpd.GeoDataFrame, err_msg="Error (emt.LabelFramePrecision): Input focus_frame must be a gpd.GeoDataFrame.", exit=True)

    gt_buff = st.GDF_Buffer(ref_frame, buff, flatten=True)
    aligned_fframe = st.align_gdf(ref_frame, focus_frame, crs_transformer=crs_transformer)

    total_overlay = 0
    precision_vals = []
    for idx,line in enumerate(aligned_fframe.geometry):
        buff_line = line.buffer(buff)
        intersect_area = 0
        for gt_poly in gt_buff.geometry:
            intersect_poly = buff_line.intersection(gt_poly)
            if intersect_poly.area > intersect_area:
                intersect_area = intersect_poly.area
        if intersect_area > 0:
            precision_vals.append(intersect_area / buff_line.area)
        else:
            precision_vals.append(0.0)

    df = pd.DataFrame({'precision': precision_vals})
    out_frame = gpd.GeoDataFrame(df, geometry=focus_frame.geometry, crs=focus_frame.crs)
    return out_frame

from lib.UnetTools import *

def Evaluate(X_tensor,Y_tensor, model):
    y_pred=model.predict(X_tensor)
    y_true=Y_tensor

    f1_value = f1_score(tf.cast(y_true, tf.float32),tf.cast(y_pred, tf.float32)) 
    dice_value = dice_coef(tf.cast(y_true, tf.float32),tf.cast(y_pred, tf.float32))

    return ['Dice Coef: {:.3f}%'.format(float(dice_value)*100), 'F1 Score: {:.3f}%'.format(float(f1_value)*100)]
