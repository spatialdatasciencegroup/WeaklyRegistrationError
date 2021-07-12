import os, sys, time, math, random


import numpy as np

import numpy.ma as ma

import geopandas as gpd
import rasterio as rio

import rasterio.features as rfeat
import rasterio.windows as rwin


import shapely.geometry as shp

from lib.annotators.Annotator import Annotator, anno_kwargs_key

"""
New candidate generation system using coordinate point shifting.

Rewrite to omit segment shifting, front-loading candidate generation, 
and facilitate future GAN implementation.

Massively simplified/quickened operations for offsets*

"""

cma_kwargs_key = {
    'annotator': ("(Annotator [None])", "Annotator to configure parent parameters by. If no annotator is passed, will use annotator default."),
    'pairs': ("(int [10])", "Number of coordinate pairs to be generated for each candidate grouping."),
    'off_dist': ("(int, meters [1])", "Base distance  to offset candidate coordinate pairs."),
    'outer_pairs': ("(int [0])", "Outer coordinate pairs generated with multiplicative offsets."),
    'outer_multiplier': ("(float [0.2])", "Multiplier used to increase outer pair distance cumulatively."),
    'minimum_weight': ("(float [1e-5])", "The lowest weight accepted for a candidate to be selected. Prevents jagged selection for unweighted areas in the predicted class map."),
    'use_origin_default': ("(bool [False])", "Option to use source point as default for groups with no weight on the predicted class map."),
    'epsilon': ("(float [0.02])", "Aka 'random_select_chance', probability that a random candidate will be selected in from each candidate group within a range of the top weighted point. Range defined by 'k'/'random_select_range'."),
    'k': ("(int [4])", "Aka 'random_select_range', number of neighbors on either side of the top point that will be considered for random selection."),
}

cma_kwargs_key.update(("Annotator - {}".format(key), item) for key, item in anno_kwargs_key.items())

class Class_Map_Annotator(Annotator):
    """ 
    Uses a predicted class map represented by a rasterio.DatasetReader raster 
    to improve shapely.geometry.Linestring annotations.
    
    Each set of candidate coordinates held in a shapely.geometry.MultiPoint, 
    each Linestring's total candidates represented by a gpd.GeoDataFrame.
    
    Candidate Generation and weighting configuration stored in class attributes.

    Attributes:
        pairs (int):        Number of coordinate pairs to be generated 
                                on either side of source annotation.
        off_dist (int):     Offset distance for each candidate point.  
        outer_pairs (int):  Number of coordinate pairs to be generated
                                with multiplicative increment.
        outer_multiplier (float):   Multiplier used to increment outer pairs.
        minimum_weight (float):     Minimum weight for candidate update
        coordinate_precision (float):  Decimal coordinate_precision used to compare coordinates.
        verbosity (0,1):    Determines verbosity for printing indexes.
        use_origin_default (bool [True]):   If True, always use origin as default candidate.
                                            If False, use previous index as default candidate 
        
        random_select_chance (float [0.02]):    Chance to use random candidate selection for each candidate group.
        random_select_range (int [4]):          Distance from top coordinate point that a random candidate may be selected. 

    """
    def __init__(self, **kwargs):
        """ 
        Initialize class map annotator. 
        
        Notes: 
            If annotator is passed in keys, update parameters from that annotator's configuration. Otherwise expects the passed parameters. 

        """

        # Initialize annotator
        if 'annotator' in kwargs.keys():
            super().__init__(**kwargs['annotator'].__dict__)
        else:
            super().__init__(**kwargs)
        
        # Default Attributes
        default_attr = dict(
            pairs = 10,                 # pairs to be generated on either side linearly.
            off_dist = 1,                   # offset dist incrementable (meters)
            outer_pairs = 0,            # Outer  pairs to be generated multiplicatively       
            outer_multiplier = 0.1,     # Multiplier used to increase outer coordinate pair offset. 
            minimum_weight = 1e-5,      # Minimum accepted candidate weight
            coordinate_precision = 1e-4,# Float precision for coordinate comparison 
            line_buffer = 3,            # Buffer to apply to candidate linestrings when weighting 
            point_buffer = 3,           # Buffer to apply to starting coordinate before weighting
            use_origin_default = True,  # Optionally default coordinate selection to source point
            epsilon = 0.02, # Chance to select a random coordinate from top points neighbors
            k = 4,    # Range on either side of top point to select random candidate
            kwargs_key = cma_kwargs_key
        )

        # Allowed Attributes
        allowed_attr = list(default_attr.keys())

        # Update with passed arguments
        default_attr.update(kwargs)        
        
        # Update kwargs
        self.__dict__.update((k,v) for k,v in default_attr.items() if k in allowed_attr)

    ### Visualization and Information

    def get_candidate_range(self):
        """ Returns the maximum range that a candidate may be generated. Ceiling function applied to final value. (Accurate) """
        
        dist = 0

        for i in range(self.pairs + self.outer_pairs):
            
            # increment
            dist += self.off_dist

            # if outer_pairs are passed, exponentially apply increase
            if i > self.pairs:
                dist += (self.outer_multiplier*(i-self.pairs))
                dist = np.round(dist, 4)
        
        return int(np.ceil(dist))

    ### CANDIDATE GENERATION

    def _point_set(self, point, prev_p, next_p):
        """ 
        Generates a set of candidate points for a single coordinate point. 
        
        Saves offset from source annotation, returning points and respective offset distance as dict.
        
        Args:
            point (shp.Point):  Target point Designed to generate candidate set.
            prev_p (shp.Point): Previous point in annotation line sequence. Used to determine slope.
            next_p (shp.Point): Next point in annotation line sequence. Used to determine slope.

        Returns:
            
            shp.MultiPoint: holds all candidate points for the source point.
                            The first point in the multipoint is always the source, 
                            points increase in distance from the source after that.    
        """

        # Get offset weights
        off_x, off_y = self.get_offsets(prev_p, next_p)

        # Initialize set with source point
        candidates = [point]

        # Initialize offset distance
        dist = 0

        for i in range(self.pairs + self.outer_pairs):
            
            # increment
            dist += self.off_dist

            # if outer_pairs are passed, exponentially apply increase
            if i > self.pairs:
                dist += (self.outer_multiplier*(i-self.pairs))
                dist = np.round(dist, 3)
        
            # Create coordinate offsets for each axis
            x_dist = off_x * dist
            y_dist = off_y * dist

            # Append points and their respective offset tuples
            candidates.append(shp.Point(((point.x + x_dist), (point.y - y_dist))))
            candidates.append(shp.Point(((point.x - x_dist), (point.y + y_dist))))

        # Filter out geometries that do not lie on raster
        for idx, point in enumerate(candidates):
            if not self.crop_window.contains(point):
                del candidates[idx]

        # VERBOSITY
        if self.verbosity == 1:
            self.coordinate_index += 1
            print("created new coordinate group, index:", self.coordinate_index)

        # May remove, empty sets should never appear if other systems work properly
        if len(candidates) == 0:
            print("WARNING: Empty set of candidate coordinates created.")    
            return None

        # Convert list of points to shp.MultiPoint
        candidates = shp.MultiPoint(candidates)

        return candidates
        

    def _generate_points(self, line):
        """ Generates candidate groups over a line from internal configuration. 
        
        Args:
            line (shp.Linestring): Linestring to generate candidate points over.
        
        Returns:
            list(shp.MultiPoint): list of candidate point groups in set, 
                                  including origin and terminating groups.
        """

        if not isinstance(line, shp.LineString):
            print("Annotator Error (generate_points): Invalid source line type, expected: {}       Passed: {}\n\n".format(type(shp.LineString), type(line)))


        origin = shp.Point(line.coords[0])
        target_point = line.interpolate(self.interval)

        # Initialize list of candidate groups with origin candidates.
        candidate_groups = [self._point_set(origin, origin, target_point)]

        i = 0
        while True:
            # prev_p starts as origin, target point starts as index 1
            prev_p, this_p, next_p = line.interpolate(self.interval*i), line.interpolate(self.interval*(i+1)), line.interpolate(self.interval*(i+2))
            i += 1

            candidate_groups.append(self._point_set(this_p, prev_p, next_p))
            
            # if the current target point has the same coordinates as the endpoint of the line, break
            if self.same_coords(this_p.coords[0], line.coords[-1]):
                break

        # VERBOSITY
        if self.verbosity == 1:
            self.line_index += 1
            print("Coordinate Set generation complete, index:", self.line_index)
            print("Total coordinate sets created:", self.coordinate_index)
            self.coordinate_index = 0

        # filter empty groups
        candidate_groups = [x for x in candidate_groups if x != None]
        return candidate_groups



    ### CANDIDATE SELECTION

    def _roll_random_candidate(self, source_idx, candidate_group, force=False):
        """ Selects a random candidate from N neighboring candidates. 
        
        Args:
            source_idx (int): Index of top candidate previously selected. 
                                Used to base random group indicies around
            candidate_group (shp.MultiPoint): Candidate group of points to sample random selection from.
            force (bool):   Optionally force a random candidate to be selected. (Default: False)
        
        Return:
            int: optimal point's index on the group after randomness applied, 
                 or 
                 top point's index if randomness is not applied. 
        """
        # Check if parameters were configured

        # 'roll' for random selection
        if (random.uniform(0, 1) <= self.epsilon) or force:
            
            # List of indices for each potentially selected point, starting with the original point.
            random_group = [source_idx]
            
            for k in range(self.k):
                # One point for either side
                index_a = source_idx + 2*(k+1)
                index_b = source_idx - 2*(k+1)

                if (index_b < 0): # Weird indexing math here, I feel smart tho 
                    index_b = abs(index_b + 1)

                # If the index is valid, add to random group for consideration
                if index_a < len(candidate_group):
                    random_group.append(index_a)
                
                if index_b < len(candidate_group):
                    random_group.append(index_b)
        
            selected_index = np.random.choice(random_group)

            return selected_index

        # If any check fails, return passed parameters
        return source_idx



    def _select_candidate(self, anchor, candidate_group, class_map, default_point_idx=0):
        """ Weight candidate lines between anchor point and potential successors from 
            predicted class map as a raster, then select optimal point with randomness.

        Args:
            anchor (shp.Point):             Shapely point selected from previous set.
            candidate_group (shp.MultiPoint):         Group of points to connect with anchor.
            class_map (rio.DatasetReader):  Predicted Class map raster
            default_point_idx (int):        Index of default point in the case 
                                            where all weights are below minimum threshold.
        
        Returns:
            int: optimal point index to continue annotation
        """

        if not isinstance(anchor, shp.Point):
            raise TypeError("Class_Map_Annotator.select_candidate(): Expected type '{}' for anchor. Recieved '{}'.".format(shp.Point.__name__, type(anchor).__name__))
        if not isinstance(candidate_group, shp.MultiPoint):
            raise TypeError("Class_Map_Annotator.select_candidate(): Expected type '{}' for candidate_group. Recieved '{}'.".format(shp.MultiPoint.__name__, type(candidate_group).__name__))
        if not isinstance(class_map, rio.DatasetReader):
            raise TypeError("Class_Map_Annotator.select_candidate(): Expected type '{}' for class_map. Recieved {}.".format(rio.DatasetReader.__name__, type(class_map).__name__))

        # Make a two-coordinate linestrings connecting the anchor to each potential point
        candidate_lines = [shp.LineString([anchor, point]).buffer(self.weight_buffer) for point in candidate_group.geoms]
        
        # Window to read candidate group from raster
        group_window = rfeat.geometry_window(class_map, candidate_lines)

        # read pixel data from class map using geometry window
        group_data = class_map.read(1, window=group_window)

        for idx, candidate_line in enumerate(candidate_lines):

            # Create a boolean mask representing the linestring from the windowed data for this candidate group.
            mask = rfeat.geometry_mask([candidate_line], out_shape=group_data.shape, transform=rwin.transform(group_window, class_map.transform), all_touched=True)

            # Apply boolean mask to raster's data
            masked_data = ma.array(group_data, mask=mask)

            # Calculate mean weight for segment
            weight = masked_data.sum() / ma.compressed(masked_data).shape[0]


            # update if weights are increased / Initiallize top point as source 
            if ( idx == 0 ) or (weight > top_weight):
                top_weight = weight
                top_idx = idx
        
        
        # If weight is lower than the minimum accepted weight, use default point as passed. 
        if (top_weight < self.minimum_weight): 
            # If the passed default index lies outside of the group (due to cropping), set origin as default
            if default_point_idx >= len(candidate_group):
                default_point_idx = 0
            top_idx = default_point_idx   
        
        # Introduce randomness to selection NOTE: could limit to points with valid weight^
        return self._roll_random_candidate(source_idx=top_idx, candidate_group=candidate_group)


    def _select_origin(self, origin_group, class_map):
        """ Selects starting point from first group in candidate group sequence.
        
        Args:
            origin_group (shp.MultiPoint): Group of points to select origin point from.
            class_map (rio.DatasetReader): Predicted class map as raster to read weights from.

        Returns:
            int: optimal starting point's index based on predicted class map weights.
        """

        # Buffer points into polygons for evaluation
        buffed_points = [point.buffer(self.point_buffer) for point in origin_group]

        
        for idx, point in enumerate(buffed_points):
            if not self.crop_window.contains(point):
                del buffed_points[idx]

        # Window to read candidate group from raster
        group_window = rfeat.geometry_window(class_map, buffed_points)

        # read pixel data from class map using geometry window
        group_data = class_map.read(1, window=group_window)

        for idx, candidate_origin in enumerate(buffed_points):

            # Create a boolean mask representing the linestring from the windowed data for this candidate group.
            mask = rfeat.geometry_mask([candidate_origin], out_shape=group_data.shape, transform=rwin.transform(group_window, class_map.transform), all_touched=True)

            # Apply boolean mask to raster's data
            masked_data = ma.array(group_data, mask=mask)

            # Calculate mean weight for segment
            weight = masked_data.sum() / ma.compressed(masked_data).shape[0]
            
            # Initiallize top point as source.
            if ( idx == 0 ):
                top_weight = weight
                top_idx = idx
            # Update point if greater than minimum accepted weight and current best weight
            elif (weight > self.minimum_weight) and (weight > top_weight): 
                
                top_weight = weight
                top_idx = idx
        
        return top_idx # ignore notice, initialized in loop


    def _new_annotation(self, line, class_map=None):
        """ 
        Create a new annotation from a Linestring and predicted class map 
        using internal parameters.

        Args:
            line (shp.LineString): Source annotation to be improved in same CRS map raster.
            class_map (rio.DatasetReader): Predicted class map used to evaluate candidates.
            use_default (bool): Optionally continue annotations with zero-weight candidates by taking the same index point as the previous  
        Returns:
            list( (float, float) ): (x, y) coordinate offset tuples from the source annotation
            shp.Linestring: Improved annotation
        """

        if not isinstance(line, shp.LineString):
            raise TypeError("Class_Map_Annotator.new_annotation(): Expected type '{}' for line, recieved: '{}'".format(shp.LineString.__name__, type(line).__name__))

        if (class_map == None):
            raise RuntimeError("Class_Map_Annotator.new_annotation(): No class map to weight candidates.")


        # Generate candidate point groups
        candidate_groups = self._generate_points(line)

        for idx, group in enumerate(candidate_groups):
            
            if (idx == 0):
                # Select 'origin' or first point in annotation linestring
                point_idx = self._select_origin(group, class_map=class_map)
                
                # Intialize optimal points with selected orgin
                optimal_points = [group[point_idx]]
            else:
                # Select next candidate by index and append to optimal points
                point_idx = self._select_candidate(optimal_points[idx-1], group, class_map, default_point_idx=default_idx)
                optimal_points.append(group[point_idx])

            # If set, always set default point to origin
            if self.use_origin_default: 
                default_idx = 0
            else:
                # Otherwise, continue default point with previous index
                default_idx = point_idx

        if self.verbosity == 1:
            print("Annotation updated, index:", self.line_index)
            print()

        return shp.LineString(optimal_points)


    def update_gdf(self, gdf, class_map, weight_buffer=2, label_offsets=False, clip=True, out_path=None):
        """ 
        Update all annotations in a geodataframe from a predicted class map.

        Notes:
            GeoDataFrame and Raster must have the same CRS.
            Optionally Return Euclidean offsets for sequential model labels using get_dists parameter.
            If euclidean offsets are labeled, it will not be possible to save the GeoDataFrame.
                
        Args:
            gdf (gpd.GeoDataFrame): Frame holding Linestring annotations .
            class_map (rio.DatasetReader):  Predicted Class Map for 
                                            annotation weighting.
            weight_buffer (float/int): Buffer to apply to annotations when considering weights over predicted class map. DOES NOT AFFECT ORIGIN BUFFER.
            clip (bool [True]): Optionally CLIP input geometries to extent of raster. 
                         Clipping geometries cuts off exterior instead of removing outlying shapes entirely.
            out_path (filepath): Optional output for updated annotation geometries, if not .shp file, saves as folder. 
        Returns:
            gpd.GeoDataFrame: Updated annotations

                if label_offsets:
            gpd.GeoDataFrame: Updated annotations.
            list( list( float ) ): List of Euclidean offsets for each updated annotation's updated coordinates 
        """

        if not isinstance(gdf, gpd.GeoDataFrame):
            print("ERROR (Class_Map_Annotator.update_gdf): Invalid argument for 'gdf'. \nExpected: {} \nRecieved: {}".format(type(gpd.GeoDataFrame), type(gdf)))
        if not isinstance(class_map, rio.DatasetReader):
            print("ERROR (Class_Map_Annotator.update_gdf): Invalid argument for 'class_map'. \nExpected: {} \nRecieved: {}".format(type(rio.DatasetReader), type(class_map)))

        # Update internal parameters
        self.weight_buffer = weight_buffer

        # set window for cropping geoms
        self.set_crop_window(class_map)

        # Clip if passed.
        if clip:
            gdf = gpd.clip(gdf, mask=self.crop_window, keep_geom_type=True)

        # Generate improved annotations
        new_annotations = [self._new_annotation(line=line, class_map=class_map) for line in gdf.geometry]
        
        # Convert all updated geometries to gdf
        annotation_frame = gpd.GeoDataFrame(geometry=new_annotations, crs=gdf.crs)
        
        # If valid out_path passed, save to file 
        if out_path:
            # Warn when saving to folder instead of file.
            if '.shp' not in os.path.splitext(out_path)[1]:
                print("Warning (update_gdf): Saving updated annotations GeoDataFrame to folder instead of file: '{}'.\nTo save as file, include the .shp extension.".format(out_path))
            
            annotation_frame.to_file(out_path)

    
        return annotation_frame
        

    def get_candidates(self, gdf, class_map=None, flatten=True, out_path=None):
        """ 
        Generate only the candidate coordinate set for visualization, 
        optionally clip to raster.

        Args:
            gdf (gpd.GeoDataFrame): Source annotation to generate candidates from.
            class_map (rio.DatasetReader): Optional raster to clip candidate geometries.
            flatten (bool, default=True): Option to flatten output geometries
                True: Return candidates as a single GeoDataFrame of all candidate points. (Default)
                or
                False: Return candidates as a list of GeoDataFrames (one per geometry in input).
            out_path (filepath): Optional filepath to save candidates. If flatten==False, out_path is ignored.

        Returns:
            gpd.GeoDataFrame: Holds candidate groups as MultiPoints in source CRS. (flatten=True)
            or 
            list(gpd.GeoDataFrame): One GeoDataFrame of candidate points for each Linestring in source. (flatten=False)
        """  

        if class_map:
            # set window for cropping geoms and clip geometries
            self.set_crop_window(class_map)
            gdf = gpd.clip(gdf, mask=self.crop_window, keep_geom_type=True)
        
        # Get list of groups of points
        all_sets = [self._generate_points(line) for line in gdf.geometry]

        if flatten:
            # Flatten list, convert to geodataframe
            all_candidates = gpd.GeoDataFrame(geometry=[point_group for line_set in all_sets for point_group in line_set], crs=gdf.crs) 

            if out_path:
                if '.shp' not in os.path.splitext(out_path)[1]:
                    print("Warning (update_gdf): Saving candidate point GeoDataFrame to folder instead of file: '{}'.\nTo save as file, include the .shp extension.".format(out_path))

                all_candidates.to_file(out_path)

            return all_candidates 
        else:
            # Return list of GeoDataFrames, one for each geometry in source annotation
            return [gpd.GeoDataFrame(geometry=point_set, crs=gdf.crs) for point_set in all_sets]
