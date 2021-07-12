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
New candidate selection system to work with single datasets at a time.

Annotating in complete lines instead of greedy coordinate point selection.

"""

cma_kwargs_key = {
    'annotator': ("(Annotator [None])", "Annotator to configure parent parameters by. If no annotator is passed, will use annotator default."),
    'pairs': ("(int [10])", "Number of coordinate pairs to be generated for each candidate grouping."),
    'off_dist': ("(int, meters [1])", "Base distance  to offset candidate coordinate pairs.")
}

cma_kwargs_key.update(("Annotator - {}".format(key), item) for key, item in anno_kwargs_key.items())

class Dynamic_Annotator(Annotator):
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
            coordinate_precision = 1e-4,# Float precision for coordinate comparison 
            weight_buffer = 3,            # Buffer to apply to candidate linestrings when weighting 
            L = 0.001,                  # weight for line length consideration when weighting canididates 
            min_p = 1e-9,      # Minimum probability to consider a line valid on the predicted class map  
            normalize_full = False,     # Optionally normalize over k^2 segments instead of k segments
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

        for i in range(self.pairs):
            
            # increment
            dist += self.off_dist

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

        # Convert list of points to shp.MultiPoint and return
        return shp.MultiPoint(candidates)
        

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

    def get_length_range(self, group_a, group_b):
        """ Get the min and max candidate line lengths slowly accross k^2 candidates.
        
        Args:
            group_a: shp.MultiPoint of target
            group_a: shp.MultiPoint of previous
        Returns:
            min_length (float), max_length (float)
        """    

        l_max, l_min = 0,0 
        for p_idx, pt in enumerate(group_a):
            lengths = [pt.distance(other_pt) for other_pt in group_b] 
            if p_idx > 0:
                lengths += [l_max, l_min]
            l_max = np.amax(lengths)
            l_min = np.amin(lengths)

        return l_min, l_max     
    
    def origin_state(self, origin_group, class_map):
        """ Assigns weights to origin group from internal predicted class map
        
        Args:
            origin_group (shp.MultiPoint): Group of points to select origin point from.
            class_map (rio.DatasetReader): Predicted class map as raster to read weights from.

        Returns:
            int: optimal starting point's index based on predicted class map weights.
        """

        # Initialize weight list
        weights = []

        origin_points = list(origin_group)

        # Buffer points into polygons for evaluation
        buffed_points = [point.buffer(self.weight_buffer) for point in origin_group]

        
        for idx, point in enumerate(buffed_points):
            if not self.crop_window.contains(point):
                del origin_points[idx]
                del buffed_points[idx]

        # Window to read candidate group from raster
        group_window = rfeat.geometry_window(class_map, buffed_points)

        # read pixel data from class map using geometry window
        group_data = self.class_map.read(1, window=group_window)

        for idx, candidate_origin in enumerate(buffed_points):

            # Create a boolean mask representing the linestring from the windowed data for this candidate group.
            mask = rfeat.geometry_mask([candidate_origin], out_shape=group_data.shape, transform=rwin.transform(group_window, class_map.transform), all_touched=True)

            # Apply boolean mask to raster's data
            masked_data = ma.array(group_data, mask=mask)

            # Calculate mean weight for segment
            weights.append(masked_data.sum() / ma.compressed(masked_data).shape[0])
        
        origin_state = {
            'xy': [pt.coords[0] for pt in origin_points],
            'parent_idx': None,
            'weight': weights
        }
        return origin_state


    def get_state_row(self, class_map, target_group, prev_group, weights: list):
        """ Gets a row of state with this set's parent nodes and  their values. """

        candidate_line_sets = []

        # Stores state for candidate group
        state = {
            'xy': [],
            'parent_idx': [],
            'weight': [],
            # NOTE: Future terms.
            # - Shape of line
            # - Direction of shift
            # - Neighboring line segment shapes
            # - - **Similarity in shape between neighboring lines should be highly weighted.
        }

        # Concatenate and buffer all points to create a window containing all needed class map data
        all_points = list(target_group) + list(prev_group)
        
        win_start = time.perf_counter()
        group_window = rfeat.geometry_window(class_map, [pt.buffer(self.weight_buffer) for pt in all_points])
        self.timer['window'] += (time.perf_counter() - win_start)
        
        # read pixel data from class map using geometry window
        read_start = time.perf_counter()
        group_data = class_map.read(1, window=group_window)
        self.timer['read'] += (time.perf_counter() - read_start)
        
        # Get terms for normalization
        ### This is slow, but I will fix it in the pre-loading. Just want to test asap
        len_r_start = time.perf_counter()
        if self.normalize_full:
            l_min, l_max = self.get_length_range(target_group, prev_group)
        
        self.timer['len_range'] += (time.perf_counter() - len_r_start)
        
        # Iterate over target group's points
        for anchor_idx, anchor in enumerate(target_group):

            # Reset top index and weight
            top_weight, top_idx = 0,0

            # Create candidate lines connecting this anchor point to every previous point
            buff_start = time.perf_counter()
            candidate_lines = [shp.LineString([anchor, point]).buffer(self.weight_buffer) for point in prev_group.geoms]
            self.timer['buffer'] += (time.perf_counter() - buff_start)

            if not self.normalize_full:
                l_min, l_max = self.get_length_range([anchor], prev_group)
            
            for idx, (candidate_line, weight_value) in enumerate(zip(candidate_lines, weights)):
                  
                # Create a boolean mask representing the linestring from the windowed data for this candidate group.
                mask = rfeat.geometry_mask([candidate_line], out_shape=group_data.shape, transform=rwin.transform(group_window, class_map.transform), all_touched=True)
                
                
                np_start = time.perf_counter()
                # Apply boolean mask to raster's data
                masked_data = ma.array(group_data, mask=mask)
                
                # Calculate mean probability weight for segment
                p_weight = masked_data.sum() / ma.compressed(masked_data).shape[0]

                # Check weight by threshold
                if p_weight <= self.min_p:
                    p_weight = 0.0
                
                # Calculate line length
                line_length = ((candidate_line.length - l_min) / (l_max - l_min)) 
                length_value = (self.L * (1 - line_length)) #- self.L *candidate_line.length #(self.L * (1 - line_length))
                p_weight += length_value 

                if (idx == 0) or ((weight_value + p_weight) > top_weight):
                    top_weight = (weight_value + p_weight)
                    top_idx = idx 
                
                self.timer['numpy'] += (time.perf_counter() - np_start)
        
            # Update state matrix
            state['xy'].append(anchor.coords[0])
            state['parent_idx'].append(top_idx)
            state['weight'].append(top_weight)

        return state


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
        #! TIMING
        self.timer = {
            'mask': 0.0,
            'read': 0.0,
            'len_range': 0.0,
            'window': 0.0,
            'buffer': 0.0,
            'candidates': 0.0,
            'numpy': 0.0,
            'origin': 0.0
        }
        
        if not isinstance(line, shp.LineString):
            raise TypeError("Class_Map_Annotator.new_annotation(): Expected type '{}' for line, recieved: '{}'".format(shp.LineString.__name__, type(line).__name__))

        if (class_map == None) and (self.class_map == None):
            raise RuntimeError("Class_Map_Annotator.new_annotation(): No class map to weight candidates.")

        self.class_map = class_map

        # Generate candidate point groups
        candidate_start = time.perf_counter()
        candidate_groups = self._generate_points(line)
        self.timer['candidates'] = time.perf_counter() - candidate_start
        
        
        origin_start = time.perf_counter()
        state_matrix = [self.origin_state(candidate_groups[0], class_map=class_map)]
        self.timer['origin'] = time.perf_counter() - origin_start
        
        for i in range(1, len(candidate_groups)):
            state_matrix.append(self.get_state_row(class_map=class_map, target_group=candidate_groups[i], prev_group=candidate_groups[i-1], weights=state_matrix[i-1]['weight']))

        return state_matrix

       

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



    def back_prop(self, state_matrix) -> list:
        """ Converts state matrix to updated linestring. """
        
        # Holds updated shapely points
        new_points = []
        
        top_weight = 0
        top_idx = 0
        for idx, weight in enumerate(state_matrix[-1]['weight']):
            if weight > top_weight:
                top_weight = weight
                top_idx = idx

        i = 1
        while i <= len(state_matrix): 
            new_points.insert(0, shp.Point(state_matrix[-i]['xy'][top_idx]))
            
            # Final points
            if state_matrix[-i]['parent_idx'] == None: break

            top_idx = state_matrix[-i]['parent_idx'][top_idx]
            i += 1
        return shp.LineString(new_points)