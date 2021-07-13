# Weakly Supervised Spatial Deep Learning
### KDD 2021 Submission

---

## Resources:
[Github](https://github.com/spatialdatasciencegroup/WeaklySupervisedSpatialDeepLearning)

---

## Project Description
TO ADD...

---

## User Interface Tool
To make this system more widely available, the `user-ready` branch will store a version of this code to allow installation and testing of the EM System.

### Acompanied Data
Data must be made available through some dropbox link. Scene01 will be used, and as much of the preprocessing as possible should be done after download.
- `merged_imagery.tif`: Contains all bands, normalized in scene01
- `imperfect_labels.shp`: Imperfect stream labels shapefile across entire merged raster
- `refined_labels.shp`: Hand-refined stream labels shapefile across entire merged raster

### User configuration
The tool will require a configured Python environment, as well as some system parameters to be set by the user upon download. Static configuration can be done in a config.py file, a template for which will be included in the repository. A `build.py` script can also be used to configure this with some sort of cli. The data will of course be retrieved manually.

### Step 01: Parse Raw Data
In a terminal-activated script, the user will parse the raw data (from dropbox) into model-ready tensors.

### Step 02: Test Baseline Model
In another notebook, the user can test the UNet model with their prepared tensors for one training iteration. 

### Step 03: Test EM Pipeline
The final step is a user-ready EM pipeline, where they can run the system step-by-step or with a fixed number of steps in a loop.

### Step 04: View Results
Output will be held in results folder, the path of which determined by `RESULTS_DIR` in `config.py`.

### Documentation
Need to create some sort of requirements file for conda and pip.