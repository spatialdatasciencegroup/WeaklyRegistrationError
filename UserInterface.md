
# User Interface Task List (Internal use only)
To make this system more widely available, the `user-ready` branch will store a version of this code to allow installation and testing of the EM 
System. For the sake of simplification, all test parameters are hard-coded based on Wenchong's best results. 


## Requirements

### Acompanied Data
Data must be made available through some dropbox link. Scene01 will be used, and as much of the preprocessing as possible should be done after download.
- `train_raster.tif`: scene01 normalized training area raster
- `test_raster.tif`: scene01 normalized testing area raster
- `imperfect_labels.shp`: Imperfect stream labels shapefile across entire merged raster
- `refined_labels.shp`: Hand-refined stream labels shapefile across entire merged raster
Although the notebooks do not require input shapefiles to be in the rasters' CRS, it's preferred.

## Additions

### config.py
Some parameters for the GeoErrors pipeline depend on the host machine, fortunately, all hard-coded system parameters (filepaths, output directories) have been converted to OS-aware paths. The remaining static directories (data source folder, tensor output, test logging) are defined in `config.py` under the project's root. The `user-ready` branch contains a template for the config file, from which the `config.py` is produced during the setup script.

### setup.py
A `setup.py` script configures the environment by creating `config.py` and any missing directories it defines.

---

## Changes

### Preprocessing Script
Instead of the preprocessing notebook we use to refine the input datasets, tensors are parsed with `preprocess.py`. The contained code is fundementally the same as the preprocessing notebook with some added stability and error printing. If the required source data is missing (train/test rasters or labels) an error is raised that tells the user where to download the missing files. Created tensors are saved to `TENSOR_DIR` defined in `config.py`. Sequential runs of the script overwrites previous sets. 

### EM Pipeline
- Has almost no tunable parameters now, and uniform printing for results.
- The EM Notebook does **not** have GPU training built in for simplicity. May want to add a note about this is the end description.
- No longer has the preload parameter to load from Juypter Storage all candidates must be generated at runtime. 
- Two schedulers have been added to reflect Wenchong's best results. 
  - `buff_dist` (buffer applied when rasterizing new annotations)
    - Step 00: **4**
    - Steps 01-13: **2**
  - `learning_rate`
    - Steps 00-05: **0.1**
    - Steps 06-11: **0.05**
    - Steps 12-13: **0.02**

---

## Removed

### UNet Notebook
In our development environment, we use a separate notebook to tune the baseline UNet model. This has been excluded since the UNet itself is not novel. The model compilation and training parameters are hard-coded, so tuning isn't necessary. 

---

## To Do

### Results Output 
It's unclear how much metadata we want to save from the system. Create a duplicate notebook with slimmed output for now so we can always return.

#### Original EM Notebook output:
- Baseline UNet training history plot (.png)
- Baseline UNet tensorboard (zip?)
- Baseline UNet weights (.h5)
- For each step:
  - Predicted class map (.tif)
  - Updated Annotation (.shp)
  - Rasterized Annotation (.tif)
  - Model Weights (.h5)
  - Model Tensorboard (zip?)
  - Model training history plot (.png)
  - Model performance markdown
- model performance results plot (.png)
- general info markdown

#### Slimmed EM Notebook output:
- Baseline UNet training history plot (.png)
- For each step:
  - Predicted Class map
  - Updated Annotation (.shp)
  - Model training history plot (.png)
- model performance results plot (.png)
- general info markdown

### Python Package Requirements
For now, specifing package names by hand should be fine since there is no detectable deprecated libraries. It's possible to add a file that `pip` can use to install required modules, but I am not sure how this works. 

---

## Notes

### Documentation