# Weakly Supervised Spatial Deep Learning

Originally written for KDD 2021 Submission.

This repository has been configured for installation and usage to verify the efficacy of the Geometric Annotation Errors system. 

## About
To add...

---

### Requirements
- **Python** >=3.7
- **tensorflow** >=2.2.0,<2.3.0
- **tensorboard** >=2.2.2
- **rasterio** >=1.1.5
- **geopandas** ==0.8.1
- **scikit-learn** ==0.23.2
- **matplotlib** >= 3.2.2
See `requirements.txt` for a full list.

## Using the Pipeline

#### 1. Geting the Source Data 
A few small image and shape files are required to run the code locally, which have been made available through ?dropbox? here: [LINK TDB]. The archive contains a small sample of labeled satellite imagery to train the UNet model during the EM algorithm.

Simply download the .zip and extract the contents to the root folder for this repository. After running the setup script, make sure `INPUT_DATA_DIR` in `config.py` is set to the extracted directory.   

Required files:
- `train_raster.tif`: Training imagery
- `train_raster.tif`: Testing imagery
- `imperfect_labels.shp`: Imperfect streamline labels 
- `refined_labels.shp`: Hand-refined ground truth streamline labels
#### 2. Set up environment
After cloning or downloading this repository, ensure you've properly set up a Python environment to execute the pipeline in. A list of the required packages is provided above. 

Next, run the `setup.py` file in the root directory. This will create `config.py` and some filepaths for hosting input data and storing results. 

Ensure the constant `INPUT_DATA_DIR` in the newly generated `config.py` file is set to the folder containing the extracted data from ?dropbox?.

#### 3.  Parsing the Training Data
To convert the raw imagery and labels into training-ready data, run `preprocess.py`. The tensors produced by this script will be saved in `TENSOR_DIR` specified in `config.py`. 

### 4. Run the Pipeline
Once the tensors have been generated, open `GeoErrors-EM.ipynb` with an IPython notebook editor to run the pipeline. Outputs for the test will be printed in the notebook and saved in an indexed sub-folder under the `RESULTS_DIR` path specified in `config.py`.

---

## Resources:
- [Github](https://github.com/spatialdatasciencegroup/WeaklySupervisedSpatialDeepLearning)
- [Paper Link]() - To add
- [Jiang Lab](https://www.jiangteam.org/)