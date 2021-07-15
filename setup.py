"""
Setup Script
============
Prepares environment for preprocessing, model training, and EM tests.

ToDo:
    pip and conda requirements
    Add source data download link
"""

import os

from numpy.testing._private.utils import print_assert_equal
if not os.path.exists('./config.py'):
    from shutil import copyfile
    copyfile('./config_template.py', './config.py')
    print("Created config.py in root.")
    #raise FileNotFoundError("Tests require configuration file in root project directory under the name 'config.py'")

# Print environment configuration
from config import *
import pathlib
required_dirs = [INPUT_DATA_DIR, TENSOR_DIR, RESULTS_DIR]
dir_infos = ["Raw Input Data", "Processed Tensor Output", "Results output"]
for di, env_dir in zip(dir_infos, required_dirs):
    if not os.path.exists(env_dir):
        pathlib.Path(env_dir).mkdir(parents=True)
        print(f"Created folder for '{di}' at '{env_dir}'")

print("Setup Complete.")
print("To run the Geometric Annotation Errors EM Pipeline, download the required data from [LINK TBD].\n")
print("See more info in README.md")