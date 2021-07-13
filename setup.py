"""
Setup Script
============
Prepares environment for preprocessing, model training, and EM tests.

ToDo:
    Initialize folders.
    Prompt path replacements for config.py
    Figure out how to force dependencies

"""
print("Starting setup script.")

import os
if not os.path.exists('./config.py'):
    from shutil import copyfile
    copyfile('./config_template.py', './config.py')
    #raise FileNotFoundError("Tests require configuration file in root project directory under the name 'config.py'")

# Print environment configuration
from config import *
required_dirs = [INPUT_DATA_DIR, TENSOR_DIR, RESULTS_DIR]
for env_dir in required_dirs:
    if not os.path.exists(env_dir):
        os.mkdir(env_dir)