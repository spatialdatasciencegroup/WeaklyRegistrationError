"""
Environment Configuration
=========================

Holds static parameters for this system to ensure the EM Pipeline runs smoothly.
"""

# Folder containing raw data from dropbox
INPUT_DATA_DIR = './data'

# Holds preprocessed tensors and csv file to re-read their offsets.
TENSOR_DIR = './tensors'

# Holds output results from preprocessing, em tests, and the UNet Baseline training
RESULTS_DIR = './results'