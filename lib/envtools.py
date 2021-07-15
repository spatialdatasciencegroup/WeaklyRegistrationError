"""
Environment Tools
=================
Handles printing, scheduling, 
and other features exclusive to public use.
"""

from datetime import datetime as dt 
from dateutil.tz import tzlocal

def gettime(fmt: str = "%I:%M:%S %p"):
    """ Returns the current time as formatted string. 
    Args:
        fmt (str): strftime format string
    """
    return dt.now(tzlocal()).strftime(fmt)

import tensorflow as tf
def get_tf_gpus():
    """ Get GPUS available to tensorflow. """
    gpus = []
    for device in tf.config.list_physical_devices('GPU'):
        gpu_name = device.name.lower().replace('physical_device:','')
        gpus.append(gpu_name)
    print(gpus)    
    return gpus

def lr_schedule(index: int):
    """ 
    A hard-coded learning rate schedule 
    based on the best-performing EM run. """
    if index in range(0,6):
        return 0.1
    elif index in range(6,12):
        return 0.05
    elif index in range(12,14):
        return 0.02
    else:
        raise RuntimeError(f"Invalid index passed: {index}. This schedule is only configured for EM tests with 14 steps.")
    