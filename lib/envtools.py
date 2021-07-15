"""
Misc Info
=========
Prints random stuff.
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