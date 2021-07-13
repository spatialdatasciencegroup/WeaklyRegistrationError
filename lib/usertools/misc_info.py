"""
Misc Info
=========
Prints random stuff.
"""

import os
from datetime import datetime as dt 
from dateutil.tz import tzlocal

def gettime(fmt: str = "%I:%M:%S %p"):
    """ Returns the current time as formatted string. 
    Args:
        fmt (str): strftime format string
    """
    return dt.now(tzlocal()).strftime(fmt)
    