### Misc. time functions for all the wacky time shenanigans
from datetime import datetime

DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DT_FORMAT_NO_MILLIS = "%Y-%m-%d %H:%M:%S"

def est2epoch(est_string):
    try: dt = datetime.strptime(est_string, DT_FORMAT)
    except ValueError: dt = datetime.strptime(est_string, DT_FORMAT_NO_MILLIS)
    epoch_time = (dt - datetime(1970,1,1)).total_seconds()
    return epoch_time + 18000   # 18000 is the est offset
    
def utc2epoch(utc_string):
    try: dt = datetime.strptime(utc_string, DT_FORMAT)
    except ValueError: dt = datetime.strptime(utc_string, DT_FORMAT_NO_MILLIS)
    epoch_time = (dt - datetime(1970,1,1)).total_seconds()
    return epoch_time

def epoch2utc(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime(DT_FORMAT)