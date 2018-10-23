import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Just disables the warning, doesn't enable AVX/FMA 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    #return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()