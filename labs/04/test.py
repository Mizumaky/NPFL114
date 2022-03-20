import os
import re
import typing
from typing import Dict

# Team: 7797f596-9326-11ec-986f-f39926f24a9c, 449dba85-9adb-11ec-986f-f39926f24a9c

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

from numba import cuda
device = cuda.get_current_device()
device.reset()