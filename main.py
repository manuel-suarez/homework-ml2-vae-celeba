import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
from glob import glob
import numpy as np

gpu_available = tf.config.list_physical_devices('GPU')
print(gpu_available)