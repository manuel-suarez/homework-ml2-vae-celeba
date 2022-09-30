import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
from glob import glob
import numpy as np

gpu_available = tf.config.list_physical_devices('GPU')
print(gpu_available)

section    = 'vae'
run_id     = '0001'
data_name  = 'faces'
RUN_FOLDER = '../run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #

# run params
# DATA_FOLDER   = '/home/mariano/Data/celebA/imgs_align/img_align_celeba_png'
DATA_FOLDER   = '/home/est_posgrado_manuel.suarez/data/img_align_celeba'
INPUT_DIM     = (128,128,3)
LATENT_DIM    = 150
BATCH_SIZE    = 384
R_LOSS_FACTOR = 100000  # 10000
EPOCHS        = 400
INITIAL_EPOCH = 0

filenames  = np.array(glob(os.path.join(DATA_FOLDER, '*.png')))
n_images        = filenames.shape[0]
steps_per_epoch = n_images//BATCH_SIZE

print('num image files : ', n_images)
print('steps per epoch : ', steps_per_epoch )