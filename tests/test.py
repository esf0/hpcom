# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
#
# print(tf.sysconfig.get_build_info())
#
import sys
print(sys.executable)
#
# import os
#
# import nvidia.cudnn
#
# CUDNN_PATH = os.path.dirname(os.path.realpath(nvidia.cudnn.__file__))
# # CUDNN_PATH = "/home/esf0/anaconda3/envs/tf/lib/python3.9/site-packages/nvidia/cudnn"
# print("path:", CUDNN_PATH)
# os.environ['LD_LIBRARY_PATH'] = f'{os.environ["CONDA_PREFIX"]}/lib/:{CUDNN_PATH}/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
# print(os.environ['LD_LIBRARY_PATH'])
# print(os.environ["CONDA_PREFIX"])
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
