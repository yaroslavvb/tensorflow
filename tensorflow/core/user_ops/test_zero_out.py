import tensorflow as tf
import numpy as np
import os
home = os.environ['HOME']
zero_out_module = tf.load_op_library(home+'/tensorflow.git/tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so')
sess = tf.InteractiveSession()
print(zero_out_module.zero_out([[1, 2], [3, 4]]).eval())
