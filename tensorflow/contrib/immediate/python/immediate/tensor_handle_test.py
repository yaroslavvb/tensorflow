# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate
from tensorflow.contrib.immediate.python.immediate import test_util

class EnvTest(tf.test.TestCase):

  # TODO(yaroslavvb): re-enable once fix to below is merged
  # https://github.com/tensorflow/tensorflow/issues/2645
  def disabled_testHandleDeletion(self):
    dtype = tf.float32
    # soft-placement to work around #2587
    config = tf.ConfigProto(log_device_placement=True,
                            allow_soft_placement=True)
    sess = tf.Session(config=config)

    # initial values live on CPU
    with tf.device("/cpu:0"):
      one = tf.constant(1, dtype=dtype)
      one_handle = sess.run(tf.get_session_handle(one))
      x = tf.get_session_handle(one)
      x_handle = sess.run(tf.get_session_handle(one))

    # addition lives on GPU
    with tf.device("/gpu:0"):
      add_holder1, add_tensor1 = tf.get_session_tensor(dtype)
      add_holder2, add_tensor2 = tf.get_session_tensor(dtype)
      add_op = tf.add(add_tensor1, add_tensor2)
      add_output = tf.get_session_handle(add_op)


    # add 1 to tensor 20 times to exceed _DEAD_HANDLES_THRESHOLD
    for i in range(20):
      x_handle = sess.run(add_output, feed_dict={add_holder1: one_handle.handle,
                                                 add_holder2: x_handle.handle})

  # TODO(yaroslavvb): re-enable after fix for below is merged
  #  https://github.com/tensorflow/tensorflow/issues/2587
  def disabled_testPlaceholderOnGpuIssueAllGpu(self):
    # https://github.com/tensorflow/tensorflow/issues/2587
    config = tf.ConfigProto(log_device_placement=True)
    with self.test_session(config=config) as sess:
      dtype=tf.float32
      with tf.device("/gpu:0"):
        a = tf.constant(1, dtype)

        a_handle = sess.run(tf.get_session_handle(a))
        b_holder, b_tensor = tf.get_session_tensor(dtype)
        print(sess.run(b_tensor, feed_dict={b_holder:
                                              a_handle.handle}))

  # TODO(yaroslavvb): reenable when fix for below is merged
  # https://github.com/tensorflow/tensorflow/issues/2586
  def disabled_testHandle(self):
    def testHandleForType(tf_dtype):
      for use_gpu in [True, False]:
        with self.test_session(use_gpu=use_gpu) as sess:
          n = 3
          input_value = tf.ones((n, n), dtype=tf_dtype)
          handle1 = tf.get_session_handle(input_value)
          handle2 = tf.get_session_handle(input_value)
          holder1, tensor1 = tf.get_session_tensor(tf_dtype)
          holder2, tensor2 = tf.get_session_tensor(tf_dtype)
          tensor3 = tf.add(tensor1, tensor2)

          py_handle1, py_handle2 = sess.run([handle1, handle2])
          feed_dict = {holder1: py_handle1.handle, holder2: py_handle2.handle}
          tensor3_numpy = sess.run(tensor3, feed_dict=feed_dict)

          np_dtype = tf_dtype.as_numpy_dtype()
          self.assertAllEqual(tensor3_numpy, 2*np.ones((n, n), dtype=np_dtype))

    testHandleForType(tf.float16)
    testHandleForType(tf.int32)
    testHandleForType(tf.float32)
    testHandleForType(tf.int64)
    testHandleForType(tf.float64)


  def testPlaceholderOnGpuIssueAllCpu(self):
    config = tf.ConfigProto(log_device_placement=True)
    with self.test_session(config=config) as sess:
      dtype=tf.float32
      with tf.device("/cpu:0"):
        a = tf.constant(1, dtype)

        a_handle = sess.run(tf.get_session_handle(a))
        b_holder, b_tensor = tf.get_session_tensor(dtype)
        print(sess.run(b_tensor, feed_dict={b_holder:
                                              a_handle.handle}))

  

if __name__ == "__main__":
  tf.test.main()
