# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for Pack Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BytesInUseOpTest(test.TestCase):

  def testSimple(self):
    with self.test_session(use_gpu=True) as sess:
      with ops.device('/gpu:0'):
        n = 42*10**6,
        dummy = array_ops.ones((), dtype=dtypes.int32)
        bytes_in_use = data_flow_ops.bytes_in_use(dummy)
        bytes1 = sess.run(bytes_in_use)
        var = tf.Variable(tf.random_uniform((n,), maxval=10, dtype=tf.int32))
        sess.run(var.initializer())
        bytes2 = sess.run(bytes_in_use)
        self.assertEqual(bytes2 - bytes1, n*4)

if __name__ == "__main__":
  test.main()
