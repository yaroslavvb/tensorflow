# Tests for immediate.Tensor

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class TensorTest(tf.test.TestCase):

  def testInit(self):
    tensor = immediate.Tensor(None, None)
    self.assertTrue(True)

  def testNumpyInit(self):
    env = immediate.Env()
    a = np.array([[1,2],[3,4]], dtype=np.float32)
    tensor1 = env.numpy_to_tensor(a)
    tensor2 = env.numpy_to_tensor(a)
    print tensor1
    print tensor2
    
    array1 = tensor1.as_numpy()
    array2 = tensor2.as_numpy()
    self.assertAllEqual(array1, array2)

  def testBool(self):
    env = immediate.Env()
    self.assertFalse(env.numpy_to_tensor(False))
    self.assertTrue(env.numpy_to_tensor(True))

if __name__ == "__main__":
  tf.test.main()
