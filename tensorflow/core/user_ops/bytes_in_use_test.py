# export TF_CPP_MIN_VLOG_LEVEL=0
# Test memory allocation op
# See
# Bytes in use 0
# Memory usage is 0
#
# Meanwhile with export TF_CPP_MIN_VLOG_LEVEL=1
# see memory reported with allocator
# I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 2 kernel_name: "Variable" tensor { dtype: DT_INT32 shape { dim { size: 42000000 } } allocation_description { requested_bytes: 168000000 allocated_bytes: 168000000 allocator_name: "cpu" allocation_id: 21 ptr: 4691558400 } } }# I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 2 kernel_name: "Variable/read" tensor { dtype: DT_INT32 shape { dim { size: 42000000 } } allocation_description { requested_bytes: 168000000 allocated_bytes: 168000000 allocator_name: "cpu" allocation_id: 21 ptr: 4691558400 } } }


import tensorflow as tf
import numpy as np
import os
custom_module = tf.load_op_library("bytes_in_use.so")
var = tf.Variable(tf.random_uniform((42*10**6,), maxval=10, dtype=tf.int32))

bytes_used = custom_module.bytes_in_use(var[0])
result = tf.reduce_sum(var) + bytes_used
result0 = tf.reduce_sum(var+0) # sum of copy of var

config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
print("Memory usage is " + str(sess.run(result-result0)))
