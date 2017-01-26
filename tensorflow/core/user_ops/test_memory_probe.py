import tensorflow as tf
import numpy as np
import os
home = os.environ['HOME']

run_with_tracing = False
custom_module = tf.load_op_library("/Users/yaroslav/tensorflow.git/tensorflow/bazel-bin/tensorflow/core/user_ops/memory_probe_ops.so")
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

for d in ["/cpu:0", "/gpu:0"]:
    with tf.device(d):
        sess = tf.InteractiveSession(config=config)
        mbs = 42
        n = mbs*250000
        inputs = tf.random_uniform((n,))
        print("Allocating %d MB variable on %s"%(mbs, d,))
        var = tf.Variable(inputs)
        probe_op = custom_module.bytes_in_use()
        max_op = custom_module.bytes_limit()
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        print("Before init %d out of %d bytes" % tuple(sess.run([probe_op, max_op])))
        sess.run(var.initializer)
        print("After init %d out of %d bytes" % tuple(sess.run([probe_op, max_op])))
        if run_with_tracing:
            print(run_metadata)
