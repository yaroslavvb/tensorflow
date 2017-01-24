load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)


tf_custom_op_library(
    name = "bytes_in_use.so",
    srcs = ["bytes_in_use.cc"],
    
    # add dep to get a hold of device.h
    # get this
    # ERROR: Analysis of target '//tensorflow/core/user_ops:bytes_in_use.so' failed; build aborted.
    deps = ["//tensorflow/core:core_cpu"],
)
