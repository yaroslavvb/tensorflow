# Description:
#   Contains implementation of imperative mode in TensorFlow
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "imperative_py",
    srcs = [
         "__init__.py",
         "python/imperative/env.py",
         "python/imperative/itensor.py",
         "python/imperative/module_rewriter.py",
         "python/imperative/op.py",
         "python/imperative/test_util.py",
         "python/imperative/util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
         "//tensorflow/python:client",
         "//tensorflow/python:framework",
         "//tensorflow/python:lib",
         "//tensorflow/python:ops",
         "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_library(
    name = "imperative_test_lib",
    srcs = [],
    srcs_version = "PY2AND3",
    deps = [
	"//tensorflow/python:framework_test_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow:tensorflow_py",
        ":imperative_py",
    ],
)

py_test(
    name = "env_test",
    size = "small",
    srcs = ["python/imperative/env_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_py",
	":imperative_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "env_cache_test",
    size = "small",
    srcs = ["python/imperative/env_cache_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_py",
	":imperative_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "module_rewriter_test",
    size = "small",
    srcs = ["python/imperative/module_rewriter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_py",
        ":imperative_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "itensor_test",
    size = "small",
    srcs = ["python/imperative/itensor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_py",
        ":imperative_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensor_handle_test",
    size = "small",
    srcs = ["python/imperative/tensor_handle_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":imperative_py",
	":imperative_test_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "math_ops_test",
    srcs = ["python/imperative/extra_tests/math_ops_test.py"],
    size = "small",
    srcs_version = "PY2AND3",
    deps = [
           ":imperative_py",
           ":imperative_test_lib",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:framework_test_lib",
#           "//tensorflow/python:kernel_tests/gradient_checker",
           "//tensorflow/python:platform_test",
           "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "batchnorm_test",
    srcs = ["python/imperative/extra_tests/batchnorm_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":imperative_py",
           ":imperative_test_lib",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:framework_test_lib",
#           "//tensorflow/python:kernel_tests/gradient_checker",
           "//tensorflow/python:platform_test",
           "//tensorflow:tensorflow_py",
    ],
)

# TODO(yaroslavvb): uncomment once all "with_devices" tests are removed
# py_test(
#     name = "image_ops_test",
#     srcs = ["python/imperative/extra_tests/image_ops_test.py"],
#     size = "medium",
#     srcs_version = "PY2AND3",
#     deps = [
#            ":imperative_py",
#            ":imperative_test_lib",
#            "//tensorflow:tensorflow_py",
#     ],

#     data = [
#         "//tensorflow/core:image_testdata",
#     ],
# )

# NOTE(yaroslavvb): uncomment once image_ops stabilizes
# py_test(
#     name = "image_ops_double_test",
#     srcs = ["python/imperative/extra_tests/image_ops_double_test.py"],
#     size = "medium",
#     srcs_version = "PY2AND3",
#     deps = [
#            ":imperative_py",
#            ":imperative_test_lib",
#            "//tensorflow:tensorflow_py",
#     ],

#     data = [
#         "//tensorflow/core:image_testdata",
#     ],
# )

py_test(
    name = "histogram_ops_test",
    srcs = ["python/imperative/extra_tests/histogram_ops_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":imperative_py",
           ":imperative_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
#           "//tensorflow/python:kernel_tests/gradient_checker",
    ],
    tags = ["manual"]  # fails on Jenkins but passes locally
)


py_test(
    name = "nn_test",
    srcs = ["python/imperative/extra_tests/nn_test.py"],
    size = "small",
    srcs_version = "PY2AND3",
    deps = [
           ":imperative_py",
           ":imperative_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
#           "//tensorflow/python:kernel_tests/gradient_checker",
    ],
)

py_test(
    name = "mnist_inference_test",
    srcs = ["python/imperative/mnist_inference_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":imperative_py",
           ":imperative_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/models/image/mnist:convolutional",
    ],

    data = [
    	 ":mnist_testdata"
    ],
)

py_test(
    name = "lbfgs_test",
    srcs = ["python/imperative/lbfgs_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":imperative_py",
           ":imperative_test_lib",
           "//tensorflow:tensorflow_py",
    ],

    data = [
    	 ":mnist_testdata"
    ],
)

filegroup(
    name = "mnist_testdata",
    srcs = [
    	"python/imperative/testdata/checkpoint",
	"python/imperative/testdata/convolutional-0",
	"python/imperative/testdata/convolutional-0.meta",
	"python/imperative/testdata/t10k-images-idx3-ubyte.gz",
        "python/imperative/testdata/t10k-labels-idx1-ubyte.gz",
	"python/imperative/testdata/mnist_data_32x32_small.npy",
        "python/imperative/testdata/mnist_labels_small.npy",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
