#!/bin/sh
echo "Running env_test.py"
python env_test.py
echo "Running env_cache_test.py"
python env_cache_test.py
echo "Running extra_tests/batchnorm_test.py"
python extra_tests/batchnorm_test.py
echo "Running extra_tests/histogram_ops_test.py"
python extra_tests/histogram_ops_test.py
echo "Running extra_tests/math_ops_test.py"
python extra_tests/math_ops_test.py
echo "Running extra_tests/nn_test.py"
python extra_tests/nn_test.py
echo "Running itensor_test.py"
python itensor_test.py
echo "Running module_rewriter_test.py"
python module_rewriter_test.py
echo "Running mnist_inference_test.py"
python mnist_inference_test.py
echo "Running lbfgs_test.py"
python lbfgs_test.py
# echo "Running image_ops_double_test"
# python extra_tests/image_ops_double_test.py
# echo "Running image_ops_test"
# python extra_tests/image_ops_test.py
