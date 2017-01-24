// build with
//TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
// g++ -std=c++11 -undefined dynamic_lookup -shared memory_op.cc -o memory_op.so -fPIC -I $TF_INC -O2

#include <limits.h>
#include <atomic>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

REGISTER_OP("BytesInUse")
    .Input("dummy_input: int32")   // TODO: need input for alloc_attrs
    .Output("bytes_in_use: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using namespace tensorflow;

class BytesInUseOp : public OpKernel {
 public:
  explicit BytesInUseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {

    // Create an scalar output
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
                                             &output_tensor));
    auto output = output_tensor->flat<int32>();
    
    // Grab memory
    AllocatorAttributes alloc_attrs = ctx->input_alloc_attr(0);
    DeviceContext* device_ctxt = ctx->op_device_context();
    auto device = static_cast<tensorflow::Device*>(ctx->device());
    Allocator* allocator = device->GetAllocator(alloc_attrs);
    AllocatorStats stats;
    allocator->GetStats(&stats);
    output(0) = stats.bytes_in_use;
    printf("Bytes in use %lld\n", stats.bytes_in_use);
    output(0) = stats.bytes_in_use;
  }
};

REGISTER_KERNEL_BUILDER(Name("BytesInUse").Device(DEVICE_CPU), BytesInUseOp);
