/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "CudaMinMaxScalerStub.h"

namespace mitk
{
namespace cuda_example
{

__device__ unsigned int GetIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void MinMaxScalerKernel(short* input,
                                   float* output,
                                   float min,
                                   float max,
                                   unsigned int n)
{
    const unsigned int index = GetIndex();

    if (index < n)
    {
        const float x = (float)input[index];
        output[index] = (x - min) / (max - min);
    }
}

void CudaMinMaxScalerStub::LaunchKernel(short *device_input,
                                        float *device_output,
                                        unsigned int n)
{
    // number of blocks and blocksize. blocks * block_size = number of threads
    constexpr unsigned int block_size = 512;
    const size_t number_of_blocks = (n + block_size - 1) / block_size;

    // get min and max values in the gpu buffer with thrust
    thrust::device_ptr<short> thrust_ptr(device_input);
    thrust::device_ptr<short> min_ptr = thrust::min_element(thrust_ptr, thrust_ptr + n);
    thrust::device_ptr<short> max_ptr = thrust::max_element(thrust_ptr, thrust_ptr + n);

    MinMaxScalerKernel<<<number_of_blocks, block_size>>>(device_input,
                                                         device_output,
                                                         static_cast<float>(min_ptr[0]),
                                                         static_cast<float>(max_ptr[0]),
                                                         n);
    cudaDeviceSynchronize();
}

}
}
