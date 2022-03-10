/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "cuda_runtime.h"

#include "mock_cuda_runtime.h"

namespace mitk
{
namespace cuda_example
{
namespace mock
{

MockCudaRuntime &MockCudaRuntime::GetInstance()
{
    static MockCudaRuntime Mock;
    return Mock;
}

}
}
}

using namespace mitk::cuda_example::mock;

unsigned int cudaMalloc(void** devPtr, size_t size)
{
    return MockCudaRuntime::GetInstance().cudaMalloc(devPtr, size);
}

unsigned int cudaFree(void *devPtr)
{
    return MockCudaRuntime::GetInstance().cudaFree(devPtr);
}

unsigned int cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return MockCudaRuntime::GetInstance().cudaMemcpy(dst, src, count, kind);
}
