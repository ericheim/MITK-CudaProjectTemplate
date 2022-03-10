/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef MOCK_CUDA_RUNTIME_H
#define MOCK_CUDA_RUNTIME_H

#include <gmock/gmock.h>

#include "mock_device_types.h"

namespace mitk
{
namespace cuda_example
{
namespace mock
{

class MockCudaRuntime final
{
public:
    static MockCudaRuntime& GetInstance();

    MOCK_METHOD(unsigned int, cudaMalloc, (void**, size_t));
    MOCK_METHOD(unsigned int, cudaFree, (void*));
    MOCK_METHOD(unsigned int, cudaMemcpy, (void*, const void*, size_t, enum cudaMemcpyKind));

    void operator=(const MockCudaRuntime&) = delete;
    MockCudaRuntime(const MockCudaRuntime&) = delete;

private:
    MockCudaRuntime() = default;
};

}
}
}

#endif
