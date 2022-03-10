/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cuda_runtime.h>
#include <mock_cuda_runtime.h>

#include "DeviceUniquePtr.h"

namespace mitk
{
namespace cuda_example
{
namespace testing
{

using namespace mock;
using ::testing::_;
using ::testing::Return;

TEST(MockTestDevicePtr, TestCudaMalloc) {
    size_t no_items = 10;
    size_t expected_bytes = no_items * sizeof(float);
    MockCudaRuntime& Mock = MockCudaRuntime::GetInstance();

    EXPECT_CALL(Mock, cudaMalloc(_, expected_bytes)).Times(1);
    EXPECT_CALL(Mock, cudaFree(_)).Times(1);

    device_unique_ptr<float> ptr = make_device_unique<float>(no_items);
}

TEST(MockTestDevicePtr, TestCudaMallocException) {
    MockCudaRuntime& Mock = MockCudaRuntime::GetInstance();

    EXPECT_CALL(Mock, cudaMalloc(_,_))
            .WillOnce(Return(1));

    EXPECT_THROW(
        device_unique_ptr<float> ptr =
            make_device_unique<float>(10);
        , Exception
    );
}

}
}
}
