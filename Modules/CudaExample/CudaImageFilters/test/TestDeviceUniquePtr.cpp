/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <DeviceUniquePtr.h>

#include <stdio.h>
#include <assert.h>

namespace mitk
{
namespace cuda_example
{
namespace test
{

TEST(DeviceUniquePtr, TestAllocateMemory) {
    // Arrange
    cudaDeviceProp prop;
    cudaPointerAttributes attributes;
    // Act
    device_unique_ptr<float> ptr = make_device_unique<float>(20);
    // Assert
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, ptr.get()), cudaSuccess);
    ASSERT_EQ(attributes.type, cudaMemoryTypeDevice);
    ASSERT_NE(nullptr, attributes.devicePointer);
}

TEST(DeviceUniquePtr, TestFreeDeviceMemory) {
    // Arrange
    cudaDeviceProp prop;
    cudaPointerAttributes attributes;
    // Act
    device_unique_ptr<float> ptr = make_device_unique<float>(20);
    ptr.release();
    // Assert
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, ptr.get()), cudaSuccess);
    ASSERT_EQ(attributes.type, cudaMemoryTypeUnregistered);
    ASSERT_EQ(nullptr, attributes.devicePointer);
}

TEST(DeviceUniquePtr, TestForExeption) {
    EXPECT_THROW(
        device_unique_ptr<float> ptr =
            make_device_unique<float>(std::numeric_limits<long>::max());
        , Exception);
}

}
}
}
