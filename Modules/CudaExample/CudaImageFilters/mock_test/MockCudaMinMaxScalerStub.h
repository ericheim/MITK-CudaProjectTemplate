#ifndef MOCK_CUDA_MIN_MAX_SCALER_KERNEL_STUB_H
#define MOCK_CUDA_MIN_MAX_SCALER_KERNEL_STUB_H

#include <gmock/gmock.h>
/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#include "CudaMinMaxScalerStub.h"

namespace mitk
{
namespace gpu
{
namespace mock
{

class MockCudaMinMaxScalerStub final : public CudaMinMaxScalerStub
{
public:
    MOCK_METHOD(void, LaunchKernel, (short*, float*, unsigned int));
};

}
}
}

#endif