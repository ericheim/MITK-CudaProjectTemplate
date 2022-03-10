/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef CUDA_MIN_MAX_SCALER_STUB_H
#define CUDA_MIN_MAX_SCALER_STUB_H

#include <MitkCudaImageFiltersExports.h>

#include "ICudaMinMaxScalerStub.h"

namespace mitk
{
namespace cuda_example
{

class MITKCUDAIMAGEFILTERS_EXPORT CudaMinMaxScalerStub final : public ICudaMinMaxScalerStub
{
public:
    void LaunchKernel(short* device_input,
                      float* device_output,
                      unsigned int n) override;
};

}
}

#endif
