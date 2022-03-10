/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef I_CUDA_MIN_MAX_SCALER_STUB_H
#define I_CUDA_MIN_MAX_SCALER_STUB_H

#include <MitkCudaImageFiltersExports.h>

namespace mitk
{
namespace cuda_example
{

class MITKCUDAIMAGEFILTERS_EXPORT ICudaMinMaxScalerStub
{
public:
    virtual void LaunchKernel(short* device_input,
                              float* device_output,
                              unsigned int n) = 0;
};

}
}

#endif