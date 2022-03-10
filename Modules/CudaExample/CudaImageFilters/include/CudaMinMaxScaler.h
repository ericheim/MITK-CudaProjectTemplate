/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef CUDA_MIN_MAX_SCALER_H
#define CUDA_MIN_MAX_SCALER_H

#include <memory>

#include <mitkImageToImageFilter.h>
#include <MitkCudaImageFiltersExports.h>

#include "ICudaMinMaxScalerStub.h"

namespace mitk
{
namespace cuda_example
{

class MITKCUDAIMAGEFILTERS_EXPORT CudaMinMaxScaler final : public mitk::ImageToImageFilter
{
public:
  // All classes that derive from an ITK-based MITK class need at least the
  // following two macros. Make sure you don't declare the constructor public
  // to force clients of your class to follow the ITK convention for
  // instantiating classes via the static New() method.
  mitkClassMacro(CudaMinMaxScaler, mitk::ImageToImageFilter)
  mitkNewMacro1Param(Self, ICudaMinMaxScalerStub&);

private:
  explicit CudaMinMaxScaler(ICudaMinMaxScalerStub& stub);
  ~CudaMinMaxScaler() = default;

  void GenerateData() override;

  ICudaMinMaxScalerStub& m_Launcher;
};

} // namespace gpu
} // namespace mitk

#endif
