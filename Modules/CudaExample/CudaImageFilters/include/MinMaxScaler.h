/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef MIN_MAX_SCALER_H
#define MIN_MAX_SCALER_H

#include <MitkCudaImageFiltersExports.h>
#include <mitkImageToImageFilter.h>

namespace mitk
{
namespace cuda_example
{

class MITKCUDAIMAGEFILTERS_EXPORT MinMaxScaler final : public mitk::ImageToImageFilter
{
public:
  mitkClassMacro(MinMaxScaler, mitk::ImageToImageFilter)
  itkFactorylessNewMacro(Self)

private:
  MinMaxScaler();
  ~MinMaxScaler() = default;

  void GenerateData() override;
};

} // namespace gpu
} // namespace mitk
#endif
