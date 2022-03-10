/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "MinMaxScaler.h"

#include <mitkImageStatisticsHolder.h>

#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>

#include <itkImageIterator.h>

namespace mitk
{
namespace cuda_example
{

template<typename TPixel, unsigned int VImageDimension>
static void ScaleImage(const itk::Image<TPixel, VImageDimension>* inputImage, mitk::Image::Pointer outputImage)
{
    ImageWriteAccessor wa(outputImage);

    float* out_img = reinterpret_cast<float*>(wa.GetData());
    const TPixel* in_img = inputImage->GetBufferPointer();

    size_t img_size = 1;

    // compute number of pixels
    for(size_t i = 0; i < outputImage->GetDimension(); ++i)
        img_size *= outputImage->GetDimensions()[i];

    // get min and max value in image
    TPixel min_val = std::numeric_limits<TPixel>::max();
    TPixel max_val = std::numeric_limits<TPixel>::min();

    for(size_t i = 0; i < img_size; ++i)
    {
        if(in_img[i] < min_val) min_val = in_img[i];
        if(in_img[i] > max_val) max_val = in_img[i];
    }

    const float min = static_cast<float>(min_val);
    const float max = static_cast<float>(max_val);

    for(size_t i = 0; i < img_size; ++i)
    {
        out_img[i] = (static_cast<float>(in_img[i]) - min) / (max - min);
    }
}

MinMaxScaler::MinMaxScaler()
{
    this->SetNumberOfRequiredInputs(1);
    this->SetNumberOfRequiredOutputs(1);
}

void MinMaxScaler::GenerateData()
{
    mitk::Image::Pointer input_image = this->GetInput();
    mitk::Image::Pointer output_image = this->GetOutput();

    output_image->Initialize(mitk::MakeScalarPixelType<float>(),
                             input_image->GetDimension(),
                             input_image->GetDimensions());

    output_image->SetClonedGeometry(input_image->GetGeometry());

    try{
        AccessIntegralPixelTypeByItk_n(input_image, ScaleImage, (output_image));
    }
    catch (const mitk::AccessByItkException& e)
    {
        MITK_ERROR << "Unsupported pixel type or image dimension: " << e.what();
    }
}

}
}
