
/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "CudaMinMaxScaler.h"

#include <cuda_runtime.h>

#include <mitkImage.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>

#include "DeviceUniquePtr.h"

namespace mitk
{
namespace cuda_example
{

CudaMinMaxScaler::CudaMinMaxScaler(ICudaMinMaxScalerStub& stub)
    : m_Launcher(stub)
{
    this->SetNumberOfRequiredInputs(1);
    this->SetNumberOfRequiredOutputs(1);
}

void CudaMinMaxScaler::GenerateData()
{
    mitk::Image::Pointer input_image = this->GetInput();
    mitk::Image::Pointer output_image = this->GetOutput();

    // Sanity check. Only 3D Images of type short are supported as input.
    if (input_image->GetPixelType().GetComponentType() != itk::ImageIOBase::SHORT ||
        input_image->GetDimension() != 3)
    {
        throw mitk::Exception("CudaMinMaxScaler only supports 3D images of type short");
    }

    // initialize output image
    output_image->Initialize(mitk::MakeScalarPixelType<float>(),
                             input_image->GetDimension(),
                             input_image->GetDimensions());

    output_image->SetClonedGeometry(input_image->GetGeometry());

    // Access raw buffers from image
    mitk::ImageReadAccessor ra(input_image, input_image->GetVolumeData());
    mitk::ImageWriteAccessor wa(output_image, output_image->GetVolumeData());
    const short* in_img = reinterpret_cast<const short*>(ra.GetData());
    float* out_img = reinterpret_cast<float*>(wa.GetData());

    // Get buffer size from image dimensions
    const auto dims = input_image->GetDimensions();
    const size_t img_size = dims[0] * dims[1] * dims[2];

    // Run min max scaler on GPU
    device_unique_ptr<short> device_in = make_device_unique<short>(img_size);
    device_unique_ptr<float> device_out = make_device_unique<float>(img_size);

    cudaMemcpy(device_in.get(), in_img, img_size * sizeof(short), cudaMemcpyHostToDevice);
    m_Launcher.LaunchKernel(device_in.get(), device_out.get(), img_size);
    cudaMemcpy(out_img, device_out.get(), img_size * sizeof(float), cudaMemcpyDeviceToHost);
}

}
}
