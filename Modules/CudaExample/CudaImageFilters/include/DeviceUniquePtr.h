/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef DEVICE_UNIQUE_PTR_H
#define DEVICE_UNIQUE_PTR_H

#include <functional>
#include <memory>

#include <cuda_runtime.h>

#include <mitkException.h>

#define CUDA_CHECK_ERR(x) { \
    if(x) throw mitk::Exception("CUDA memory allocation error."); \
}\

namespace mitk
{
namespace cuda_example
{

template <typename T>
using device_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
static device_unique_ptr<T> make_device_unique(size_t n)
{
    auto allocator = [](size_t s) {
        void* ptr;
        CUDA_CHECK_ERR(cudaMalloc((void**) &ptr, s));
        return ptr;
    };

    auto deleter = [](T* ptr) {
        CUDA_CHECK_ERR(cudaFree((void*) ptr));
    };

    return device_unique_ptr<T>((T*) allocator(n * sizeof(T)), deleter);
}

}
}

#endif
