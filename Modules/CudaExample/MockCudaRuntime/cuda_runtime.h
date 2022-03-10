/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H

#include <cstdio>

#include "mock_device_types.h"

unsigned int cudaMalloc(void** devPtr, size_t size);

unsigned int cudaFree(void* devPtr);

unsigned int cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

#endif
