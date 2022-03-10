/*============================================================================
Copyright (c) Eric Heim
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef MOCK_DEVICE_TYPES_H
#define MOCK_DEVICE_TYPES_H

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost        =   0,  /**< Host   -> Host */
    cudaMemcpyHostToDevice      =   1,  /**< Host   -> Device */
    cudaMemcpyDeviceToHost      =   2,  /**< Device -> Host */
    cudaMemcpyDeviceToDevice    =   3,  /**< Device -> Device */
    cudaMemcpyDefault           =   4   /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

#endif