MITK CUDA Project Template
=========================

This project is based on the [MITK Project Template](https://github.com/MITK/MITK-ProjectTemplate).
It includes a complete CMake-based set-up to get started with [CUDA](https://developer.nvidia.com/cuda-toolkit) and [MITK](https://github.com/MITK/MITK).
The module shows how to use GTest for unit testing cuda modules, mock cuda runtime api calls and
wrap cuda device pointers into c++ smartpointers.

Features
--------

- CudaExample:
  - Cuda based MinMaxScaler image-to-image filter
  - How to wrap cuda device pointers into STL smartpointer
  - Example Unit tests
  - Example Mock tests
- MockCudaRuntime
  - Example how to mock cuda_runtime_api calls
- Example plugin
  - GUI for the cuda example image filter

How it works
------------
Prequisites are a working Qt installation and [CUDA](https://developer.nvidia.com/cuda-toolkit).

1. Clone [MITK](https://github.com/MITK/MITK) and checkout the latest release tag or at least the stable master branch
2. Click on "Use this template", or clone/fork the MITK-ProjectTemplate, checking out the matching tag or branch
3. Configure the MITK superbuild and set the CMake cache variable `MITK_EXTENSION_DIRS` to your working copy of the project template
4. Generate and build the MITK superbuild

The project template is virtually integrated right into the MITK superbuild and MITK build as if it would be part of MITK. You can extend MITK with your own modules, plugins, command-line apps, and external projects without touching the MITK source code resp. repository.

Supported platforms and other requirements
------------------------------------------

See the [MITK documentation](http://docs.mitk.org/2021.10/).

License
-------

Copyright (c) Eric Heim<br>
All rights reserved.

The MITK CUDA Project Template is based on [MITK](https://github.com/MITK/MITK) as well as the [MITK-ProjectTemplate](https://github.com/MITK/MITK-ProjectTemplate/blob/master/) and as such available as free open-source software under a [3-clause BSD license](https://github.com/ericheim/MITK-CudaProjectTemplate/blob/master/LICENSE).
