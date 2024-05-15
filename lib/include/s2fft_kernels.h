#ifndef _S2FFT_KERNELS_H
#define _S2FFT_KERNELS_H

#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include "cufft.h"
#include <cufftXt.h>
typedef long long int int64;

namespace s2fftKernels {

HRESULT launch_spectral_folding(int* data, int* output, const int& nside, const int& L,
                                const int64& equatorial_offset_start, const int64& equatorial_offset_end,
                                cudaStream_t stream);

HRESULT launch_spectral_extension(int* data, int* output, const int& nside, const int& L,
                                  const int64& equatorial_offset_start, const int64& equatorial_offset_end,
                                  cudaStream_t stream);
}  // namespace s2fftKernels

#endif  // _S2FFT_KERNELS_H