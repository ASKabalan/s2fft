#include "s2fft_kernels.h"
#include "hresult.h"
#include <cmath>  // has to be included before cuda/std/complex
#include <cstddef>
#include <cuda/std/complex>
#include <iostream>

namespace s2fftKernels {

template <typename real_t>
__global__ void spectral_folding(void* data, void* output, int nside, int L) {
    using complex = cuda::std::complex<real_t>;
    complex* data_c = reinterpret_cast<complex*>(data);
    complex* output_c = reinterpret_cast<complex*>(output);
    // few inits
    int polar_rings = nside - 1;
    int equator_rings = 3 * nside + 1;
    int total_rings = 4 * nside - 1;
    int ftm_size = 2 * L;
    // Compute number of pixels
    int total_pixels = 12 * nside * nside;
    int upper_pixels = nside * (nside - 1) * 2;
    int equator_pixels = 4 * nside * equator_rings;
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos(0);
    int indx(0);
    // Compute nphi of current ring
    int nphi(0);

    // ring index
    int ring_index = current_indx / (2 * L);
    // offset for the FTM slice
    int offset = current_indx % (2 * L);
    int ftm_offset = ring_index * (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring
    int offset_ring(0);

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        offset_ring = ring_index * (ring_index + 1) * 2;
        pos = 1;
    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        nphi = 4 * (total_rings - ring_index);
        // Compute lower pixel offset
        int reverse_ring_index = total_rings - ring_index;
        offset_ring = total_pixels - (reverse_ring_index * (reverse_ring_index + 1) * 2);
        pos = -1;
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
        offset_ring = upper_pixels + (ring_index - nside + 1) * 4 * nside;
        pos = 0;
    }

    int slice_start = (L - nphi / 2);
    int slice_end = slice_start + nphi;

    // Fill up the healpix ring
    int ring_print = 0;
    if (offset >= slice_start && offset < slice_end) {
        int center_offset = offset - slice_start;
        indx = center_offset + offset_ring;

        output_c[indx] = data_c[current_indx];
    }
    __syncthreads();
    // printf("Current index %d slice start %d slice end %d nphi %d\n", current_indx, slice_start, slice_end,
    //        nphi);
    // fold the negative part of the spectrum
    if (offset < slice_start) {
        int folded_index = -(1 + offset) % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_start - (1 + offset);

        folded_index = folded_index + offset_ring;
        target_index = target_index + ftm_offset;
        atomicAdd(&output_c[folded_index], data_c[target_index]);
    }
    // fold the positive part of the spectrum
    else if (offset >= slice_end) {
        int folded_index = (offset - slice_end) % nphi;
        folded_index = folded_index < 0 ? nphi + folded_index : folded_index;
        int target_index = slice_end + (offset - slice_end);

        folded_index = folded_index + offset_ring;
        target_index = target_index + ftm_offset;
        atomicAdd(&output_c[folded_index], data_c[target_index]);
    }
}

template <typename real_t>
__global__ void spectral_extension(void* data, void* output, int nside, int L) {
    using complex = cuda::std::complex<real_t>;
    complex* data_c = reinterpret_cast<complex*>(data);
    complex* output_c = reinterpret_cast<complex*>(output);
    // few inits
    int polar_rings = nside - 1;
    int equator_rings = 3 * nside + 1;
    int total_rings = 4 * nside - 1;
    int ftm_size = 2 * L;
    // Compute number of pixels
    int total_pixels = 12 * nside * nside;
    int upper_pixels = nside * (nside - 1) * 2;
    int equator_pixels = 4 * nside * equator_rings;
    // Which ring are we working on
    int current_indx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos(0);
    int indx(0);
    // Compute nphi of current ring
    int nphi(0);

    // ring index
    int ring_index = current_indx / (2 * L);
    // offset for the FTM slice
    int offset = current_indx % (2 * L);
    // offset for original healpix ring
    // Sum of all elements from 0 to n is  n * (n + 1) / 2 in o(1) time .. times 4 to get the number of
    // elements before current ring
    int offset_ring(0);

    // Upper Polar rings
    if (ring_index < nside - 1) {
        nphi = 4 * (ring_index + 1);
        offset_ring = ring_index * (ring_index + 1) * 2;
        pos = 1;
    }
    // Lower Polar rings
    else if (ring_index > 3 * nside - 1) {
        nphi = 4 * (total_rings - ring_index);
        // Compute lower pixel offset
        int reverse_ring_index = total_rings - ring_index;
        offset_ring = total_pixels - (reverse_ring_index * (reverse_ring_index + 1) * 2);
        pos = -1;
    }
    // Equatorial ring
    else {
        nphi = 4 * nside;
        offset_ring = upper_pixels + (ring_index - nside + 1) * 4 * nside;
        pos = 0;
    }

    if (offset < L - nphi / 2) {
        indx = (-(L - nphi / 2 - offset)) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output_c[current_indx] = data_c[indx];
    }

    // Compute the central part of the spectrum
    else if (offset >= L - nphi / 2 && offset < L + nphi / 2) {
        int center_offset = offset - /*negative part offset*/ (L - nphi / 2);
        indx = center_offset + offset_ring;
        output_c[current_indx] = data_c[indx];
    }
    // Compute the positive part of the spectrum
    else {
        int reverse_offset = ftm_size - offset;
        indx = (L - (int)((nphi + 1) / 2) - reverse_offset) % nphi;
        indx = indx < 0 ? nphi + indx : indx;
        indx = indx + offset_ring;
        output_c[current_indx] = data_c[indx];
    }
}

template <typename real_t>
HRESULT launch_spectral_folding(void* data, void* output, const int& nside, const int& L,
                                cudaStream_t stream) {
    int block_size = 128;
    int ftm_elements = 2 * L * (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    spectral_folding<real_t><<<grid_size, block_size, 0, stream>>>(data, output, nside, L);
    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

template <typename real_t>
HRESULT launch_spectral_extension(void* data, void* output, const int& nside, const int& L,
                                  cudaStream_t stream) {
    // Launch the kernel
    int block_size = 128;
    int ftm_elements = 2 * L * (4 * nside - 1);
    int grid_size = (ftm_elements + block_size - 1) / block_size;

    spectral_extension<real_t><<<grid_size, block_size, 0, stream>>>(data, output, nside, L);

    checkCudaErrors(cudaGetLastError());
    return S_OK;
}

// Specializations
template HRESULT launch_spectral_folding<float>(void* data, void* output, const int& nside, const int& L,
                                                cudaStream_t stream);
template HRESULT launch_spectral_folding<double>(void* data, void* output, const int& nside, const int& L,
                                                 cudaStream_t stream);

}  // namespace s2fftKernels