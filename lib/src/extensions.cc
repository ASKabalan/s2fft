
#include "kernel_nanobind_helpers.h"
#include "kernel_helpers.h"
#include <nanobind/nanobind.h>
#include <cstddef>
#include "cuda_runtime.h"
#include "plan_cache.h"
#include "s2fft_kernels.h"
#include "s2fft.h"

namespace nb = nanobind;

namespace s2fft {

void healpix_forward(cudaStream_t stream, void** buffers, s2fftDescriptor descriptor) {
    void* data = buffers[0];
    size_t* work = reinterpret_cast<size_t*>(buffers[1]);
    void* output = buffers[2];

    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.double_precision) {
        auto executor = std::make_shared<s2fftExec<double>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the fft part
        executor->Forward(descriptor, stream, data);
        // Run the spectral extension part
        s2fftKernels::launch_spectral_extension<double>(data, output, descriptor.nside,
                                                        descriptor.harmonic_band_limit, stream);

    } else {
        auto executor = std::make_shared<s2fftExec<float>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the fft part
        executor->Forward(descriptor, stream, data);
        // Run the spectral extension part
        s2fftKernels::launch_spectral_extension<float>(data, output, descriptor.nside,
                                                       descriptor.harmonic_band_limit, stream);
    }
}

void healpix_backward(cudaStream_t stream, void** buffers, s2fftDescriptor descriptor) {
    void* data = buffers[0];
    size_t* work = reinterpret_cast<size_t*>(buffers[1]);
    void* output = buffers[2];

    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.double_precision) {
        auto executor = std::make_shared<s2fftExec<double>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the spectral folding part
        s2fftKernels::launch_spectral_folding<double>(data, output, descriptor.nside,
                                                      descriptor.harmonic_band_limit, stream);
        // Run the fft part
        executor->Backward(descriptor, stream, output);

    } else {
        auto executor = std::make_shared<s2fftExec<float>>();
        PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
        // Run the spectral folding part
        s2fftKernels::launch_spectral_folding<float>(data, output, descriptor.nside,
                                                     descriptor.harmonic_band_limit, stream);
        // Run the fft part
        executor->Backward(descriptor, stream, output);
    }
}

void healpix_fft_cuda(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
    // Get the descriptor from the opaque parameter
    s2fftDescriptor descriptor = *UnpackDescriptor<s2fftDescriptor>(opaque, opaque_len);
    size_t work_size;
    // Execute the kernel based on the Precision
    if (descriptor.forward) {
        healpix_forward(stream, buffers, descriptor);
    } else {
        healpix_backward(stream, buffers, descriptor);
    }
}

nb::dict Registration() {
    nb::dict dict;
    dict["healpix_fft_cuda"] = EncapsulateFunction(healpix_fft_cuda);
    return dict;
}

}  // namespace s2fft

NB_MODULE(_s2fft, m) {
    m.def("registration", &s2fft::Registration);

    m.def("build_healpix_fft_descriptor",
          [](int nside, int harmonic_band_limit, bool reality, bool forward, bool double_precision) {
              size_t work_size;
              // Only backward for now
              s2fft::fft_norm norm = s2fft::fft_norm::BACKWARD;
              // Always shift
              bool shift = true;
              s2fft::s2fftDescriptor descriptor(nside, harmonic_band_limit, reality, forward, norm, shift,
                                                double_precision);

              if (double_precision) {
                  auto executor = std::make_shared<s2fft::s2fftExec<double>>();
                  s2fft::PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
                  executor->Initialize(descriptor, work_size);
                  return std::pair<size_t, nb::bytes>(work_size, s2fft::PackDescriptor(descriptor));
              } else {
                  auto executor = std::make_shared<s2fft::s2fftExec<float>>();
                  s2fft::PlanCache::GetInstance().GetS2FFTExec(descriptor, executor);
                  executor->Initialize(descriptor, work_size);
                  return std::pair<size_t, nb::bytes>(work_size, s2fft::PackDescriptor(descriptor));
              }
          });
}