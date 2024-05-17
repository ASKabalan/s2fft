
#ifndef PLAN_CACHE_H
#define PLAN_CACHE_H

#include "logger.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda/std/complex>
#include "hresult.h"
#include "s2fft.h"
#include <unordered_map>

namespace s2fft {

class PlanCache {
public:
    static PlanCache &GetInstance() {
        static PlanCache instance;
        return instance;
    }

    HRESULT GetS2FFTExec(s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<float>> &executor);

    HRESULT GetS2FFTExec(s2fftDescriptor &descriptor, std::shared_ptr<s2fftExec<double>> &executor);

    ~PlanCache() {}

private:
    bool is_initialized = false;

    std::unordered_map<s2fftDescriptor, std::shared_ptr<s2fftExec<double>>, std::hash<s2fftDescriptor>,
                       std::equal_to<>>
            m_Descriptors64;
    std::unordered_map<s2fftDescriptor, std::shared_ptr<s2fftExec<float>>, std::hash<s2fftDescriptor>,
                       std::equal_to<>>
            m_Descriptors32;

    PlanCache();

public:
    PlanCache(PlanCache const &) = delete;
    void operator=(PlanCache const &) = delete;
};
}  // namespace s2fft

#endif  // PLAN_CACHE_H