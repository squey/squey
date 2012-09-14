#ifndef PVKERNEL_CUDA_CONSTEXPR_H
#define PVKERNEL_CUDA_CONSTEXPR_H

#ifdef __CUDACC__
// nvcc does not support C++0x !
#define CUDA_CONSTEXPR const
#else
#define CUDA_CONSTEXPR constexpr
#endif

#endif
