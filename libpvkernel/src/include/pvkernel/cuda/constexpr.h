/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVKERNEL_CUDA_CONSTEXPR_H
#define PVKERNEL_CUDA_CONSTEXPR_H

#ifdef __CUDACC__
// nvcc does not support C++0x !
#define CUDA_CONSTEXPR const
#else
#define CUDA_CONSTEXPR constexpr
#endif

#endif
