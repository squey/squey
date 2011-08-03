//! \file PVCudaDefine.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <picviz/limits.h>



//cuda error detection
#define check_cuda() { cudaError_t err = cudaGetLastError();if(err!=cudaSuccess){  PVLOG_CUDA_ERROR("%s  :  file %s   : line %d\n",cudaGetErrorString( err ),__FILE__,__LINE__);cudaDeviceReset();} }

//cuda enable printf in kernel
#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define CUPRINTF {;}
#else				/Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) { printf(fmt);}
#endif

#define dbg { PVLOG_INFO("CUDA DBG : %s:%d\n",__FILE__,__LINE__);}


#define atomic_exec(mutex_to_lock,code) { bool lock=true; while(lock){if(0==atomicCAS(mutex_to_lock,0,1)){ code  atomicExch(mutex_to_lock,0);lock=false; }  } }



