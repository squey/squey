/**
 * \file selection_kernels.cu
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "selection_kernels.h"

__global__ void picviz_selection_AB2C_or_k(uint32_t *da, uint32_t *db, uint32_t *dc)
{
	int chunk_index = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	
	for (i=chunk_index*100;i < (chunk_index+1)*100;++i) {
//	    da[i] = 0xff;
	    dc[i] = da[i] | db[i];
 	}
}
