/**
 * \file cuda_cb.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef CUDA_CB
#define CUDA_CB

#include "collision_buf.h"
#include <string.h>

class CudaCB
{
public:
	__host__ __device__
	CudaCB()
	{
	}

	__host__ __device__
	CudaCB(const& CudaCB other)
	{
		copy_cb(other);
	}

public:
	__host__ __device__
	CudaCB& operator=(const CudaCB& other)
	{
		if (&other == this) {
			return *this;
		}
		copy_cb(other);
	}

	__host__ __device__
	CudaCB operator|=(const CudaCB& other)
	{
		for (size_t i = 0; i < NB_INT_CB; i++) {
			_cb[i] |= other._cb[i];
		}
	}
private:
	inline void copy_cb(const CudaCB& other) { memcpy(_cb, other._cb, SIZE_CB); }
	int _cb[NB_INT_CB];
};
