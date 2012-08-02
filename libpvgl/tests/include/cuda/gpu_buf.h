/**
 * \file gpu_buf.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef CUDA_GPUBUF_H
#define CUDA_GPUBUF_H

#include <iostream>
#include <common/common.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda/common.h>

#include <tbb/atomic.h>

template <class T>
class GPUPipeBuffer
{
public:
	GPUPipeBuffer(size_t chunk_size, size_t min_transfert = 0):
		_buf(NULL), _dev_buf(NULL), _size(0), _cur(0), _chunk_size(chunk_size), _pos_last_transfert(0), _min_transfert(min_transfert), _in_use()
	{
		if (min_transfert == 0) {
			_min_transfert = (4*1024*1024)/(sizeof(T));
			_min_transfert = (_min_transfert*chunk_size)/chunk_size;
		}
		verify_cuda(cudaStreamCreate(&_stream));
		_must_transfert = false;
	}
	~GPUPipeBuffer()
	{
		verify_cuda(cudaStreamDestroy(_stream));
		free();
	}
public:
	void allocate(size_t n, size_t aligned)
	{
		_size = n*_chunk_size;
		_allocate_host(_size, aligned);
		_allocate_dev(_size);
	}

	void free()
	{
		if (_free_buf) {
			cudaFreeHost(_free_buf);
			cudaFree(_dev_buf);
			_free_buf = NULL;
			_dev_buf = NULL;
			_buf = 0;
			_size = 0;
			_cur = 0;
		}
	}

public:
	inline T* host() { return _buf; }
	inline T* dev() { return _dev_buf; }

	bool push_data(int n)
	{
		bool ret = _push_data(n);
		return ret;
	}

	inline void rewind() { _cur = 0; _pos_last_transfert = 0;_must_transfert = false; }

	inline size_t size() const { return _size; }
	inline size_t cur() const { return _cur; }
	inline cudaStream_t stream() const { return _stream; }
	inline size_t transfert()
	{
		size_t n_elts = _cur-_pos_last_transfert;
		cudaMemcpyAsync(&_dev_buf[_pos_last_transfert], &_buf[_pos_last_transfert], n_elts*sizeof(T), cudaMemcpyHostToDevice, _stream);
		_pos_last_transfert = _cur;
		_must_transfert = false;
		return n_elts;
	}
	inline bool is_full() const { return _size-_cur < _chunk_size; }
	inline bool can_use() { return !_in_use.compare_and_swap(true, false); }
	inline void free_use()
	{
		tbb::atomic<bool> t;
		t = false;
		_in_use = t;
	}

private:
	void _allocate_host(size_t n, size_t aligned)
	{
		verify_cuda(cudaHostAlloc(&_free_buf, n*sizeof(T)+aligned, 0));
		_buf = (T*) ((((uintptr_t)_free_buf+aligned-1)/aligned)*aligned);
	}

	void _allocate_dev(size_t n)
	{
		verify_cuda(cudaMalloc(&_dev_buf, n*sizeof(T)));
	}

	bool _push_data(size_t n)
	{
		_cur += n;
		VERIFY(!_must_transfert);
		VERIFY(_cur <= _size);
		//printf("cur: %u | size: %u | last-trans: %u | min-trans: %u | chunk-size: %u", _cur, _size, _pos_last_transfert, _min_transfert, _chunk_size);
		_must_transfert = (_cur - _pos_last_transfert >= _min_transfert || is_full());
		return _must_transfert;
	}

private:
	T* _buf;
	T* _dev_buf;
	T* _free_buf;
	size_t _size;
	size_t _cur;
	size_t _chunk_size;
	cudaStream_t _stream;
	size_t _pos_last_transfert;
	size_t _min_transfert;
	bool _must_transfert;
	tbb::atomic<bool> _in_use;
};

#endif
