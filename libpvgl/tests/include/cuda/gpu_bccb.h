/**
 * \file gpu_bccb.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef GPU_BCODE_COLLISION_H
#define GPU_BCODE_COLLISION_H

#include <common/common.h>
#include <cuda/common.h>

#include <code_bz/types.h>
#include <code_bz/bcode_cb.h>

#include <tbb/concurrent_queue.h>

template <class T>
class GPUPipeBuffer;

void gpu_bccb(PVBCode* codes, size_t n, BCodeCB cb);
void gpu_bccb_2dim(PVBCode* codes, size_t n, BCodeCB cb);
void gpu_bccb_launch_kernel(PVBCode* dev_codes, size_t n, BCodeCB device_cb, int nblocks, cudaEvent_t event, cudaStream_t stream);

class GPUBccb
{
public:
	GPUBccb();
	~GPUBccb();

public:
	void push_bcode_gpu(PVBCode* codes, PVBCode* dev_codes, size_t n, cudaEvent_t* event);
	bool push_bcode_gpu(GPUPipeBuffer<PVBCode>& buf, size_t n, cudaEvent_t* event_);
	void launch_bcode_gpu(PVBCode* dev_buffer, size_t n, cudaEvent_t* event_, cudaStream_t stream);
	void commit_bcode_gpu_and_wait(GPUPipeBuffer<PVBCode>& buf);
	void commit_bcode_gpu(GPUPipeBuffer<PVBCode>& buf);
	void copy_bccb_from_device(BCodeCB host_cb) const;

public:
	PVBCode* allocate_host_bcode_buffer(size_t n, PVBCode** dev_buf, PVBCode** free_buf);
	static void free_host_bcode_buffer(PVBCode* buf, PVBCode* dev_buf);

private:
	BCodeCB _device_bccb;
	int _nblocks;
};


#endif
