#include <common/common.h>
#include <cuda/gpu_bccb.h>
#include <cuda/gpu_buf.h>

GPUBccb::GPUBccb()
{
	// Allocate GPU collision buffer
	verify_cuda(cudaMalloc(&_device_bccb, SIZE_BCODECB));
	verify_cuda(cudaMemset(_device_bccb, 0x00, SIZE_BCODECB));
	_nblocks = get_number_blocks()*2;
}

GPUBccb::~GPUBccb()
{
	cudaFree(_device_bccb);
}

PVBCode* GPUBccb::allocate_host_bcode_buffer(size_t n, PVBCode** dev_buf, PVBCode** free_buf)
{
	// Use pinned memory for faster memory transfers
	// Align the memory on  a 16-byte boundary
	PVBCode* ret;
	verify_cuda(cudaHostAlloc(free_buf, n*sizeof(PVBCode)+17, 0));
	ret = (PVBCode*) ((((uintptr_t)(*free_buf)+15)/16)*16);

	// Allocate a buffer for the device
	verify_cuda(cudaMalloc(dev_buf, sizeof(PVBCode)*n));
	//posix_memalign((void**) &ret, 16, sizeof(PVBCode)*n);
	//*free_buf = ret;
	return ret;
}

void GPUBccb::free_host_bcode_buffer(PVBCode* buf, PVBCode* dev_buf)
{
	verify_cuda(cudaFreeHost(buf));
	verify_cuda(cudaFree(dev_buf));
	//free(buf);
}

void GPUBccb::push_bcode_gpu(PVBCode* codes, PVBCode* dev_codes, size_t n, cudaEvent_t* event_)
{
	cudaEvent_t event = NULL;
	if (event_) {
		verify_cuda(cudaEventCreate(event_));
		event = *event_;
	}
	
	//PVBCode* device_codes;
	//verify_cuda(cudaHostGetDevicePointer(&device_codes, codes, 0));
	verify_cuda(cudaMemcpyAsync(dev_codes, codes, n*sizeof(PVBCode), cudaMemcpyHostToDevice));
	gpu_bccb_launch_kernel(dev_codes, n, _device_bccb, _nblocks, event, 0);
}

bool GPUBccb::push_bcode_gpu(GPUPipeBuffer<PVBCode>& buf, size_t n, cudaEvent_t* event_)
{
	if (buf.push_data(n)) {
		size_t n_elts = buf.transfert();
		launch_bcode_gpu(buf.dev(), n_elts, event_, buf.stream());
		return true;
	}
	return false;
}

void GPUBccb::launch_bcode_gpu(PVBCode* dev_buffer, size_t n, cudaEvent_t* event_, cudaStream_t stream)
{
	cudaEvent_t event = NULL;
	if (event_) {
		verify_cuda(cudaEventCreate(event_));
		event = *event_;
	}
	gpu_bccb_launch_kernel(dev_buffer, n, _device_bccb, _nblocks, event, stream);
}

void GPUBccb::commit_bcode_gpu_and_wait(GPUPipeBuffer<PVBCode>& buf)
{
	if (cudaStreamQuery(buf.stream()) == cudaSuccess || buf.cur() > 0) {
		cudaEvent_t event;
		cudaEventCreate(&event);
		gpu_bccb_launch_kernel(buf.dev(), buf.cur(), _device_bccb, _nblocks, event, buf.stream());
		cudaEventSynchronize(event);
	}
}

void GPUBccb::commit_bcode_gpu(GPUPipeBuffer<PVBCode>& buf)
{
	printf("commit: buf.cur() == %u\n", buf.cur());
	if (cudaStreamQuery(buf.stream()) == cudaSuccess || buf.cur() > 0) {
		size_t ncodes = buf.transfert();
		printf("commit: transfered %u elements\n", ncodes);
		gpu_bccb_launch_kernel(buf.dev(), buf.cur(), _device_bccb, _nblocks, NULL, buf.stream());
	}
}

void GPUBccb::copy_bccb_from_device(BCodeCB host_cb) const
{
	cudaMemcpy(host_cb, _device_bccb, SIZE_BCODECB, cudaMemcpyDeviceToHost);
}
