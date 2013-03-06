
#include <pvparallelview/PVHitGraphBuffer.h>

#include <cassert>
#include <cstdlib>
#include <string.h>

// 
// PVHitGraphBuffer
//

PVParallelView::PVHitGraphBuffer::PVHitGraphBuffer(uint32_t nbits, uint32_t nblocks):
	_nbits(nbits),
	_nblocks(nblocks),
	_size_block(1<<nbits)
{
	posix_memalign((void**) &_buf, 16, size_bytes());
	posix_memalign((void**) &_zoomed_buf, 16, size_bytes());
}

PVParallelView::PVHitGraphBuffer::~PVHitGraphBuffer()
{
	if (_buf) {
		free(_buf);
	}
	if (_zoomed_buf) {
		free(_zoomed_buf);
	}
}

void PVParallelView::PVHitGraphBuffer::set_zero()
{
	memset(_buf, 0, size_bytes());
}


void PVParallelView::PVHitGraphBuffer::shift_left(int n)
{
	// TODO: implement
}

void PVParallelView::PVHitGraphBuffer::shift_right(int n)
{
	// TODO: implement
}

void PVParallelView::PVHitGraphBuffer::process_zoom_reduction_inplace(const float alpha)
{
	assert(alpha >= 0.5f && alpha <= 1.0f);
	const int sint = size_int();
	for (int idx = 1; idx < sint; idx++) {
		// AG: this must be the same "rounding" method than the one used
		// in the CUDA kernel.
		// TODO: this formula should be exported in a common place and used
		// everywhere a "pixel" position is needed for a given zoom level.
		const int new_idx = (int) (((float)idx*alpha) + 0.5f);
		if (new_idx != idx) {
			_buf[new_idx] += _buf[idx];
			_buf[idx] = 0;
		}
	}
}

void PVParallelView::PVHitGraphBuffer::process_zoom_reduction(const float alpha, uint32_t* res)
{
	assert(alpha >= 0.5f && alpha <= 1.0f);
	memset(res, 0, size_bytes());
	res[0] = _buf[0];
	const int sint = size_int();
	for (int idx = 1; idx < sint; idx++) {
		const int new_idx = (int) (((float)idx*alpha) + 0.5f);
		res[new_idx] += _buf[idx];
	}
}
