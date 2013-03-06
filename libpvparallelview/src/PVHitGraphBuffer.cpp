
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


void PVParallelView::PVHitGraphBuffer::shift_left(const uint32_t n, const float alpha)
{
	assert(n <= nblocks());
	// We need to shift both the original buffer and the "zoom reduced" one
	const uint32_t nb_moved_blocks = nblocks()-n;
	
	// Original buffer
	memmove(buffer(), buffer_block(n), nb_moved_blocks*size_block()*sizeof(uint32_t));
	memset(buffer_block(nb_moved_blocks), 0, n*size_block()*sizeof(uint32_t));

	// Zoomed buffer
	memmove(zoomed_buffer(), zoomed_buffer_block(n, alpha), nb_moved_blocks*size_zoomed_block(alpha)*sizeof(uint32_t));
	memset(zoomed_buffer_block(nb_moved_blocks, alpha), 0, n*size_zoomed_block(alpha)*sizeof(uint32_t));
}

void PVParallelView::PVHitGraphBuffer::shift_right(const uint32_t n, const float alpha)
{
	assert(n <= nblocks());
	// We need to shift both the original buffer and the "zoom reduced" one
	const uint32_t nb_moved_blocks = nblocks()-n;
	
	// Original buffer
	memmove(buffer_block(n), buffer(), nb_moved_blocks*size_block()*sizeof(uint32_t));
	memset(buffer(), 0, n*size_block()*sizeof(uint32_t));

	// Zoomed buffer
	memmove(zoomed_buffer_block(n, alpha), zoomed_buffer(), nb_moved_blocks*size_zoomed_block(alpha)*sizeof(uint32_t));
	memset(zoomed_buffer(), 0, n*size_zoomed_block(alpha)*sizeof(uint32_t));
}

void PVParallelView::PVHitGraphBuffer::process_zoom_reduction_inplace(const float alpha)
{
	assert((alpha >= 0.5f) && (alpha < 1.0f));
	const int sint = size_int();
	for (int idx = 1; idx < sint; idx++) {
		// AG: this must be the same "rounding" method than the one used
		// in the CUDA kernel.
		// TODO: this formula should be exported in a common place and used
		// everywhere a "pixel" position is needed for a given zoom level.
		const int new_idx = (int) ((float)idx*alpha);
		if (new_idx != idx) {
			_buf[new_idx] += _buf[idx];
			_buf[idx] = 0;
		}
	}
}

void PVParallelView::PVHitGraphBuffer::process_zoom_reduction(const float alpha, uint32_t* res)
{
	assert((alpha >= 0.5f) && (alpha < 1.0f));
	memset(res, 0, size_bytes());
	res[0] = _buf[0];
	const int sint = size_int();
	for (int idx = 1; idx < sint; idx++) {
		const int new_idx = (int) ((float)idx*alpha);
		res[new_idx] += _buf[idx];
	}
}
