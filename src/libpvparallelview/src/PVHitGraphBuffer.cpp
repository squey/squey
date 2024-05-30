//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVHitGraphBuffer.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

//
// PVHitGraphBuffer
//

PVParallelView::PVHitGraphBuffer::PVHitGraphBuffer(uint32_t nbits, uint32_t nblocks)
    : _nbits(nbits), _nblocks(nblocks), _size_block(1 << nbits)
{
	posix_memalign((void**)&_buf, 16, size_bytes());
}

PVParallelView::PVHitGraphBuffer::~PVHitGraphBuffer()
{
	if (_buf) {
		free(_buf);
	}
}

void PVParallelView::PVHitGraphBuffer::set_zero()
{
	memset(_buf, 0, size_bytes());
}

void PVParallelView::PVHitGraphBuffer::shift_left(const uint32_t n)
{
	assert(n <= nblocks());
	const uint32_t nb_moved_blocks = nblocks() - n;

	memmove(buffer(), buffer_block(n), nb_moved_blocks * size_block() * sizeof(uint32_t));
	memset(buffer_block(nb_moved_blocks), 0, n * size_block() * sizeof(uint32_t));
}

void PVParallelView::PVHitGraphBuffer::shift_zoomed_left(const uint32_t n, const float alpha)
{
	assert(n <= nblocks());
	const uint32_t nb_moved_blocks = nblocks() - n;

	memmove(buffer(), zoomed_buffer_block(n, alpha),
	        nb_moved_blocks * size_zoomed_block(alpha) * sizeof(uint32_t));
	memset(zoomed_buffer_block(nb_moved_blocks, alpha), 0,
	       n * size_zoomed_block(alpha) * sizeof(uint32_t));
}

void PVParallelView::PVHitGraphBuffer::shift_right(const uint32_t n)
{
	assert(n <= nblocks());
	const uint32_t nb_moved_blocks = nblocks() - n;

	memmove(buffer_block(n), buffer(), nb_moved_blocks * size_block() * sizeof(uint32_t));
	memset(buffer(), 0, n * size_block() * sizeof(uint32_t));
}

void PVParallelView::PVHitGraphBuffer::shift_zoomed_right(const uint32_t n, const float alpha)
{
	assert(n <= nblocks());
	const uint32_t nb_moved_blocks = nblocks() - n;

	memmove(zoomed_buffer_block(n, alpha), buffer(),
	        nb_moved_blocks * size_zoomed_block(alpha) * sizeof(uint32_t));
	memset(buffer(), 0, n * size_zoomed_block(alpha) * sizeof(uint32_t));
}

bool PVParallelView::PVHitGraphBuffer::copy_from(PVHitGraphBuffer const& o)
{
	if (o.size_int() != size_int()) {
		return false;
	}

	memcpy(_buf, o._buf, size_bytes());

	return true;
}

uint32_t PVParallelView::PVHitGraphBuffer::get_max_count() const
{
	// GCC should vectorize this !
	const size_t nints = size_block() * nblocks();
	uint32_t ret = 0;
	for (size_t i = 0; i < nints; i++) {
		const uint32_t v = _buf[i];
		if (v > ret) {
			ret = v;
		}
	}
	return ret;
}

uint32_t PVParallelView::PVHitGraphBuffer::get_zoomed_max_count(const float alpha) const
{
	// GCC should vectorize this !
	const size_t nints = size_zoomed_block(alpha) * nblocks();
	uint32_t ret = 0;
	for (size_t i = 0; i < nints; i++) {
		const uint32_t v = _buf[i];
		if (v > ret) {
			ret = v;
		}
	}
	return ret;
}
