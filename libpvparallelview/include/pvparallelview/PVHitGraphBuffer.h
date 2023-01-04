/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVHITGRAPHBUFFER_H
#define PVPARALLELVIEW_PVHITGRAPHBUFFER_H

#include <pvparallelview/PVHitGraphCommon.h>

#include <boost/noncopyable.hpp>

#include <cstddef>
#include <cstdint>

namespace PVParallelView
{

class PVHitGraphBuffer : boost::noncopyable
{
  public:
	PVHitGraphBuffer(uint32_t nbits, uint32_t nblocks);
	~PVHitGraphBuffer();

	PVHitGraphBuffer(PVHitGraphBuffer&& o)
	{
		_buf = o._buf;
		o._buf = nullptr;

		_nbits = o._nbits;
		_nblocks = o._nblocks;
		_size_block = o._size_block;
	}

  public:
	inline size_t size_bytes() const { return size_int() * sizeof(uint32_t); }
	inline size_t size_int() const { return nblocks() * size_block(); }

  public:
	PVHitGraphBuffer& operator=(PVHitGraphBuffer&& o)
	{
		if (&o != this) {
			_buf = o._buf;
			o._buf = nullptr;

			_nbits = o._nbits;
			_nblocks = o._nblocks;
			_size_block = o._size_block;
		}
		return *this;
	}

	bool copy_from(PVHitGraphBuffer const& other);

  public:
	inline uint32_t* buffer() { return _buf; }
	inline uint32_t const* buffer() const { return _buf; }

	inline uint32_t* buffer_block(uint32_t n) { return &_buf[n * size_block()]; }
	inline uint32_t const* buffer_block(uint32_t n) const { return &_buf[n * size_block()]; }

	inline uint32_t at(uint32_t block, uint32_t idx) const { return buffer_block(block)[idx]; }

	inline uint32_t* zoomed_buffer_block(uint32_t n, const float alpha)
	{
		return &_buf[n * size_zoomed_block(alpha)];
		// return &_zoomed_buf[n*size_zoomed_block(alpha)];
	}
	inline uint32_t const* zoomed_buffer_block(uint32_t n, const float alpha) const
	{
		// return &_zoomed_buf[n*size_zoomed_block(alpha)];
		return &_buf[n * size_zoomed_block(alpha)];
	}

	uint32_t get_zoomed_max_count(const float alpha) const;
	uint32_t get_max_count() const;

  public:
	void set_zero();
	void shift_left(const uint32_t nblocks);
	void shift_right(const uint32_t nblocks);
	void shift_zoomed_left(const uint32_t nblocks, const float alpha);
	void shift_zoomed_right(const uint32_t nblocks, const float alpha);

  public:
	inline uint32_t nbits() const { return _nbits; }
	// returned size is in number of integers
	inline uint32_t size_block() const { return _size_block; }
	inline uint32_t size_zoomed_block(const float alpha) const
	{
		return (int)((float)size_block() * alpha);
	}
	inline uint32_t nblocks() const { return _nblocks; }

  private:
	uint32_t* _buf;

	uint32_t _nbits;
	uint32_t _nblocks;
	uint32_t _size_block;
};
} // namespace PVParallelView

#endif
