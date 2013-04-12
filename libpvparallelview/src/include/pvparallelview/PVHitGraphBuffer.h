#ifndef PVPARALLELVIEW_PVHITGRAPHBUFFER_H
#define PVPARALLELVIEW_PVHITGRAPHBUFFER_H

#include <pvparallelview/PVHitGraphCommon.h>

#include <boost/noncopyable.hpp>

#include <cstddef>
#include <cstdint>

namespace PVParallelView {

class PVHitGraphBuffer: boost::noncopyable
{
public:
	PVHitGraphBuffer(uint32_t nbits, uint32_t nblocks);
	~PVHitGraphBuffer();

	PVHitGraphBuffer(PVHitGraphBuffer&& o)
	{
		_buf = o._buf;
		o._buf = NULL;

		_zoomed_buf = o._zoomed_buf;
		o._zoomed_buf = NULL;

		_nbits = o._nbits;
		_nblocks = o._nblocks;
		_size_block = o._size_block;
	}

public:
	inline size_t size_bytes() const { return size_int()*sizeof(uint32_t); }
	inline size_t size_int()   const { return nblocks()*size_block(); }

public:
	PVHitGraphBuffer& operator=(PVHitGraphBuffer&& o)
	{
		if (&o != this) {
			_buf = o._buf;
			o._buf = NULL;

			_zoomed_buf = o._zoomed_buf;
			o._zoomed_buf = NULL;

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

	inline uint32_t* buffer_block(uint32_t n) { return &_buf[n*size_block()]; }
	inline uint32_t const* buffer_block(uint32_t n) const { return &_buf[n*size_block()]; }

	inline uint32_t at(uint32_t block, uint32_t idx) const { return buffer_block(block)[idx]; }

	inline uint32_t* zoomed_buffer() { return _zoomed_buf; }
	inline uint32_t const* zoomed_buffer() const { return _zoomed_buf; }

	inline uint32_t* zoomed_buffer_block(uint32_t n, const float alpha)
	{
		return &_zoomed_buf[n*size_zoomed_block(alpha)];
	}
	inline uint32_t const* zoomed_buffer_block(uint32_t n, const float alpha) const
	{
		return &_zoomed_buf[n*size_zoomed_block(alpha)];
	}

public:
	void set_zero();
	void shift_left(const uint32_t nblocks);
	void shift_right(const uint32_t nblocks);
	void shift_zoomed_left(const uint32_t nblocks, const float alpha);
	void shift_zoomed_right(const uint32_t nblocks, const float alpha);
	void process_zoom_reduction_inplace(const float alpha);
	inline void process_zoom_reduction(const float alpha) { process_zoom_reduction(alpha, _zoomed_buf); }

public:
	inline uint32_t nbits() const { return _nbits; }
	// returned size is in number of integers
	inline uint32_t size_block() const { return _size_block; }
	inline uint32_t size_zoomed_block(const float alpha) const { return (int)((float)size_block()*alpha); }
	inline uint32_t nblocks() const { return _nblocks; }

private:
	void process_zoom_reduction(const float alpha, uint32_t* res);

private:
	uint32_t* _buf;
	uint32_t* _zoomed_buf;

	uint32_t _nbits;
	uint32_t _nblocks;
	uint32_t _size_block;

};

}

#endif
