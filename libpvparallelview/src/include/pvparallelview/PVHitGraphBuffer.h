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
	static constexpr size_t NBITS = PVHitGraphCommon::NBITS;
	static constexpr size_t SIZE_BLOCK = 1<<NBITS;
	static constexpr size_t NBLOCKS = PVHitGraphCommon::NBLOCKS;

public:
	PVHitGraphBuffer();
	~PVHitGraphBuffer();

	PVHitGraphBuffer(PVHitGraphBuffer&& o)
	{
		_buf = o._buf;
		o._buf = NULL;
	}

public:
	static inline size_t size_bytes() { return NBLOCKS*SIZE_BLOCK*sizeof(uint32_t); }

public:
	PVHitGraphBuffer& operator=(PVHitGraphBuffer&& o)
	{
		if (&o != this) {
			_buf = o._buf;
			o._buf = NULL;
		}
		return *this;
	}

public:
	inline uint32_t* buffer() { return _buf; }
	inline uint32_t const* buffer() const { return _buf; }

	inline uint32_t* buffer_block(int n) { return &_buf[n*SIZE_BLOCK]; }
	inline uint32_t const* buffer_block(int n) const { return &_buf[n*SIZE_BLOCK]; }

	inline uint32_t at(int block, int idx) const { return buffer_block(block)[idx]; }

public:
	void set_zero();
	void shift_left(int nblocks);
	void shift_right(int nblocks);

private:
	uint32_t* _buf;
};

}

#endif
