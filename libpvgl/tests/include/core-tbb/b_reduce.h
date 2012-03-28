#ifndef BCODEREDUCE_H
#define BCODEREDUCE_H

#include <common/common.h>
#include <code_bz/bcode_cb.h>
#include <code_bz/types.h>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <cassert>
#include <string.h>

class BCodeReduce
{
public:
	BCodeReduce(PVBCode* codes):
		_codes(codes)
	{
		//_buf = allocate_BCodeCB();
		memset(_buf, 0,SIZE_BCODECB);
	}

	BCodeReduce(BCodeReduce const& x, tbb::split):
		_codes(x._codes)
	{
		//_buf = allocate_BCodeCB();
		memset(_buf, 0, SIZE_BCODECB);
	}

	~BCodeReduce()
	{
		//free_BCodeCB(_buf);
	}
public:
	void operator()(const tbb::blocked_range<size_t>& r)
	{
		PVBCode tmp;
		PVBCode const* codes = _codes;
		size_t end = r.end();
		for (size_t i = r.begin(); i != end; i++) {
			tmp = codes[i];
			// Set CB
			assert_bcode_valid(tmp);
			B_SET(_buf[CB_INT_OFFSET(tmp.int_v)], CB_BITN(tmp.int_v));
		}
	}

	void join(BCodeReduce const& x)
	{
		const uint32_t DECLARE_ALIGN(16) *xcb = x._buf;
		for (size_t i = 0; i < NB_INT_BCODECB; i++) {
			_buf[i] |= xcb[i];
		}
	}

	BCodeCB cb() { return (BCodeCB) &_buf[0]; }
private:
	uint32_t DECLARE_ALIGN(16) _buf[NB_INT_BCODECB];
	PVBCode* _codes;
};

#endif
