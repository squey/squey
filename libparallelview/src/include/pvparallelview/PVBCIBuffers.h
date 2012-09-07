#ifndef PVPARALLELVIEW_PBBCIBUFFERS_H
#define PVPARALLELVIEW_PBBCIBUFFERS_H

#include <pvkernel/core/general.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>

#include <tbb/concurrent_queue.h>

#include <cassert>

namespace PVParallelView {

template <size_t N>
class PVBCIBuffers
{
	static_assert(NBUCKETS % 16 == 0, "NBUCKETS must be a multiple of 16.");
	static_assert(N >= 2, "The number of BCI buffers must be >= 2.");

	typedef PVBCICode<> bci_type;
	typedef bci_type::int_type int_type;

public:
	PVBCIBuffers()
	{
		_codes = reinterpret_cast<int_type*>(bci_type::allocate_codes(NBUCKETS*N));
		_free_bufs.set_capacity(N);
		for (size_t i = 0; i < N; i++) {
			_free_bufs.push(get_buffer_n(i));
		}
	}

	~PVBCIBuffers()
	{
		bci_type::free_codes(reinterpret_cast<bci_type*>(_codes));
	}

public:
	template <size_t Bbits>
	PVBCICode<Bbits>* get_available_buffer()
	{
		int_type* ret;
		_free_bufs.pop(ret);
		return reinterpret_cast<PVBCICode<Bbits>*>(ret);
	}

	template <size_t Bbits>
	void return_buffer(PVBCICode<Bbits>* bci_buf)
	{
		int_type* buf = reinterpret_cast<int_type*>(bci_buf);
		assert(buf >= _codes && buf < get_buffer_n(N));
		assert(std::distance(_codes, buf) % NBUCKETS == 0);
#ifdef NDEBUG
		_free_bufs.try_push(buf);
#else
		bool success = _free_bufs.try_push(buf);
		assert(success);
#endif
	}
private:
	int_type* get_buffer_n(size_t i)
	{
		assert(i <= N);
		int_type* const ret = &_codes[NBUCKETS*i];
		assert((uintptr_t)ret % 16 == 0);
		return ret;
	}

private:
	int_type* _codes;
	tbb::concurrent_bounded_queue<int_type*> _free_bufs;
};

}

#endif
