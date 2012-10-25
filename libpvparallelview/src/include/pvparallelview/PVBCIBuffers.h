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
	static_assert(PARALLELVIEW_MAX_BCI_CODES % 16 == 0, "PARALLELVIEW_MAX_BCI_CODES must be a multiple of 16.");
	static_assert(N >= 2, "The number of BCI buffers must be >= 2.");

	typedef PVBCICode<> bci_type;
	typedef PVBCICodeBase bci_base_type;

public:
	PVBCIBuffers()
	{
		_codes = reinterpret_cast<bci_base_type*>(bci_type::allocate_codes(PARALLELVIEW_MAX_BCI_CODES*N));
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
	bci_base_type* get_available_buffer()
	{
		bci_base_type* ret;
		_free_bufs.pop(ret);
		return ret;
	}

	void return_buffer(bci_base_type* buf)
	{
		assert(buf >= _codes && buf < get_buffer_n(N));
		assert(std::distance(_codes, buf) % PARALLELVIEW_MAX_BCI_CODES == 0);
#ifdef NDEBUG
		_free_bufs.try_push(buf);
#else
		bool success = _free_bufs.try_push(buf);
		assert(success);
#endif
	}

private:
	bci_base_type* get_buffer_n(size_t i)
	{
		assert(i <= N);
		bci_base_type* const ret = &_codes[PARALLELVIEW_MAX_BCI_CODES*i];
		assert((uintptr_t)ret % 16 == 0);
		return ret;
	}

private:
	bci_base_type* _codes;
	tbb::concurrent_bounded_queue<bci_base_type*> _free_bufs;
};

}

#endif
