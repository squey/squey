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

#ifndef PVPARALLELVIEW_PBBCIBUFFERS_H
#define PVPARALLELVIEW_PBBCIBUFFERS_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>

#include <tbb/concurrent_queue.h>

#include <cassert>

namespace PVParallelView
{

class PVBCIDrawingBackend;

class PVBCIBuffersAlloc
{
  public:
	typedef PVBCICodeBase bci_base_type;

	static bci_base_type* allocate(size_t n, PVBCIDrawingBackend& backend);
	static void free(bci_base_type* codes, PVBCIDrawingBackend& backend);
};

/**
 * Manager for BCI Buffers.
 *
 * It is a pool of BCI buffers that can be take and return.
 *
 * Depending on initialisation, buffer will be allocated using default allocator or backend
 *allocator.
 *
 * A buffer is big enought to contains data for a full zone.
 */
template <size_t N>
class PVBCIBuffers : private PVBCIBuffersAlloc
{
	static_assert(PARALLELVIEW_MAX_BCI_CODES % 16 == 0,
	              "PARALLELVIEW_MAX_BCI_CODES must be a multiple of 16.");
	static_assert(N >= 2, "The number of BCI buffers must be >= 2.");

  public:
	explicit PVBCIBuffers(PVBCIDrawingBackend& backend) : _backend(&backend)
	{
		// "+2" as sizeof(bci) == 8 and we need 15 more bytes for potential alignment issues
		_org_codes = allocate(PARALLELVIEW_MAX_BCI_CODES * N + 2, backend);

		if (((uintptr_t)(_org_codes)&15) == 0) {
			// Already aligned, good.
			_codes = _org_codes;
		} else {
			_codes = (bci_base_type*)((((uintptr_t)_org_codes + 15) / 16) * 16);
		}

		for (size_t i = 0; i < N; i++) {
			_free_bufs.push(get_buffer_n(i));
		}
	}

	~PVBCIBuffers() { free(_org_codes, *_backend); }

  public:
	bci_base_type* get_available_buffer()
	{
		bci_base_type* ret;
#ifdef NDEBUG
		_free_bufs.try_pop(ret);
#else
		bool success = _free_bufs.try_pop(ret);
		assert(success);
#endif
		return ret;
	}

	void return_buffer(bci_base_type* buf)
	{
		assert(buf >= _codes && buf < get_buffer_n(N));
		assert(std::distance(_codes, buf) % PARALLELVIEW_MAX_BCI_CODES == 0);
		_free_bufs.push(buf);
	}

  private:
	bci_base_type* get_buffer_n(size_t i)
	{
		assert(i <= N);
		bci_base_type* const ret = &_codes[PARALLELVIEW_MAX_BCI_CODES * i];
		assert((uintptr_t)ret % 16 == 0);
		return ret;
	}

  private:
	bci_base_type* _codes;
	bci_base_type* _org_codes;
	PVBCIDrawingBackend* _backend;
	tbb::concurrent_queue<bci_base_type*> _free_bufs;
};
} // namespace PVParallelView

#endif
