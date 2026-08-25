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

#include <pvkernel/core/squey_intrin.h>
#include <pvhwloc.h>

#include <squey/PVSelection.h>

#include <pvkernel/core/PVCpuFeatures.h>
#include <pvparallelview/PVHitGraphDataOMP.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

//#include <numa.h>
#include <omp.h>

// Constants used by the OMP code
//#define NBITS (PVParallelView::PVHitGraphCommon::NBITS) // Number of bits used
// by the reduction
//#define BUFFER_SIZE (PVParallelView::PVHitGraphBuffer::SIZE_BLOCK) // Number
// of integers in one block

//
// OMP specific context structure
//

PVParallelView::PVHitGraphDataOMP::omp_ctx_t::omp_ctx_t(uint32_t size)
{
	// "size" is the number of integers of a thread-specific buffer
	// (thus = nblocks * size_int_block)
	_core_num = pvhwloc::core_count();
	_buffers = new uint32_t*[_core_num];
	_buffer_size = size;

	for (uint32_t i = 0; i < _core_num; ++i) {
		_buffers[i] = new uint32_t[_buffer_size];
		/**
		 * NOTE: using NUMA allocator leads to a deadlock in the libc. A simple work-around
		 * is to avoid using it.
		 *
		 * (uint32_t*)numa_alloc_onnode(_buffer_size * sizeof(uint32_t), numa_node_of_cpu(i));
		 */
		memset(_buffers[i], 0, size * sizeof(uint32_t));
	}
}

PVParallelView::PVHitGraphDataOMP::omp_ctx_t::~omp_ctx_t()
{
	if (_buffers) {
		for (uint32_t i = 0; i < _core_num; ++i) {
			if (_buffers[i]) {
				delete[] _buffers[i];
				/**
				 * NOTE: using NUMA allocator leads to a deadlock in the libc. A
				 * simple work-around is to avoid using it.
				 *
				 * numa_free(_buffers[i], _buffer_size * sizeof(uint32_t));
				 */
			}
		}
		delete[] _buffers;
	}
}

void PVParallelView::PVHitGraphDataOMP::omp_ctx_t::clear()
{
	for (uint32_t i = 0; i < _core_num; ++i) {
		memset(_buffers[i], 0, _buffer_size * sizeof(uint32_t));
	}
}

namespace hitgraph_baseline
{
void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t* col_y1, const uint64_t y_min, const int zoom, const double alpha, uint32_t* buffer, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t nbits, size_t size_block_int);
void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t* col_y1, const uint64_t y_min, const int zoom, const double alpha, uint32_t* buffer, int block_count, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t nbits, size_t size_block_int);
void count_y1_sel_omp_sse_v4(const PVRow row_count, const uint32_t* col_y1, const Squey::PVSelection& selection, const uint64_t y_min, const int zoom, const double& alpha, uint32_t* buffer, int block_count, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t nbits, size_t size_block_int);
}

// Off x86 the avx2 unit is not built at all, as there is no wider instruction set to
// build it for. has_avx2() is a compile time false there, so the branches below are
// dead, but they still have to name something : alias the namespace onto the baseline.
#if defined(__x86_64__) || defined(_M_X64)
namespace hitgraph_avx2
{
void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t* col_y1, const uint64_t y_min, const int zoom, const double alpha, uint32_t* buffer, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t nbits, size_t size_block_int);
void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t* col_y1, const uint64_t y_min, const int zoom, const double alpha, uint32_t* buffer, int block_count, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t nbits, size_t size_block_int);
void count_y1_sel_omp_sse_v4(const PVRow row_count, const uint32_t* col_y1, const Squey::PVSelection& selection, const uint64_t y_min, const int zoom, const double& alpha, uint32_t* buffer, int block_count, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t nbits, size_t size_block_int);
}
#else
namespace hitgraph_avx2 = hitgraph_baseline;
#endif


//
// Public interfaces
//

PVParallelView::PVHitGraphDataOMP::PVHitGraphDataOMP(uint32_t nbits, uint32_t nblocks)
    : PVHitGraphDataInterface(nbits, nblocks), _omp_ctx(nblocks * size_block())
{
}

void PVParallelView::PVHitGraphDataOMP::process_all(ProcessParams const& p,
                                                    PVHitGraphBuffer& buf) const
{
	process_all_with(p, buf, PVCore::has_avx2());
}

// The kernels are built twice, once for the baseline and once with -mavx2. Dispatching
// here rather than inside the loops keeps it to one test per call, out of the hot path.
void PVParallelView::PVHitGraphDataOMP::process_all_with(ProcessParams const& p,
                                                         PVHitGraphBuffer& buf,
                                                         bool avx2) const
{
	int nblocks_ = std::min((uint32_t)p.nblocks, nblocks() - p.block_start);
	if (nblocks_ <= 0) {
		return;
	}

	_omp_ctx.clear();

	uint32_t* const buf_block = buf.zoomed_buffer_block(p.block_start, p.alpha);
	if (nblocks_ == 1) {
		if (avx2) {
			hitgraph_avx2::count_y1_omp_sse_v4(p.nrows, p.col_scaled, p.y_min, p.zoom, p.alpha,
			                                   buf_block, _omp_ctx, nbits(), size_block());
		} else {
			hitgraph_baseline::count_y1_omp_sse_v4(p.nrows, p.col_scaled, p.y_min, p.zoom, p.alpha,
			                                       buf_block, _omp_ctx, nbits(), size_block());
		}
	} else {
		if (avx2) {
			hitgraph_avx2::count_y1_omp_sse_v4(p.nrows, p.col_scaled, p.y_min, p.zoom, p.alpha,
			                                   buf_block, nblocks_, _omp_ctx, nbits(), size_block());
		} else {
			hitgraph_baseline::count_y1_omp_sse_v4(p.nrows, p.col_scaled, p.y_min, p.zoom, p.alpha,
			                                       buf_block, nblocks_, _omp_ctx, nbits(), size_block());
		}
	}
}

void PVParallelView::PVHitGraphDataOMP::process_sel(ProcessParams const& p,
                                                    PVHitGraphBuffer& buf,
                                                    Squey::PVSelection const& sel) const
{
	process_sel_with(p, buf, sel, PVCore::has_avx2());
}

void PVParallelView::PVHitGraphDataOMP::process_sel_with(ProcessParams const& p,
                                                         PVHitGraphBuffer& buf,
                                                         Squey::PVSelection const& sel,
                                                         bool avx2) const
{
	int nblocks_ = std::min((uint32_t)p.nblocks, nblocks() - p.block_start);
	if (nblocks_ <= 0) {
		return;
	}

	_omp_ctx.clear();

	uint32_t* const buf_block = buf.zoomed_buffer_block(p.block_start, p.alpha);
	if (avx2) {
		hitgraph_avx2::count_y1_sel_omp_sse_v4(p.nrows, p.col_scaled, sel, p.y_min, p.zoom, p.alpha,
		                                       buf_block, nblocks_, _omp_ctx, nbits(), size_block());
	} else {
		hitgraph_baseline::count_y1_sel_omp_sse_v4(p.nrows, p.col_scaled, sel, p.y_min, p.zoom,
		                                           p.alpha, buf_block, nblocks_, _omp_ctx, nbits(),
		                                           size_block());
	}
}
