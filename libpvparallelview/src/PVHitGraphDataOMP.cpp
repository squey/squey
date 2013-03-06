#include <pvkernel/core/PVHardwareConcurrency.h>
#include <picviz/PVSelection.h>
#include <pvparallelview/PVHitGraphDataOMP.h>

#include <numa.h>
#include <omp.h>

// Constants used by the OMP code
//#define NBITS (PVParallelView::PVHitGraphCommon::NBITS) // Number of bits used by the reduction
//#define BUFFER_SIZE (PVParallelView::PVHitGraphBuffer::SIZE_BLOCK) // Number of integers in one block

//
// OMP specific context structure
//

PVParallelView::PVHitGraphDataOMP::omp_ctx_t::omp_ctx_t(uint32_t size)
{
	// "size" is the number of integers of a thread-specific buffer
	// (thus = nblocks * size_int_block)
	_core_num = PVCore::PVHardwareConcurrency::get_physical_core_number();
	_buffers = new uint32_t * [_core_num];
	_buffer_size = size;

	for(uint32_t i = 0; i < _core_num; ++i) {
		_buffers[i] = (uint32_t*)numa_alloc_onnode(_buffer_size * sizeof(uint32_t),
		                                           numa_node_of_cpu(i));
		memset(_buffers[i], 0, size * sizeof(uint32_t));
	}
}

PVParallelView::PVHitGraphDataOMP::omp_ctx_t::~omp_ctx_t()
{
	if (_buffers) {
		for(uint32_t i = 0; i < _core_num; ++i) {
			if (_buffers[i]) {
				numa_free(_buffers[i],
				          _buffer_size * sizeof(uint32_t));
			}
		}
		delete [] _buffers;
	}
}

void PVParallelView::PVHitGraphDataOMP::omp_ctx_t::clear()
{
	for (uint32_t i = 0; i < _core_num; ++i) {
		memset(_buffers[i], 0, _buffer_size * sizeof(uint32_t));
	}
}

//
// OMP algorithms
//

// Optimised version for 1 block, no-selection
static void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, PVParallelView::PVHitGraphDataOMP::omp_ctx_t &ctx,
						 size_t nbits, size_t size_block_int)
{
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t idx_mask = (1 << nbits) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			const __m128i res_sse = _mm_cmpeq_epi32(p_sse,
			                                        _mm_set1_epi32(0));

			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

			const __m128i off_sse = _mm_and_si128(_mm_srli_epi32(y_sse,
			                                                     idx_shift),
			                                      idx_mask_sse);

			if(_mm_extract_epi32(res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(off_sse, 0)];
			}
			if(_mm_extract_epi32(res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(off_sse, 1)];
			}
			if(_mm_extract_epi32(res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(off_sse, 2)];
			}
			if(_mm_extract_epi32(res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(off_sse, 3)];
			}
		}
	}

	// last values
	uint32_t *first_buffer = ctx.get_core_buffer(0);
	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int p = base - base_y;
		if (p != 0) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++first_buffer[idx];
	}

	// final reduction
	size_t packed_size = size_block_int & ~3;
	for (size_t j = 0; j < packed_size; j += 4) {
		__m128i global_sse = _mm_setzero_si128();

		for(int i = 0; i < ctx.get_core_num(); ++i) {
			uint32_t *core_buffer = ctx.get_core_buffer(i);
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			global_sse = _mm_add_epi32(global_sse, local_sse);
		}
		_mm_store_si128((__m128i*) &buffer[j], global_sse);
	}
}

// Version for N blocks (N >= 2), no-selection
static void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, int block_count, PVParallelView::PVHitGraphDataOMP::omp_ctx_t &ctx,
						 size_t nbits, size_t size_block_int)
{
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t idx_mask = (1 << nbits) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			const __m128i res_sse = _mm_andnot_si128(_mm_cmplt_epi32(p_sse,
			                                                         _mm_set1_epi32(0)),
			                                         _mm_cmplt_epi32(p_sse,
			                                                         _mm_set1_epi32(block_count)));

			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

			const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, nbits),
			                                      _mm_and_si128(_mm_srli_epi32(y_sse,
			                                                                   idx_shift),
			                                                    idx_mask_sse));

			if(_mm_extract_epi32(res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(off_sse, 0)];
			}
			if(_mm_extract_epi32(res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(off_sse, 1)];
			}
			if(_mm_extract_epi32(res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(off_sse, 2)];
			}
			if(_mm_extract_epi32(res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(off_sse, 3)];
			}
		}
	}

	// last values
	uint32_t *first_buffer = ctx.get_core_buffer(0);
	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++first_buffer[(p<<nbits) + idx];
	}

	// final reduction
	size_t packed_size = (size_block_int * block_count) & ~3;
	for (size_t j = 0; j < packed_size; j += 4) {
		__m128i global_sse = _mm_setzero_si128();

		for(int i = 0; i < ctx.get_core_num(); ++i) {
			uint32_t *core_buffer = ctx.get_core_buffer(i);
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			global_sse = _mm_add_epi32(global_sse, local_sse);
		}
		_mm_store_si128((__m128i*) &buffer[j], global_sse);
	}
}

// Version for N blocks (N>=1), with selection
void count_y1_sel_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                             const Picviz::PVSelection &selection,
                             const uint64_t y_min, const int zoom,
                             uint32_t *buffer, int block_count, PVParallelView::PVHitGraphDataOMP::omp_ctx_t &ctx,
                             size_t nbits, size_t size_block_int)
{
	static DECLARE_ALIGN(16)__m128i mask[16] = { _mm_set_epi32( 0,  0,  0,  0),
	                                             _mm_set_epi32( 0,  0,  0, -1),
	                                             _mm_set_epi32( 0,  0, -1,  0),
	                                             _mm_set_epi32( 0,  0, -1, -1),
	                                             _mm_set_epi32( 0, -1,  0,  0),
	                                             _mm_set_epi32( 0, -1,  0, -1),
	                                             _mm_set_epi32( 0, -1, -1,  0),
	                                             _mm_set_epi32( 0, -1, -1, -1),
	                                             _mm_set_epi32(-1,  0,  0,  0),
	                                             _mm_set_epi32(-1,  0,  0, -1),
	                                             _mm_set_epi32(-1,  0, -1,  0),
	                                             _mm_set_epi32(-1,  0, -1, -1),
	                                             _mm_set_epi32(-1, -1,  0,  0),
	                                             _mm_set_epi32(-1, -1,  0, -1),
	                                             _mm_set_epi32(-1, -1, -1,  0),
	                                             _mm_set_epi32(-1, -1, -1, -1) };

	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t idx_mask = (1 << nbits) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for

		for(PVRow i = 0; i < packed_row_count; i += 4) {
			uint32_t f = selection.get_lines_fast(i, 4);
			if (f == 0) {
				continue;
			}
			const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			const __m128i res_sse = _mm_and_si128(_mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_set1_epi32(0)),
			                                                       _mm_cmplt_epi32(p_sse, _mm_set1_epi32(block_count))),
			                                      mask[f]);
			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

			const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, nbits),
			                                      _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
			                                                    idx_mask_sse));

			if(_mm_extract_epi32(res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(off_sse, 0)];
			}
			if(_mm_extract_epi32(res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(off_sse, 1)];
			}
			if(_mm_extract_epi32(res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(off_sse, 2)];
			}
			if(_mm_extract_epi32(res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(off_sse, 3)];
			}
		}
	}

	uint32_t *first_buffer = ctx.get_core_buffer(0);
	for(PVRow i = packed_row_count; i < row_count; ++i) {
		if (!selection.get_line_fast(i)) {
			continue;
		}
		const uint32_t y = col_y1[i];
		const uint32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++first_buffer[(p<<nbits) + idx];
	}

	// final reduction
	size_t packed_size = (size_block_int * block_count) & ~3;
	for (size_t j = 0; j < packed_size; j += 4) {
		__m128i global_sse = _mm_setzero_si128();

		for(int i = 0; i < ctx.get_core_num(); ++i) {
			uint32_t *core_buffer = ctx.get_core_buffer(i);
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			global_sse = _mm_add_epi32(global_sse, local_sse);
		}
		_mm_store_si128((__m128i*) &buffer[j], global_sse);
	}
}

//
// Public interfaces
//

PVParallelView::PVHitGraphDataOMP::PVHitGraphDataOMP(uint32_t nbits, uint32_t nblocks):
	PVHitGraphDataInterface(nbits, nblocks),
	_omp_ctx(nblocks*size_block())
{
}
void PVParallelView::PVHitGraphDataOMP::process_bg(ProcessParams const& p)
{
	int nblocks_ = std::min((uint32_t) p.nblocks, nblocks() - p.block_start);
	if (nblocks_ <= 0) {
		return;
	}

	_omp_ctx.clear();

	uint32_t* const buf_block = buffer_all().buffer_block(p.block_start);
	if (nblocks_ == 1) {
		count_y1_omp_sse_v4(p.nrows, p.col_plotted, p.y_min, p.zoom, buf_block, _omp_ctx, nbits(), size_block());
	}
	else {
		count_y1_omp_sse_v4(p.nrows, p.col_plotted, p.y_min, p.zoom, buf_block, nblocks_, _omp_ctx, nbits(), size_block());
	}
}

void PVParallelView::PVHitGraphDataOMP::process_sel(ProcessParams const& p, Picviz::PVSelection const& sel)
{
	int nblocks_ = std::min((uint32_t) p.nblocks, nblocks() - p.block_start);
	if (nblocks_ <= 0) {
		return;
	}

	_omp_ctx.clear();

	uint32_t* const buf_block = buffer_all().buffer_block(p.block_start);
	count_y1_sel_omp_sse_v4(p.nrows, p.col_plotted, sel, p.y_min, p.zoom, buf_block, nblocks_, _omp_ctx, nbits(), size_block());
}
