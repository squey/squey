#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/PVHitGraphDataOMP.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

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

static void merge_ctx_buffers(uint32_t* __restrict buffer, PVParallelView::PVHitGraphDataOMP::omp_ctx_t& ctx, size_t size_int_merge)
{
	size_t packed_size = size_int_merge & ~3;
	size_t j;
	for (j = 0; j < packed_size; j += 4) {
		__m128i global_sse = _mm_setzero_si128();

		for(int i = 0; i < ctx.get_core_num(); i++) {
			uint32_t *core_buffer = ctx.get_core_buffer(i);
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			global_sse = _mm_add_epi32(global_sse, local_sse);
		}
		_mm_storeu_si128((__m128i*) &buffer[j], global_sse);
	}
	for (; j < size_int_merge; j++) {
		uint32_t v = 0;
		for (int i = 0; i < ctx.get_core_num(); i++) {
			uint32_t *core_buffer = ctx.get_core_buffer(i);
			v += core_buffer[j];
		}
		buffer[j] = v;
	}
}

//
// OMP algorithms
//

// Optimised version for 1 block, no-selection
static void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                                const uint64_t y_min, const int zoom,
                                const double alpha,
                                uint32_t *buffer,
                                PVParallelView::PVHitGraphDataOMP::omp_ctx_t &ctx,
                                size_t nbits, size_t size_block_int)
{
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const int32_t base_y = (uint64_t)(y_min) >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const __m128i y_min_ref_sse = _mm_set1_epi32(y_min_ref);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			__m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			/* p = base - base_ref
			 * if (p < 0)
			 *   continue
			 */
			const __m128i res_sse = _mm_cmpeq_epi32(p_sse,
			                                        _mm_set1_epi32(0));

			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

			const __m128i off_sse = PVParallelView::PVHitGraphSSEHelpers::buffer_offset_from_y_sse(y_sse, p_sse, y_min_ref_sse, alpha_sse, zoom_mask_sse, idx_shift, zoom_shift, nbits);

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
		// AG: a 64-bit integer is used for 'y', because if zoom_shift is 32, then y >> 32 wouldn't be 0 !
		uint64_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int32_t p = base - base_y;
		if (p != 0) {
			continue;
		}
		y = (y - y_min_ref) * alpha;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++first_buffer[idx];
	}

	// final reduction
	size_t merge_size = size_block_int * alpha;
	merge_ctx_buffers(buffer, ctx, merge_size);
}

// Version for N blocks (N >= 2), no-selection
static void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                                const uint64_t y_min, const int zoom,
                                const double alpha,
                                uint32_t *buffer, int block_count,
                                PVParallelView::PVHitGraphDataOMP::omp_ctx_t &ctx,
                                size_t nbits, size_t size_block_int)
{
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const int32_t base_y = (uint64_t)(y_min) >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const __m128i y_min_ref_sse = _mm_set1_epi32(y_min_ref);


#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			__m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			__m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			/* p = base - base_ref
			 * if ((p < 0) || (p >= block_count))
			 *   continue
			 */
			const __m128i res_sse = _mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_setzero_si128()),
			                                         _mm_cmplt_epi32(p_sse, _mm_set1_epi32(block_count)));

			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

			const __m128i off_sse = PVParallelView::PVHitGraphSSEHelpers::buffer_offset_from_y_sse(y_sse, p_sse, y_min_ref_sse, alpha_sse, zoom_mask_sse, idx_shift, zoom_shift, nbits);

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
		// AG: a 64-bit integer is used for 'y', because if zoom_shift is 32, then y >> 32 wouldn't be 0 !
		uint64_t y = col_y1[i];
		const int32_t base = (uint64_t)(y) >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		y = (y-y_min_ref)*alpha;
		p = (uint64_t)(y) >> zoom_shift;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++first_buffer[(p<<nbits) | idx];
	}

	// final reduction
	size_t merge_size = ((size_t)(size_block_int * alpha)) * block_count;
	merge_ctx_buffers(buffer, ctx, merge_size);
}

// Version for N blocks (N>=1), with selection
void count_y1_sel_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                             const Picviz::PVSelection &selection,
                             const uint64_t y_min, const int zoom,
                             const double &alpha,
                             uint32_t *buffer, int block_count,
                             PVParallelView::PVHitGraphDataOMP::omp_ctx_t &ctx,
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
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const uint32_t base_y = (uint64_t)(y_min) >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const __m128i y_min_ref_sse = _mm_set1_epi32(y_min_ref);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.get_core_num())
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			uint32_t f = selection.get_lines_fast(i, 4);
			if (f == 0) {
				continue;
			}

			__m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			__m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			/* p = base - base_ref
			 * if (!sel.is_set(y) && (p < 0) || (p >= block_count))
			 *   continue
			 */
			const __m128i res_sse = _mm_and_si128(_mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_setzero_si128()),
			                                                       _mm_cmplt_epi32(p_sse, _mm_set1_epi32(block_count))),
			                                      mask[f]);
			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

			const __m128i off_sse = PVParallelView::PVHitGraphSSEHelpers::buffer_offset_from_y_sse(y_sse, p_sse, y_min_ref_sse, alpha_sse, zoom_mask_sse, idx_shift, zoom_shift, nbits);

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
		// AG: a 64-bit integer is used for 'y', because if zoom_shift is 32, then y >> 32 wouldn't be 0 !
		uint64_t y = col_y1[i];
		const uint32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		y = (y-y_min_ref)*alpha;
		p = y >> zoom_shift;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++first_buffer[(p<<nbits) + idx];
	}

	// final reduction
	size_t merge_size = size_block_int * block_count * alpha;
	merge_ctx_buffers(buffer, ctx, merge_size);
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

	uint32_t* const buf_block = buffer_all().zoomed_buffer_block(p.block_start, p.alpha);
	if (nblocks_ == 1) {
		count_y1_omp_sse_v4(p.nrows, p.col_plotted, p.y_min, p.zoom, p.alpha, buf_block, _omp_ctx, nbits(), size_block());
	}
	else {
		count_y1_omp_sse_v4(p.nrows, p.col_plotted, p.y_min, p.zoom, p.alpha, buf_block, nblocks_, _omp_ctx, nbits(), size_block());
	}
}

void PVParallelView::PVHitGraphDataOMP::process_sel(ProcessParams const& p, Picviz::PVSelection const& sel)
{
	int nblocks_ = std::min((uint32_t) p.nblocks, nblocks() - p.block_start);
	if (nblocks_ <= 0) {
		return;
	}

	_omp_ctx.clear();

	uint32_t* const buf_block = buffer_sel().zoomed_buffer_block(p.block_start, p.alpha);
	count_y1_sel_omp_sse_v4(p.nrows, p.col_plotted, sel, p.y_min, p.zoom, p.alpha, buf_block, nblocks_, _omp_ctx, nbits(), size_block());
}
