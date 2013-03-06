
#include <pvbase/types.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/picviz_intrin.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/PVHitGraphData.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoneProcessing.h>

#include <numa.h>      // numa_*
#include <omp.h>

#include "common.h"

#define NBITS 10
#define BUFFER_SIZE (1<<NBITS)

/*****************************************************************************
 * raw algorithms without selection
 *****************************************************************************/

/* optimized version for 1 block
 */
void count_y1_seq_v4(const PVRow row_count, const uint32_t *col_y1,
                     const uint64_t y_min, const int zoom,
                     uint32_t *buffer)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int p = base - base_y;
		if (p != 0) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++buffer[idx];
	}
}

/* version for N blocks
 */
void count_y1_seq_v4(const PVRow row_count, const uint32_t *col_y1,
                     const uint64_t y_min, const int zoom,
                     uint32_t *buffer, int block_count)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++buffer[(p<<NBITS) + idx];
	}
}

/* optimized version for 1 block
 */
void count_y1_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                     const uint64_t y_min, const int zoom,
                     uint32_t *buffer)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	PVRow packed_row_count = row_count & ~3;

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
			++buffer[_mm_extract_epi32(off_sse, 0)];
		}
		if(_mm_extract_epi32(res_sse, 1)) {
			++buffer[_mm_extract_epi32(off_sse, 1)];
		}
		if(_mm_extract_epi32(res_sse, 2)) {
			++buffer[_mm_extract_epi32(off_sse, 2)];
		}
		if(_mm_extract_epi32(res_sse, 3)) {
			++buffer[_mm_extract_epi32(off_sse, 3)];
		}
	}

	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t base = y >> zoom_shift;
		int p = base - base_y;
		if (p != 0) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++buffer[idx];
	}
}

/* version for N blocks
 */
void count_y1_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                     const uint64_t y_min, const int zoom,
                     uint32_t *buffer, int block_count)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	PVRow packed_row_count = row_count & ~3;

	for(PVRow i = 0; i < packed_row_count; i += 4) {
		const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
		const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
		const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

		const __m128i res_sse = _mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_set1_epi32(0)),
		                                         _mm_cmplt_epi32(p_sse, _mm_set1_epi32(block_count)));

		if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
			continue;
		}

		const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
		                                      _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
		                                                    idx_mask_sse));

		if(_mm_extract_epi32(res_sse, 0)) {
			++buffer[_mm_extract_epi32(off_sse, 0)];
		}
		if(_mm_extract_epi32(res_sse, 1)) {
			++buffer[_mm_extract_epi32(off_sse, 1)];
		}
		if(_mm_extract_epi32(res_sse, 2)) {
			++buffer[_mm_extract_epi32(off_sse, 2)];
		}
		if(_mm_extract_epi32(res_sse, 3)) {
			++buffer[_mm_extract_epi32(off_sse, 3)];
		}
	}

	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++buffer[(p<<NBITS) + idx];
	}
}

struct omp_ctx_t
{
	omp_ctx_t(uint32_t size = BUFFER_SIZE)
	{
		_core_num = PVCore::PVHardwareConcurrency::get_logical_core_number();
		_buffers = new uint32_t * [_core_num];
		_buffer_size = size;

		for(uint32_t i = 0; i < _core_num; ++i) {
			_buffers[i] = (uint32_t*)numa_alloc_onnode(_buffer_size * sizeof(uint32_t),
				                                          numa_node_of_cpu(i));
			memset(_buffers[i], 0, size * sizeof(uint32_t));
		}
	}

	~omp_ctx_t()
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

	void clear()
	{
#pragma omp parallel for
		for(uint32_t i = 0; i < _core_num; ++i) {
			uint32_t *buffer = _buffers[i];
			memset(buffer, 0, _buffer_size * sizeof(uint32_t));
			for(size_t j = 0; j < _buffer_size; j+=16) {
				_mm_clflush(buffer + j);
			}
		}
	}

	int get_core_num() const
	{
		return _core_num;
	}

	uint32_t *get_core_buffer(int i)
	{
		return _buffers[i];
	}

	uint32_t   _buffer_size;
	uint32_t   _core_num;
	int        _block_count;
	uint32_t **_buffers;
};

/* optimized version for 1 block
 */
void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, omp_ctx_t &ctx)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
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
	size_t packed_size = BUFFER_SIZE & ~3;
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

/* version for N blocks
 */
void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, int block_count, omp_ctx_t &ctx)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
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

			const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
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
		++first_buffer[(p<<NBITS) + idx];
	}

	// final reduction
	size_t packed_size = (BUFFER_SIZE * block_count) & ~3;
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

/*****************************************************************************
 * raw algorithms with selection
 *****************************************************************************/

void count_y1_sel_seq_v4(const PVRow row_count, const uint32_t *col_y1,
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, int block_count)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;

	for(size_t i = 0; i < row_count; ++i) {
		if(!selection.get_line(i)) {
			continue;
		}
		const uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p >= block_count)) {
			continue;
		}
		const uint32_t idx = (y >> idx_shift) & idx_mask;
		++buffer[(p<<NBITS) + idx];
	}
}

void count_y1_sel_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, int block_count)
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

	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	PVRow packed_row_count = row_count & ~3;

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

		const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
		                                      _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
		                                                    idx_mask_sse));

		if(_mm_extract_epi32(res_sse, 0)) {
			++buffer[_mm_extract_epi32(off_sse, 0)];
		}
		if(_mm_extract_epi32(res_sse, 1)) {
			++buffer[_mm_extract_epi32(off_sse, 1)];
		}
		if(_mm_extract_epi32(res_sse, 2)) {
			++buffer[_mm_extract_epi32(off_sse, 2)];
		}
		if(_mm_extract_epi32(res_sse, 3)) {
			++buffer[_mm_extract_epi32(off_sse, 3)];
		}
	}

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
		++buffer[(p<<NBITS) + idx];
	}
}

void count_y1_sel_omp_v4(const PVRow row_count, const uint32_t *col_y1,
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, int block_count, omp_ctx_t &ctx)
{

	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;

#pragma omp parallel
	{
		uint32_t *my_buffer = ctx.get_core_buffer(omp_get_thread_num());

#pragma omp for
		for(size_t i = 0; i < row_count; ++i) {
			if(!selection.get_line(i)) {
				continue;
			}
			const uint32_t y = col_y1[i];
			const int32_t base = y >> zoom_shift;
			int p = base - base_y;
			if ((p < 0) || (p >= block_count)) {
				continue;
			}
			const uint32_t idx = (y >> idx_shift) & idx_mask;
			++my_buffer[(p<<NBITS) + idx];
		}
	}

	// final reduction
	size_t packed_size = (BUFFER_SIZE * block_count) & ~3;
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

void count_y1_sel_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                             const Picviz::PVSelection &selection,
                             const uint64_t y_min, const int zoom,
                             uint32_t *buffer, int block_count, omp_ctx_t &ctx)
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

	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
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

			const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
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
		++first_buffer[(p<<NBITS) + idx];
	}

	// final reduction
	size_t packed_size = (BUFFER_SIZE * block_count) & ~3;
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

void count_y1_sel_seq_v5(const PVRow row_count, const uint32_t *col_y1,
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, int block_count)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const uint32_t zoom_shift = 32 - zoom;
	const int32_t base_y = y_min >> zoom_shift;

	selection.visit_selected_lines([=](const PVRow r)
	                               {
		                               const uint32_t y = col_y1[r];
		                               const int32_t base = y >> zoom_shift;
		                               int p = base - base_y;
		                               if ((p >= 0) && (p < block_count)) {
			                               const uint32_t idx = (y >> idx_shift) & idx_mask;
			                               ++buffer[(p<<NBITS) + idx];
		                               }
	                               }, row_count);
}

void count_y1_sel_packed_v5(const PVRow row_count, const uint32_t *col_y1,
                            const Picviz::PVSelection &selection,
                            const uint64_t y_min, const int zoom,
                            uint32_t *buffer, int block_count)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	selection.visit_selected_lines_packed<4>([=](const PVRow *packed)
	                                         {
		                                         PVRow v[4];
		                                         for(int i = 0; i < 4; ++i) {
			                                         v[i] = col_y1[packed[i]];
		                                         }
		                                         const __m128i y_sse = _mm_load_si128((const __m128i*) &v);
		                                         const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
		                                         const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

		                                         const __m128i res_sse = _mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_set1_epi32(0)),
			                                                                          _mm_cmplt_epi32(p_sse, _mm_set1_epi32(block_count)));

		                                         if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
			                                         return;
		                                         }
		                                         const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
			                                                                       _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
				                                                                             idx_mask_sse));

		                                         if(_mm_extract_epi32(res_sse, 0)) {
			                                         ++buffer[_mm_extract_epi32(off_sse, 0)];
		                                         }
		                                         if(_mm_extract_epi32(res_sse, 1)) {
			                                         ++buffer[_mm_extract_epi32(off_sse, 1)];
		                                         }
		                                         if(_mm_extract_epi32(res_sse, 2)) {
			                                         ++buffer[_mm_extract_epi32(off_sse, 2)];
		                                         }
		                                         if(_mm_extract_epi32(res_sse, 3)) {
			                                         ++buffer[_mm_extract_epi32(off_sse, 3)];
		                                         }
	                                         },
	                                         [=](const PVRow r)
	                                         {
		                                         const uint32_t y = col_y1[r];
		                                         const int32_t base = y >> zoom_shift;
		                                         int p = base - base_y;
		                                         if ((p >= 0) && (p < block_count)) {
			                                         const uint32_t idx = (y >> idx_shift) & idx_mask;
			                                         ++buffer[(p<<NBITS) + idx];
		                                         }
	                                         }, row_count);
}

/*****************************************************************************
 * public API
 *****************************************************************************/

uint32_t *hist_plotted_seq(const uint32_t y_min, const int zoom, uint32_t *buffer,
                           const uint32_t * col, const PVRow row_count, int block_count)
{
	if (block_count == 1) {
		count_y1_seq_v4(row_count, col, y_min, zoom, buffer);
	} else {
		count_y1_seq_v4(row_count, col, y_min, zoom, buffer, block_count);
	}

	return buffer;
}

uint32_t *hist_plotted_sse(const uint32_t y_min, const int zoom, uint32_t *buffer,
                           const uint32_t * col, const PVRow row_count, int block_count)
{
	if (block_count == 1) {
		count_y1_sse_v4(row_count, col, y_min, zoom, buffer);
	} else {
		count_y1_sse_v4(row_count, col, y_min, zoom, buffer, block_count);
	}

	return buffer;
}

uint32_t *hist_plotted_omp_sse(const uint32_t y_min, const int zoom, uint32_t *buffer,
                               const uint32_t * col, const PVRow row_count, int block_count,
                               omp_ctx_t &ctx)
{
	if (block_count == 1) {
		count_y1_omp_sse_v4(row_count, col, y_min, zoom, buffer, ctx);
	} else {
		count_y1_omp_sse_v4(row_count, col, y_min, zoom, buffer, block_count, ctx);
	}

	return buffer;
}

/*****************************************************************************
 * main & co
 *****************************************************************************/

typedef enum {
	ARG_COL = 0,
	ARG_BLOCKS,
	ARG_MIN,
	ARG_ZOOM,
} EXTRA_ARG;

bool compare(uint32_t *ref, uint32_t *tab, int block_count)
{
	bool ret = true;
	for(int i = 0; i < BUFFER_SIZE * block_count; ++i) {
		if (tab[i] != ref[i]) {
			std::cerr << "differs at " << i
			          << ": " << tab[i] << " instead of " << ref[i]
			          << std::endl;
			ret = false;
		}
	}
	return ret;
}

void test_no_sel(const size_t real_buffer_size,
                 const uint32_t y_min, const int zoom, PVParallelView::PVZoneTree const& zt,
                 const uint32_t* col_a, const size_t row_count,
                 const int block_count)
{
	/* sequential
	 */
	uint32_t *hist_seq = new uint32_t [real_buffer_size];
	memset(hist_seq, 0, real_buffer_size * sizeof(uint32_t));
	BENCH_START(seq);
	uint32_t *res_seq = hist_plotted_seq(y_min, zoom, hist_seq,
	                                     col_a, row_count, block_count);
	BENCH_END(seq, "hist_seq", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));

	/* SSE
	 */
	uint32_t *hist_sse = new uint32_t [real_buffer_size];
	memset(hist_sse, 0, real_buffer_size * sizeof(uint32_t));
	BENCH_START(sse);
	uint32_t *res_sse = hist_plotted_sse(y_min, zoom, hist_sse,
	                                     col_a, row_count, block_count);
	BENCH_END(sse, "hist_sse", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(res_seq, res_sse, block_count)) {
		std::cout << "ok" << std::endl;
	}

	/* OMP+SSE
	 */
	uint32_t *hist_omp_sse = new uint32_t [real_buffer_size];
	memset(hist_omp_sse, 0, real_buffer_size * sizeof(uint32_t));
	omp_ctx_t omp_sse_ctx(real_buffer_size);
	omp_sse_ctx.clear();

	BENCH_START(omp_sse);
	uint32_t *res_omp_sse = hist_plotted_omp_sse(y_min, zoom, hist_omp_sse,
	                                             col_a, row_count, block_count, omp_sse_ctx);
	BENCH_END(omp_sse, "hist_omp_sse", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(res_seq, res_omp_sse, block_count)) {
		std::cout << "ok" << std::endl;
	}

	/* Library code
	 */
	/*PVParallelView::PVHitGraphData lib_omp;
	BENCH_START(lib);
	lib_omp.process_all(PVParallelView::PVHitGraphData::ProcessParams(zt, col_a, row_count, y_min, zoom, 0, block_count));
	BENCH_END(lib, "library-code", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	if (compare(res_seq, lib_omp.buffer_all().buffer(), block_count)) {
		std::cout << "ok" << std::endl;
	}*/

	delete [] hist_seq;
	delete [] hist_sse;
	delete [] hist_omp_sse;
}

void test_sel(const size_t real_buffer_size,
              const uint32_t y_min, const int zoom,
              const Picviz::PVSelection &selection,
              const uint32_t* col_a, const size_t row_count,
              const int block_count)
{
	/* basic one
	 */
	uint32_t *hist_sel_seq_v4 = new uint32_t [real_buffer_size];
	memset(hist_sel_seq_v4, 0, real_buffer_size * sizeof(uint32_t));
	BENCH_START(seq_v4);
	count_y1_sel_seq_v4(row_count, col_a, selection,
	                    y_min, zoom, hist_sel_seq_v4,
	                    block_count);
	BENCH_END(seq_v4, "hist_sel_seq_v4", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));

	uint32_t *hist_sel_sse_v4 = new uint32_t [real_buffer_size];
	memset(hist_sel_sse_v4, 0, real_buffer_size * sizeof(uint32_t));
	BENCH_START(sse_v4);
	count_y1_sel_sse_v4(row_count, col_a, selection,
	                    y_min, zoom, hist_sel_sse_v4,
	                    block_count);
	BENCH_END(sse_v4, "hist_sel_sse_v4", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(hist_sel_seq_v4, hist_sel_sse_v4, block_count)) {
		std::cout << "ok" << std::endl;
	}

	/* OMP's basic one
	 */
	uint32_t *hist_sel_omp_v4 = new uint32_t [real_buffer_size];
	memset(hist_sel_omp_v4, 0, real_buffer_size * sizeof(uint32_t));
	omp_ctx_t omp_ctx(real_buffer_size);
	omp_ctx.clear();

	BENCH_START(omp_v4);
	count_y1_sel_omp_v4(row_count, col_a, selection,
	                    y_min, zoom, hist_sel_omp_v4,
	                    block_count, omp_ctx);
	BENCH_END(omp_v4, "hist_sel_omp_v4", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(hist_sel_seq_v4, hist_sel_omp_v4, block_count)) {
		std::cout << "ok" << std::endl;
	}

	/* OMP'n SSE version
	 */
	uint32_t *hist_sel_omp_sse_v4 = new uint32_t [real_buffer_size];
	memset(hist_sel_omp_sse_v4, 0, real_buffer_size * sizeof(uint32_t));
	omp_ctx.clear();

	BENCH_START(omp_sse_v4);
	count_y1_sel_omp_sse_v4(row_count, col_a, selection,
	                    y_min, zoom, hist_sel_omp_sse_v4,
	                    block_count, omp_ctx);
	BENCH_END(omp_sse_v4, "hist_sel_omp_sse_v4", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(hist_sel_seq_v4, hist_sel_omp_sse_v4, block_count)) {
		std::cout << "ok" << std::endl;
	}

	/* using visit_selected_lines
	 */
	uint32_t *hist_sel_seq_v5 = new uint32_t [real_buffer_size];
	memset(hist_sel_seq_v5, 0, real_buffer_size * sizeof(uint32_t));
	BENCH_START(seq_v5);
	count_y1_sel_seq_v5(row_count, col_a, selection,
	                    y_min, zoom, hist_sel_seq_v5,
	                    block_count);
	BENCH_END(seq_v5, "hist_sel_seq_v5", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(hist_sel_seq_v4, hist_sel_seq_v5, block_count)) {
		std::cout << "ok" << std::endl;
	}

	/* using visit_selected_lines_packed
	 */
	uint32_t *hist_sel_packed_v5 = new uint32_t [real_buffer_size];
	memset(hist_sel_packed_v5, 0, real_buffer_size * sizeof(uint32_t));
	BENCH_START(packed_v5);
	count_y1_sel_packed_v5(row_count, col_a, selection,
	                       y_min, zoom, hist_sel_packed_v5,
	                       block_count);
	BENCH_END(packed_v5, "hist_sel_packed_v5", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));
	std::cout << "compare to ref: ";
	if (compare(hist_sel_seq_v4, hist_sel_packed_v5, block_count)) {
		std::cout << "ok" << std::endl;
	}
}

int main(int argc, char **argv)
{
	set_extra_param(4, "col block_count y_min zoom");

	Picviz::PVPlotted::uint_plotted_table_t plotted;
	PVCol col_count;
	PVRow row_count;

	std::cout << "initialization" << std::endl;
	if (false == create_plotted_table_from_args(plotted, row_count, col_count, argc, argv)) {
		exit(1);
	}

	int pos = extra_param_start_at();
	int col = atoi(argv[pos + ARG_COL]);
	int block_count = atoi(argv[pos + ARG_BLOCKS]);
	uint32_t y_min = atol(argv[pos + ARG_MIN]);
	int zoom = atol(argv[pos + ARG_ZOOM]);

	if (zoom == 0) {
		zoom = 1;
		std::cout << "INFO: setting zoom to 1 because block algorithm have an exception for zoom == 0"
		          << std::endl;
	}

	PVParallelView::PVZoneProcessing zp(plotted, row_count, col, col + 1);
	PVParallelView::PVZoneTree& zt = *new PVParallelView::PVZoneTree();
	zt.process(zp);

	const uint32_t *col_a = zp.get_plotted_col_a();
	int real_buffer_size = BUFFER_SIZE * block_count;

	test_no_sel(real_buffer_size, y_min, zoom, zt, col_a, row_count, block_count);

	Picviz::PVSelection sel;

	std::cout << "select_all()" << std::endl;
	sel.select_all();
	test_sel(real_buffer_size, y_min, zoom, sel, col_a, row_count, block_count);

	std::cout << "select_none()" << std::endl;
	sel.select_none();
	test_sel(real_buffer_size, y_min, zoom, sel, col_a, row_count, block_count);

	std::cout << "select_random()" << std::endl;
	sel.select_random();
	test_sel(real_buffer_size, y_min, zoom, sel, col_a, row_count, block_count);

	return 0;
}
