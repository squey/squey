
#include <pvbase/types.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/picviz_intrin.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/PVZoneProcessing.h>

#include <numa.h>      // numa_*
#include <omp.h>

#include "common.h"

#define NBITS 10
#define BUFFER_SIZE (1<<NBITS)

/*****************************************************************************
 * raw algorithms
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

		for (int j = 0; j < 4; ++j) {
			if(_mm_extract_epi32(res_sse, j)) {
				++buffer[_mm_extract_epi32(off_sse, j)];
			}
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

		for (int j = 0; j < 4; ++j) {
			if(_mm_extract_epi32(res_sse, j)) {
				++buffer[_mm_extract_epi32(off_sse, j)];
			}
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
		_core_num = PVCore::PVHardwareConcurrency::get_physical_core_number();
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
		for(uint32_t i = 0; i < _core_num; ++i) {
			memset(_buffers[i], 0, _buffer_size * sizeof(uint32_t));
			// for(int j = 0; j < _buffer_size; j+=16) {
			// 	_mm_clflush(&buffers[i][j]);
			// }
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

			for (int j = 0; j < 4; ++j) {
				if(_mm_extract_epi32(res_sse, j)) {
					++my_buffer[_mm_extract_epi32(off_sse, j)];
				}
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

			for (int j = 0; j < 4; ++j) {
				if(_mm_extract_epi32(res_sse, j)) {
					++my_buffer[_mm_extract_epi32(off_sse, j)];
				}
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
 * public API
 *****************************************************************************/

uint32_t *hist_plotted_seq(const uint32_t y_min, const int zoom, uint32_t *buffer,
                           const uint32_t * col, const PVRow row_count, int block_count)
{
	// if (block_count == 1) {
	// 	count_y1_seq_v4(row_count, col, y_min, zoom, buffer);
	// } else {
		count_y1_seq_v4(row_count, col, y_min, zoom, buffer, block_count);
	// }

	return buffer;
}

uint32_t *hist_plotted_sse(const uint32_t y_min, const int zoom, uint32_t *buffer,
                           const uint32_t * col, const PVRow row_count, int block_count)
{
	// if (block_count == 1) {
	// 	count_y1_sse_v4(row_count, col, y_min, zoom, buffer);
	// } else {
		count_y1_sse_v4(row_count, col, y_min, zoom, buffer, block_count);
	// }

	return buffer;
}

uint32_t *hist_plotted_omp_sse(const uint32_t y_min, const int zoom, uint32_t *buffer,
                               const uint32_t * col, const PVRow row_count, int block_count,
                               omp_ctx_t &ctx)
{
	// if (block_count == 1) {
	// 	count_y1_omp_sse_v4(row_count, col, y_min, zoom, buffer, ctx);
	// } else {
		count_y1_omp_sse_v4(row_count, col, y_min, zoom, buffer, block_count, ctx);
	// }

	return buffer;
}

/*****************************************************************************
 * main & co
 *****************************************************************************/

typedef enum {
	ARG_COL = 0,
	ARG_MIN,
	ARG_ZOOM,
} EXTRA_ARG;

bool compare(uint32_t *ref, uint32_t *tab, int block_count)
{
	for(int i = 0; i < BUFFER_SIZE * block_count; ++i) {
		if (tab[i] != ref[i]) {
			std::cerr << "differs at " << i
			          << ": " << tab[i] << " instead of " << ref[i]
			          << std::endl;
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv)
{
	set_extra_param(3, "col y_min zoom");

	Picviz::PVPlotted::uint_plotted_table_t plotted;
	PVCol col_count;
	PVRow row_count;

	std::cout << "initialization" << std::endl;
	if (false == create_plotted_table_from_args(plotted, row_count, col_count, argc, argv)) {
		exit(1);
	}

	int pos = extra_param_start_at();
	int col = atoi(argv[pos + ARG_COL]);
	uint32_t y_min = atol(argv[pos + ARG_MIN]);
	int zoom = atol(argv[pos + ARG_ZOOM]);

	if (zoom == 0) {
		zoom = 1;
		std::cout << "INFO: setting zoom to 1 because block algorithm have an exception for zoom == 0"
		          << std::endl;
	}

	PVParallelView::PVZoneProcessing zp(plotted, row_count, col, col + 1);

	const uint32_t *col_a = zp.get_plotted_col_a();
	int block_count = 1;
	int real_buffer_size = BUFFER_SIZE * block_count;

	/* sequential
	 */
	uint32_t *hist_seq = new uint32_t [real_buffer_size];
	BENCH_START(seq);
	uint32_t *res_seq = hist_plotted_seq(y_min, zoom, hist_seq,
	                                     col_a, row_count, block_count);
	BENCH_END(seq, "hist_seq", row_count, sizeof(uint32_t), BUFFER_SIZE, sizeof(uint32_t));

	/* SSE
	 */
	uint32_t *hist_sse = new uint32_t [real_buffer_size];
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

	return 0;
}
