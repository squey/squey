
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

/* sequential version using shift'n mask which uses indexed block
 */
void count_y1_seq_v3(const PVRow row_count, const uint32_t *col_y1,
                     const uint64_t y_min, const int zoom,
                     uint32_t *buffer)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_base = y_min >> zoom_shift;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;
		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++buffer[block_idx];
		}
	}
}

void count_y1_sse_v3(const PVRow row_count, const uint32_t *col_y1,
                     const uint64_t y_min, const int zoom,
                     uint32_t *buffer)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

	PVRow packed_row_count = row_count & ~3;

	for(PVRow i = 0; i < packed_row_count; i += 4) {
		const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
		const __m128i block_base_sse = _mm_srli_epi32(y_sse, zoom_shift);

		const __m128i test_res_sse = _mm_cmpeq_epi32(block_base_sse, zoom_base_sse);

		if (_mm_test_all_zeros(test_res_sse, all_zeros_mask_sse)) {
			continue;
		}

		const __m128i block_idx_sse = _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
		                                            idx_mask_sse);

		for (int j = 0; j < 4; ++j) {
			if(_mm_extract_epi32(test_res_sse, j)) {
				++buffer[_mm_extract_epi32(block_idx_sse, j)];
			}
		}
	}

	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++buffer[block_idx];
		}
	}
}

struct omp_sse_v3_ctx_t
{
	omp_sse_v3_ctx_t(uint32_t size = BUFFER_SIZE)
	{
		core_num = PVCore::PVHardwareConcurrency::get_physical_core_number();
		buffers = new uint32_t * [core_num];
		buffer_size = size;

		if (getenv("USE_NUMA") != NULL) {
			use_numa = true;
		} else {
			use_numa = false;
		}

		for(uint32_t i = 0; i < core_num; ++i) {
			if (use_numa) {
				buffers[i] = (uint32_t*)numa_alloc_onnode(size * sizeof(uint32_t), numa_node_of_cpu(i));
			} else {
				posix_memalign((void**)&(buffers[i]), 16, size * sizeof(uint32_t));
			}
			memset(buffers[i], 0, size * sizeof(uint32_t));
		}
	}

	~omp_sse_v3_ctx_t()
	{
		if (buffers) {
			for(uint32_t i = 0; i < core_num; ++i) {
				if (buffers[i]) {
					if (use_numa) {
						numa_free(buffers[i], BUFFER_SIZE * sizeof(uint32_t));
					} else {
						free(buffers[i]);
					}
				}
			}
			delete [] buffers;
		}
	}

	void clear()
	{
		for(uint32_t i = 0; i < core_num; ++i) {
			memset(buffers[i], 0, BUFFER_SIZE * sizeof(uint32_t));
			for(int j = 0; j < BUFFER_SIZE; j+=16) {
				_mm_clflush(&buffers[i][j]);
			}
		}
	}

	uint32_t   buffer_size;
	uint32_t   core_num;
	uint32_t **buffers;
	bool       use_numa;
};

void count_y1_omp_sse_v3(const PVRow row_count, const uint32_t *col_y1,
                         const uint64_t y_min, const int zoom,
                         uint32_t *buffer, omp_sse_v3_ctx_t &ctx)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t idx_mask = (1 << NBITS) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.core_num)
	{
		uint32_t *my_buffer = ctx.buffers[omp_get_thread_num()];

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i block_base_sse = _mm_srli_epi32(y_sse, zoom_shift);

			const __m128i test_res_sse = _mm_cmpeq_epi32(block_base_sse, zoom_base_sse);

			if (_mm_test_all_zeros(test_res_sse, all_zeros_mask_sse)) {
				continue;
			}

			const __m128i block_idx_sse = _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
			                                            idx_mask_sse);

			for (int j = 0; j < 4; ++j) {
				if(_mm_extract_epi32(test_res_sse, j)) {
					++my_buffer[_mm_extract_epi32(block_idx_sse, j)];
				}
			}
		}
	}

	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++buffer[block_idx];
		}
	}

	// final reduction
	for(size_t i = 0; i < ctx.core_num; ++i) {
		size_t packed_size = BUFFER_SIZE & ~3;
		uint32_t *core_buffer = ctx.buffers[i];
		for (size_t j = 0; j < packed_size; j += 4) {
			const __m128i global_sse = _mm_load_si128((const __m128i*) &buffer[j]);
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			const __m128i res_sse = _mm_add_epi32(global_sse, local_sse);
			_mm_store_si128((__m128i*) &buffer[j], res_sse);
		}
	}
}


/*****************************************************************************
 * public API
 *****************************************************************************/

uint32_t *hist_plotted_seq(const uint32_t y_min, const int zoom, uint32_t *buffer,
                           const uint32_t * col, const PVRow row_count)
{
	uint32_t base = y_min >> (32-zoom);
	uint32_t offset = y_min - base;

	count_y1_seq_v3(row_count, col, base, zoom, buffer);

	return buffer + offset;
}

uint32_t *hist_plotted_sse(const uint32_t y_min, const int zoom, uint32_t *buffer,
                           const uint32_t * col, const PVRow row_count)
{
	uint32_t base = y_min >> (32-zoom);
	uint32_t offset = y_min - base;

	count_y1_sse_v3(row_count, col, base, zoom, buffer);

	return buffer + offset;
}

uint32_t *hist_plotted_omp_sse(const uint32_t y_min, const int zoom, uint32_t *buffer,
                               const uint32_t * col, const PVRow row_count,
                               omp_sse_v3_ctx_t &ctx)
{
	uint32_t base = y_min >> (32-zoom);
	uint32_t offset = y_min - base;

	count_y1_omp_sse_v3(row_count, col, base, zoom, buffer, ctx);

	return buffer + offset;
}

/*****************************************************************************
 * main & co
 *****************************************************************************/

typedef enum {
	ARG_COL = 0,
	ARG_MIN,
	ARG_ZOOM,
} EXTRA_ARG;

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
	uint32_t *hist_buffer = new uint32_t [BUFFER_SIZE];

	// sequential
	uint32_t *res_seq = hist_plotted_seq(y_min, zoom, hist_buffer,
	                                     col_a, row_count);
	std::cout << res_seq << std::endl;

	// SSE
	uint32_t *res_sse = hist_plotted_sse(y_min, zoom, hist_buffer,
	                                     col_a, row_count);
	std::cout << res_sse << std::endl;

	// OMP+SSE
	omp_sse_v3_ctx_t omp_sse_ctx;
	omp_sse_ctx.clear();

	uint32_t *res_omp_sse = hist_plotted_omp_sse(y_min, zoom, hist_buffer,
	                                             col_a, row_count, omp_sse_ctx);
	std::cout << res_omp_sse << std::endl;

	return 0;
}
