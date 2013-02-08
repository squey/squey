
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <picviz/PVPlotted.h>

#include <pvparallelview/PVZoneProcessing.h>

#include "common.h"

#include <iostream>
#include <x86intrin.h>
#include <stdlib.h> // posix_memalign

#include <omp.h>
#include <numa.h>

/* TODO:
 * - coder l'algo séquentiel
 * - vectoriser (Cf le code du quadtree)
 * - paralléliser (omp et tbb)
 * - ajouter l'utilisation d'une sélection
 */

/*****************************************************************************
 * sequential algos
 *****************************************************************************/

void count_y1_seq_v1(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     uint32_t *buffer, const size_t buffer_size)
{
	const uint64_t dy = y_max - y_min;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		if ((y < y_min) || (y > y_max))
			continue;
		const uint64_t idx = ((y - y_min) * buffer_size) / dy;
		++buffer[idx];
	}
}

/* sequential version using shift'n mask but which keeps relative indexes
 */
void count_y1_seq_v2(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int shift = (32 - 10) - zoom;
	const uint32_t mask = (1 << 10) - 1;
	const uint32_t y_m = y_min;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		if ((y < y_min) || (y > y_max))
			continue;
		const uint32_t idx = ((y - y_m) >> shift) & mask;
		++buffer[idx];
	}
}


/* sequential version using shift'n mask which uses indexed block
 */
void count_y1_seq_v3(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - 10) - zoom;
	const uint32_t idx_mask = (1 << 10) - 1;
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

#ifdef __AVX__
inline __m256i mm256_srli_epi32(const __m256i v, const int count)
{
	const __m128i v0 = _mm256_extractf128_si256(v, 0);
	const __m128i v1 = _mm256_extractf128_si256(v, 1);

	const __m128i v0s = _mm_srli_epi32(v0, count);
	const __m128i v1s = _mm_srli_epi32(v1, count);

	return _mm256_insertf128_si256(_mm256_castsi128_si256(v0s), v1s, 1);
}

void count_y1_avx_v3(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - 10) - zoom;
	const int zoom_shift = 32 - zoom;
	constexpr uint32_t idx_mask = (1 << 10) - 1;
	const uint32_t zoom_base = y_min >> zoom_shift;

	const uint32_t row_count_avx = (row_count/8)*8;
	const __m256i avx_idx_mask = _mm256_set1_epi32(idx_mask);
	const __m256i avx_zoom_base = _mm256_set1_epi32(zoom_base);
	const __m256i avx_ff = _mm256_set1_epi32(0xFFFFFFFF);

	size_t i;
	for(i = 0; i < row_count_avx; i += 8) {
		const __m256i avx_y = _mm256_load_si256((__m256i const*) &col_y1[i]);
		const __m256i avx_block_base = mm256_srli_epi32(avx_y, zoom_shift);
		const __m256i avx_block_idx = reinterpret_cast<__m256i>(_mm256_and_ps(reinterpret_cast<__m256>(mm256_srli_epi32(avx_y, idx_shift)),
		                                                                      reinterpret_cast<__m256>(avx_idx_mask)));

		const __m256i avx_cmp = reinterpret_cast<__m256i>(_mm256_cmp_ps(reinterpret_cast<__m256>(avx_block_base), reinterpret_cast<__m256>(avx_zoom_base), _CMP_EQ_OQ));
		if (_mm256_testz_si256(avx_cmp, avx_ff)) {
			// They are all false
			continue;
		}

		for (int j = 0; j < 8; j++) {
			if (_mm256_extract_epi32(avx_cmp, j)) {
				buffer[_mm256_extract_epi32(avx_block_idx, j)]++;
			}
		}
	}
	for (; i < row_count; i++) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;
		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++buffer[block_idx];
		}
	}
}
#endif

/*****************************************************************************
 * SSE algos
 *****************************************************************************/

/* TODO: change it to use SSE intrinsics (and libdivide, see in pvkernel/core)
 */
void count_y1_sse_v1(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     uint32_t *buffer, const size_t buffer_size)
{
	const uint64_t dy = y_max - y_min;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		// if ((y < y_min) || (y > y_max))
		// 	continue;
		const uint64_t idx = ((y - y_min) * buffer_size) / dy;
		++buffer[idx];
	}
}

void count_y1_sse_v3(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - 10) - zoom;
	const uint32_t idx_mask = (1 << 10) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

	size_t packed_row_count = row_count & ~3;

	for(size_t i = 0; i < packed_row_count; i += 4) {
		const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
		const __m128i block_base_sse = _mm_srli_epi32(y_sse, zoom_shift);

		const __m128i test_res_sse = _mm_cmpeq_epi32(block_base_sse, zoom_base_sse);

		if (_mm_test_all_zeros(test_res_sse, all_zeros_mask_sse)) {
			continue;
		}

		const __m128i block_idx_sse = _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
		                                            idx_mask_sse);

		if(_mm_extract_epi32(test_res_sse, 0)) {
			++buffer[_mm_extract_epi32(block_idx_sse, 0)];
		}
		if(_mm_extract_epi32(test_res_sse, 1)) {
			++buffer[_mm_extract_epi32(block_idx_sse, 1)];
		}
		if(_mm_extract_epi32(test_res_sse, 2)) {
			++buffer[_mm_extract_epi32(block_idx_sse, 2)];
		}
		if(_mm_extract_epi32(test_res_sse, 3)) {
			++buffer[_mm_extract_epi32(block_idx_sse, 3)];
		}
	}

	for(size_t i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++buffer[block_idx];
		}
	}
}

/*****************************************************************************
 * OMP + SSE algos
 *****************************************************************************/

struct omp_sse_v3_ctx_t
{
	omp_sse_v3_ctx_t(uint32_t size)
	{
		core_num = PVCore::PVHardwareConcurrency::get_logical_core_number();
		buffers = new uint32_t * [core_num];
		buffer_size = size;

		if (getenv("USE_NUMA") != NULL) {
			use_numa = true;
			std::cout << "using numa grouped reduction buffer" << std::endl;
		} else {
			use_numa = false;
			std::cout << "using normally allocated reduction buffer" << std::endl;
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
						numa_free(buffers[i], buffer_size * sizeof(uint32_t));
					} else {
						free(buffers[i]);
					}
				}
			}
			delete buffers;
		}
	}

	uint32_t   buffer_size;
	uint32_t   core_num;
	uint32_t **buffers;
	bool       use_numa;
};

void count_y1_omp_sse_v3(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const uint64_t y_max, const int zoom,
                         uint32_t *buffer, const size_t buffer_size, omp_sse_v3_ctx_t &ctx)
{
	const int idx_shift = (32 - 10) - zoom;
	const uint32_t idx_mask = (1 << 10) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

	size_t packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.core_num)
	{
		uint32_t *my_buffer = ctx.buffers[omp_get_thread_num()];

#pragma omp for
		for(size_t i = 0; i < packed_row_count; i += 4) {
			const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i block_base_sse = _mm_srli_epi32(y_sse, zoom_shift);

			const __m128i test_res_sse = _mm_cmpeq_epi32(block_base_sse, zoom_base_sse);

			if (_mm_test_all_zeros(test_res_sse, all_zeros_mask_sse)) {
				continue;
			}

			const __m128i block_idx_sse = _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
			                                            idx_mask_sse);

			if(_mm_extract_epi32(test_res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 0)];
			}
			if(_mm_extract_epi32(test_res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 1)];
			}
			if(_mm_extract_epi32(test_res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 2)];
			}
			if(_mm_extract_epi32(test_res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 3)];
			}
		}
	}

	for(size_t i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++buffer[block_idx];
		}
	}

	// final reduction
	for(size_t i = 0; i < ctx.core_num; ++i) {
		size_t packed_size = buffer_size & ~3;
		uint32_t *core_buffer = ctx.buffers[i];
		for (size_t j = 0; j < packed_size; j += 4) {
			const __m128i global_sse = _mm_load_si128((const __m128i*) &buffer[j]);
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			const __m128i res_sse = _mm_add_epi32(global_sse, local_sse);
			_mm_store_si128((__m128i*) &buffer[j], res_sse);
		}
	}
}

void count_y1_omp_sse_v3_2(const PVRow row_count, const uint32_t *col_y1, const uint32_t *col_y2,
                           const Picviz::PVSelection &selection,
                           const uint64_t y_min, const uint64_t y_max, const int zoom,
                           uint32_t *buffer, const size_t buffer_size, omp_sse_v3_ctx_t &ctx)
{
	const int idx_shift = (32 - 10) - zoom;
	const uint32_t idx_mask = (1 << 10) - 1;
	const __m128i idx_mask_sse = _mm_set1_epi32(idx_mask);
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

	size_t packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.core_num)
	{
		uint32_t *my_buffer = ctx.buffers[omp_get_thread_num()];

#pragma omp for
		for(size_t i = 0; i < packed_row_count; i += 4) {
			const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i block_base_sse = _mm_srli_epi32(y_sse, zoom_shift);

			const __m128i test_res_sse = _mm_cmpeq_epi32(block_base_sse, zoom_base_sse);

			if (_mm_test_all_zeros(test_res_sse, all_zeros_mask_sse)) {
				continue;
			}

			const __m128i block_idx_sse = _mm_and_si128(_mm_srli_epi32(y_sse, idx_shift),
			                                            idx_mask_sse);

			if(_mm_extract_epi32(test_res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 0)];
			}
			if(_mm_extract_epi32(test_res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 1)];
			}
			if(_mm_extract_epi32(test_res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 2)];
			}
			if(_mm_extract_epi32(test_res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(block_idx_sse, 3)];
			}
		}
	}

	// last values
	uint32_t *first_buffer = ctx.buffers[0];
	for(size_t i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = (y >> idx_shift) & idx_mask;
			++first_buffer[block_idx];
		}
	}

	// final reduction
	size_t packed_size = buffer_size & ~3;
	for (size_t j = 0; j < packed_size; j += 4) {
		__m128i global_sse = _mm_setzero_si128();

		for(size_t i = 0; i < ctx.core_num; ++i) {
			uint32_t *core_buffer = ctx.buffers[i];
			const __m128i local_sse = _mm_load_si128((const __m128i*) &core_buffer[j]);
			global_sse = _mm_add_epi32(global_sse, local_sse);
		}
		_mm_store_si128((__m128i*) &buffer[j], global_sse);
	}
}

/*****************************************************************************
 * some macros to reduce code duplication
 *****************************************************************************/

#define DEF_TEST(ALGO,...)	  \
	uint32_t *buffer_ ## ALGO; \
	posix_memalign((void**)&buffer_ ## ALGO, 16, buffer_size * sizeof(uint32_t)); \
	memset(buffer_ ## ALGO, 0, sizeof(uint32_t) * buffer_size); \
	{ \
		BENCH_START(ALGO); \
		count_y1_ ## ALGO (row_count, col_y1, col_y2, selection, y_min, y_max, zoom, buffer_ ## ALGO, buffer_size, ##__VA_ARGS__); \
		BENCH_END(ALGO, #ALGO " count", row_count, sizeof(uint32_t), buffer_size, sizeof(uint32_t)); \
	}

#define CMP_TEST(ALGO,ALGO_REF)	  \
	if (memcmp(buffer_ ## ALGO, buffer_ ## ALGO_REF, buffer_size * sizeof(uint32_t)) != 0) { \
		std::cerr << "algorithms count_y1_" #ALGO " and count_y1_" #ALGO_REF " differ" << std::endl; \
		for(int i = 0; i < buffer_size; ++i) { \
			if (buffer_ ## ALGO [i] != buffer_ ## ALGO_REF [i]) { \
				std::cerr << "  at " << i << ": ref == " << buffer_ ## ALGO_REF [i] << " buffer == " << buffer_ ## ALGO [i] << std::endl; \
			} \
		} \
	} else { \
		std::cout << "count_y1_" #ALGO " is ok" << std::endl; \
	}

/*****************************************************************************
 * main & cie
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

	std::cout << "loading data" << std::endl;
	if (false == create_plotted_table_from_args(plotted, row_count, col_count, argc, argv)) {
		exit(1);
	}

	int pos = extra_param_start_at();

	int col = atoi(argv[pos + ARG_COL]);
	uint64_t y_min = atol(argv[pos + ARG_MIN]);
	int zoom = atol(argv[pos + ARG_ZOOM]);

	if (zoom == 0) {
		zoom = 1;
		std::cout << "INFO: setting zoom to 1 because block algorithm have an exception for zoom == 0"
		          << std::endl;
	}
	uint64_t y_max = y_min + (1UL << (32 - zoom));

	PVParallelView::PVZoneProcessing zp(plotted, row_count, col, col + 1);

	const uint32_t *col_y1 = zp.get_plotted_col_a();
	const uint32_t *col_y2 = zp.get_plotted_col_b();

	int buffer_size = 1024;

	Picviz::PVSelection selection;

	std::cout << "start test" << std::endl;

	DEF_TEST(seq_v1);

	DEF_TEST(seq_v2);
	CMP_TEST(seq_v2, seq_v1);

	DEF_TEST(seq_v3);
	CMP_TEST(seq_v3, seq_v1);

	DEF_TEST(sse_v3);
	CMP_TEST(sse_v3, seq_v1);

#ifdef __AVX__
	DEF_TEST(avx_v3);
	CMP_TEST(avx_v3, seq_v1);
#endif

	omp_sse_v3_ctx_t omp_sse_v3_ctx(buffer_size);

	DEF_TEST(omp_sse_v3, omp_sse_v3_ctx);
	CMP_TEST(omp_sse_v3, seq_v1);

	omp_sse_v3_ctx_t omp_sse_v32_ctx(buffer_size);
	DEF_TEST(omp_sse_v3_2, omp_sse_v32_ctx);
	CMP_TEST(omp_sse_v3_2, seq_v1);

	return 0;
}
