
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/picviz_intrin.h>

#include <pvparallelview/PVHitGraphData.h>
#include <pvparallelview/PVZoneTree.h>

#include "common.h"

#include <iostream>
#include <x86intrin.h> // SSE
#include <stdlib.h>    // posix_memalign

#include <omp.h>       // OMP
#include <numa.h>      // numa_*
#include <numaif.h>    // mbind
#include <sys/mman.h>  // madvise

static bool verbose = false;

/* TODO:
 * - paralléliser (omp et tbb)
 * - vectoriser seq_v2
 * - paralléliser seq_v2
 * - ajouter l'utilisation d'une sélection
 */

#define NBITS 10
#define V4_N 1
#define BUFFER_SIZE (1<<NBITS)

/*****************************************************************************
 * sequential algos
 *****************************************************************************/

void count_y1_seq_v1(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const uint64_t dy = y_max - y_min;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		if ((y < y_min) || (y > y_max))
			continue;
		const uint64_t idx = ((y - y_min) * alpha * buffer_size) / dy;
		++buffer[idx];
	}
}

/* sequential version using shift'n mask but which keeps relative indexes
 */
void count_y1_seq_v2(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int shift = (32 - NBITS) - zoom;
	const uint32_t mask = (1 << NBITS) - 1;
	const uint32_t y_m = y_min;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		if ((y < y_min) || (y > y_max))
			continue;
		const uint32_t idx = ((uint32_t)((y - y_m) * alpha) >> shift) & mask;
		++buffer[idx];
	}
}


/* sequential version using shift'n mask which uses indexed block
 */
void count_y1_seq_v3(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) - 1;
	const int32_t zoom_base = y_min >> zoom_shift;

	for(size_t i = 0; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const int32_t block_base = y >> zoom_shift;
		if (block_base != zoom_base) {
			continue;
		}
		const uint32_t block_idx = ((uint32_t)((y & zoom_mask) * alpha)) >> idx_shift;
		++buffer[block_idx];
	}
}


/* sequential version using shift'n mask and N block
 */
void count_y1_seq_v4(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) -1;
	const int32_t base_y = y_min >> zoom_shift;
	const uint32_t y_min_ref = (uint32_t)base_y << zoom_shift;

	for(size_t i = 0; i < row_count; ++i) {
		uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int32_t p = base - base_y;
		if (p < 0 || p >= V4_N) {
			continue;
		}
		y = (y-y_min_ref)*alpha;
		p = y >> zoom_shift;
		const uint32_t idx = ((uint32_t)(y & zoom_mask)) >> idx_shift;
		++buffer[(p<<NBITS) + idx];
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

void count_y1_avx_v3(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const int zoom_shift = 32 - zoom;
	constexpr uint32_t idx_mask = (1 << NBITS) - 1;
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

		if (_mm256_extract_epi32(avx_cmp, 0)) {
			buffer[_mm256_extract_epi32(avx_block_idx, 0)]++;
		}
		if (_mm256_extract_epi32(avx_cmp, 1)) {
			buffer[_mm256_extract_epi32(avx_block_idx, 1)]++;
		}
		if (_mm256_extract_epi32(avx_cmp, 2)) {
			buffer[_mm256_extract_epi32(avx_block_idx, 2)]++;
		}
		if (_mm256_extract_epi32(avx_cmp, 3)) {
			buffer[_mm256_extract_epi32(avx_block_idx, 3)]++;
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

void count_y1_sse_v3(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) -1;
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

	PVRow packed_row_count = row_count & ~3;

	for(PVRow i = 0; i < packed_row_count; i += 4) {
		const __m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
		const __m128i block_base_sse = _mm_srli_epi32(y_sse, zoom_shift);

		const __m128i test_res_sse = _mm_cmpeq_epi32(block_base_sse, zoom_base_sse);

		if (_mm_test_all_zeros(test_res_sse, all_zeros_mask_sse)) {
			continue;
		}

#ifdef __AVX__
		const __m128i tmp1_sse = _mm_and_si128(y_sse, zoom_mask_sse);
		const __m256d tmp1_avx = _mm256_cvtepi32_pd (tmp1_sse);
		const __m256d tmp2_avx = _mm256_mul_pd(tmp1_avx, alpha_sse);
		const __m128i tmp2_sse = _mm256_cvttpd_epi32(tmp2_avx);
		const __m128i block_idx_sse = _mm_srli_epi32(tmp2_sse, idx_shift);

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
#elif defined __SSE2__
		const __m128i tmp_sse = _mm_and_si128(y_sse, zoom_mask_sse);
		const __m128i tmp1_lo_sse = tmp_sse;
		const __m128i tmp1_hi_sse = _mm_shuffle_epi32(tmp_sse, _MM_SHUFFLE(0, 0, 3, 2));
		const __m128d tmp2_lo_sse = _mm_cvtepi32_pd(tmp1_lo_sse);
		const __m128d tmp2_hi_sse = _mm_cvtepi32_pd(tmp1_hi_sse);
		const __m128d tmp3_lo_sse = _mm_mul_pd(tmp2_lo_sse, alpha_sse);
		const __m128d tmp3_hi_sse = _mm_mul_pd(tmp2_hi_sse, alpha_sse);
		const __m128i tmp4_lo_sse = _mm_cvttpd_epi32(tmp3_lo_sse);
		const __m128i tmp4_hi_sse = _mm_cvttpd_epi32(tmp3_hi_sse);
		const __m128i tmp5_lo_sse = _mm_srli_epi32(tmp4_lo_sse, idx_shift);
		const __m128i tmp5_hi_sse = _mm_srli_epi32(tmp4_hi_sse, idx_shift);

		if(_mm_extract_epi32(test_res_sse, 0)) {
			++buffer[_mm_extract_epi32(tmp5_lo_sse, 0)];
		}
		if(_mm_extract_epi32(test_res_sse, 1)) {
			++buffer[_mm_extract_epi32(tmp5_lo_sse, 1)];
		}
		if(_mm_extract_epi32(test_res_sse, 2)) {
			++buffer[_mm_extract_epi32(tmp5_hi_sse, 0)];
		}
		if(_mm_extract_epi32(test_res_sse, 3)) {
			++buffer[_mm_extract_epi32(tmp5_hi_sse, 1)];
		}
#else
#error you need at least SSE2 intrinsics
#endif
	}

	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = ((uint32_t)((y & zoom_mask) * alpha)) >> idx_shift;
			++buffer[block_idx];
		}
	}
}

void count_y1_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                     const Picviz::PVSelection &selection,
                     const uint64_t y_min, const uint64_t y_max, const int zoom,
                     double alpha,
                     uint32_t *buffer, const size_t buffer_size)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) -1;
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const uint32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint32_t)base_y << zoom_shift;
	const __m128i y_min_ref_sse = _mm_set1_epi32(y_min_ref);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

	PVRow packed_row_count = row_count & ~3;

	for(PVRow i = 0; i < packed_row_count; i += 4) {
		__m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
		const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
		__m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

		const __m128i res_sse = _mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_set1_epi32(0)),
		                                         _mm_cmplt_epi32(p_sse, _mm_set1_epi32(V4_N)));

		if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
			continue;
		}

#ifdef __AVX__
		y_sse = _mm_sub_epi32(y_sse, y_min_ref_sse);
		const __m256d tmp1_avx = _mm256_cvtepi32_pd (y_sse);
		const __m256d tmp2_avx = _mm256_mul_pd(tmp1_avx, alpha_sse);
		y_sse = _mm256_cvttpd_epi32(tmp2_avx);
		p_sse = _mm_srli_epi32(y_sse, zoom_shift);

		const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
		                                      _mm_srli_epi32(_mm_and_si128(y_sse,
		                                                                   zoom_mask_sse),
		                                                     idx_shift));
#elif defined __SSE2__
		y_sse = _mm_sub_epi32(y_sse, y_min_ref_sse);
		const __m128i tmp1_lo_sse = _mm_shuffle_epi32(y_sse, _MM_SHUFFLE(0, 0, 2, 0));
		const __m128i tmp1_hi_sse = _mm_shuffle_epi32(y_sse, _MM_SHUFFLE(0, 0, 3, 1));
		const __m128d tmp2_lo_sse = _mm_cvtepi32_pd(tmp1_lo_sse);
		const __m128d tmp2_hi_sse = _mm_cvtepi32_pd(tmp1_hi_sse);
		const __m128d tmp3_lo_sse = _mm_mul_pd(tmp2_lo_sse, alpha_sse);
		const __m128d tmp3_hi_sse = _mm_mul_pd(tmp2_hi_sse, alpha_sse);
		const __m128i tmp4_lo_sse = _mm_cvttpd_epi32(tmp3_lo_sse);
		const __m128i tmp4_hi_sse = _mm_cvttpd_epi32(tmp3_hi_sse);

		y_sse = _mm_unpacklo_epi32(tmp4_lo_sse, tmp4_hi_sse);

		p_sse = _mm_srli_epi32(y_sse, zoom_shift);

		const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
		                                      _mm_srli_epi32(_mm_and_si128(y_sse,
		                                                                   zoom_mask_sse),
		                                                     idx_shift));
#else
#error you need at least SSE2 intrinsics
#endif

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
		if ((p < 0) || (p>=V4_N)) {
			continue;
		}
		const uint32_t idx = ((uint32_t)((y & zoom_mask) * alpha)) >> idx_shift;
		++buffer[(p<<NBITS) + idx];
	}
}

/*****************************************************************************
 * OMP + SSE algos
 *****************************************************************************/

struct omp_sse_v3_ctx_t
{
	omp_sse_v3_ctx_t(uint32_t size)
	{
		core_num = PVCore::PVHardwareConcurrency::get_physical_core_number();
		buffers = new uint32_t * [core_num];
		buffer_size = size;

		// if (verbose) {
		// 	std::cout << "allocating " << core_num << " reduction buffers" << std::endl;
		// }
		if (getenv("USE_NUMA") != NULL) {
			use_numa = true;
			// if (verbose) {
			// 	std::cout << "using numa grouped reduction buffer" << std::endl;
			// }
		} else {
			use_numa = false;
			// if (verbose) {
			// 	std::cout << "using normally allocated reduction buffer" << std::endl;
			// }
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
			delete [] buffers;
		}
	}

	void clear()
	{
		for(uint32_t i = 0; i < core_num; ++i) {
			memset(buffers[i], 0, buffer_size * sizeof(uint32_t));
			for(size_t j = 0; j < buffer_size; j+=16) {
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
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const uint64_t y_max, const int zoom,
                         double alpha,
                         uint32_t *buffer, const size_t buffer_size, omp_sse_v3_ctx_t &ctx)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) -1;
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

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

#ifdef __AVX__
			const __m128i tmp1_sse = _mm_and_si128(y_sse, zoom_mask_sse);
			const __m256d tmp1_avx = _mm256_cvtepi32_pd (tmp1_sse);
			const __m256d tmp2_avx = _mm256_mul_pd(tmp1_avx, alpha_sse);
			const __m128i tmp2_sse = _mm256_cvttpd_epi32(tmp2_avx);
			const __m128i block_idx_sse = _mm_srli_epi32(tmp2_sse, idx_shift);

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
#elif defined __SSE2__
			const __m128i tmp_sse = _mm_and_si128(y_sse, zoom_mask_sse);
			const __m128i tmp1_lo_sse = tmp_sse;
			const __m128i tmp1_hi_sse = _mm_shuffle_epi32(tmp_sse, _MM_SHUFFLE(0, 0, 3, 2));
			const __m128d tmp2_lo_sse = _mm_cvtepi32_pd(tmp1_lo_sse);
			const __m128d tmp2_hi_sse = _mm_cvtepi32_pd(tmp1_hi_sse);
			const __m128d tmp3_lo_sse = _mm_mul_pd(tmp2_lo_sse, alpha_sse);
			const __m128d tmp3_hi_sse = _mm_mul_pd(tmp2_hi_sse, alpha_sse);
			const __m128i tmp4_lo_sse = _mm_cvttpd_epi32(tmp3_lo_sse);
			const __m128i tmp4_hi_sse = _mm_cvttpd_epi32(tmp3_hi_sse);
			const __m128i tmp5_lo_sse = _mm_srli_epi32(tmp4_lo_sse, idx_shift);
			const __m128i tmp5_hi_sse = _mm_srli_epi32(tmp4_hi_sse, idx_shift);

			if(_mm_extract_epi32(test_res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(tmp5_lo_sse, 0)];
			}
			if(_mm_extract_epi32(test_res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(tmp5_lo_sse, 1)];
			}
			if(_mm_extract_epi32(test_res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(tmp5_hi_sse, 0)];
			}
			if(_mm_extract_epi32(test_res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(tmp5_hi_sse, 1)];
			}
#else
#error you need at least SSE2 intrinsics
#endif
		}
	}

	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = ((uint32_t)((y & zoom_mask) * alpha)) >> idx_shift;
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

/* This version inverts the 2 loops of the final reduction to avoid loading ant storing at each
 * iteration.
 */
void count_y1_omp_sse_v3_2(const PVRow row_count, const uint32_t *col_y1,
                           const Picviz::PVSelection &selection,
                           const uint64_t y_min, const uint64_t y_max, const int zoom,
                           double alpha,
                           uint32_t *buffer, const size_t buffer_size, omp_sse_v3_ctx_t &ctx)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) -1;
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const uint32_t zoom_base = y_min >> zoom_shift;
	const __m128i zoom_base_sse = _mm_set1_epi32(zoom_base);
	const __m128i all_zeros_mask_sse = _mm_set1_epi32(-1);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

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

#ifdef __AVX__
			const __m128i tmp1_sse = _mm_and_si128(y_sse, zoom_mask_sse);
			const __m256d tmp1_avx = _mm256_cvtepi32_pd (tmp1_sse);
			const __m256d tmp2_avx = _mm256_mul_pd(tmp1_avx, alpha_sse);
			const __m128i tmp2_sse = _mm256_cvttpd_epi32(tmp2_avx);
			const __m128i block_idx_sse = _mm_srli_epi32(tmp2_sse, idx_shift);

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
#elif defined __SSE2__
			const __m128i tmp_sse = _mm_and_si128(y_sse, zoom_mask_sse);
			const __m128i tmp1_lo_sse = tmp_sse;
			const __m128i tmp1_hi_sse = _mm_shuffle_epi32(tmp_sse, _MM_SHUFFLE(0, 0, 3, 2));
			const __m128d tmp2_lo_sse = _mm_cvtepi32_pd(tmp1_lo_sse);
			const __m128d tmp2_hi_sse = _mm_cvtepi32_pd(tmp1_hi_sse);
			const __m128d tmp3_lo_sse = _mm_mul_pd(tmp2_lo_sse, alpha_sse);
			const __m128d tmp3_hi_sse = _mm_mul_pd(tmp2_hi_sse, alpha_sse);
			const __m128i tmp4_lo_sse = _mm_cvttpd_epi32(tmp3_lo_sse);
			const __m128i tmp4_hi_sse = _mm_cvttpd_epi32(tmp3_hi_sse);
			const __m128i tmp5_lo_sse = _mm_srli_epi32(tmp4_lo_sse, idx_shift);
			const __m128i tmp5_hi_sse = _mm_srli_epi32(tmp4_hi_sse, idx_shift);

			if(_mm_extract_epi32(test_res_sse, 0)) {
				++my_buffer[_mm_extract_epi32(tmp5_lo_sse, 0)];
			}
			if(_mm_extract_epi32(test_res_sse, 1)) {
				++my_buffer[_mm_extract_epi32(tmp5_lo_sse, 1)];
			}
			if(_mm_extract_epi32(test_res_sse, 2)) {
				++my_buffer[_mm_extract_epi32(tmp5_hi_sse, 0)];
			}
			if(_mm_extract_epi32(test_res_sse, 3)) {
				++my_buffer[_mm_extract_epi32(tmp5_hi_sse, 1)];
			}
#else
#error you need at least SSE2 intrinsics
#endif
		}
	}

	// last values
	uint32_t *first_buffer = ctx.buffers[0];
	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const uint32_t block_base = y >> zoom_shift;

		if (block_base == zoom_base) {
			const uint32_t block_idx = ((uint32_t)((y & zoom_mask) * alpha)) >> idx_shift;
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


void count_y1_omp_sse_v4(const PVRow row_count, const uint32_t *col_y1,
                         const Picviz::PVSelection &selection,
                         const uint64_t y_min, const uint64_t y_max, const int zoom,
                         double alpha,
                         uint32_t *buffer, const size_t buffer_size, omp_sse_v3_ctx_t &ctx)
{
	const int idx_shift = (32 - NBITS) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = (1 << zoom_shift) -1;
	const __m128i zoom_mask_sse = _mm_set1_epi32(zoom_mask);
	const int32_t base_y = y_min >> zoom_shift;
	const __m128i base_y_sse = _mm_set1_epi32(base_y);

	const uint32_t y_min_ref = (uint32_t)base_y << zoom_shift;
	const __m128i y_min_ref_sse = _mm_set1_epi32(y_min_ref);

#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(alpha);
#elif defined __SSE2__
	const __m128d alpha_sse = _mm_set1_pd(alpha);
#else
#error you need at least SSE2 intrinsics
#endif

	PVRow packed_row_count = row_count & ~3;

#pragma omp parallel num_threads(ctx.core_num)
	{
		uint32_t *my_buffer = ctx.buffers[omp_get_thread_num()];

#pragma omp for
		for(PVRow i = 0; i < packed_row_count; i += 4) {
			__m128i y_sse = _mm_load_si128((const __m128i*) &col_y1[i]);
			const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
			__m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

			const __m128i res_sse = _mm_andnot_si128(_mm_cmplt_epi32(p_sse, _mm_set1_epi32(0)),
			                                         _mm_cmplt_epi32(p_sse, _mm_set1_epi32(V4_N)));

			if (_mm_test_all_zeros(res_sse, _mm_set1_epi32(-1))) {
				continue;
			}

#ifdef __AVX__
			y_sse = _mm_sub_epi32(y_sse, y_min_ref_sse);
			const __m256d tmp1_avx = _mm256_cvtepi32_pd (y_sse);
			const __m256d tmp2_avx = _mm256_mul_pd(tmp1_avx, alpha_sse);
			y_sse = _mm256_cvttpd_epi32(tmp2_avx);
			p_sse = _mm_srli_epi32(y_sse, zoom_shift);

			const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
			                                      _mm_srli_epi32(_mm_and_si128(y_sse,
			                                                                   zoom_mask_sse),
			                                                     idx_shift));
#elif defined __SSE2__
			y_sse = _mm_sub_epi32(y_sse, y_min_ref_sse);
			const __m128i tmp1_lo_sse = _mm_shuffle_epi32(y_sse, _MM_SHUFFLE(0, 0, 2, 0));
			const __m128i tmp1_hi_sse = _mm_shuffle_epi32(y_sse, _MM_SHUFFLE(0, 0, 3, 1));
			const __m128d tmp2_lo_sse = _mm_cvtepi32_pd(tmp1_lo_sse);
			const __m128d tmp2_hi_sse = _mm_cvtepi32_pd(tmp1_hi_sse);
			const __m128d tmp3_lo_sse = _mm_mul_pd(tmp2_lo_sse, alpha_sse);
			const __m128d tmp3_hi_sse = _mm_mul_pd(tmp2_hi_sse, alpha_sse);
			const __m128i tmp4_lo_sse = _mm_cvttpd_epi32(tmp3_lo_sse);
			const __m128i tmp4_hi_sse = _mm_cvttpd_epi32(tmp3_hi_sse);

			y_sse = _mm_unpacklo_epi32(tmp4_lo_sse, tmp4_hi_sse);

			p_sse = _mm_srli_epi32(y_sse, zoom_shift);

			const __m128i off_sse = _mm_add_epi32(_mm_slli_epi32(p_sse, NBITS),
			                                      _mm_srli_epi32(_mm_and_si128(y_sse,
			                                                                   zoom_mask_sse),
			                                                     idx_shift));
#else
#error you need at least SSE2 intrinsics
#endif

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
	uint32_t *first_buffer = ctx.buffers[0];
	for(PVRow i = packed_row_count; i < row_count; ++i) {
		const uint32_t y = col_y1[i];
		const int32_t base = y >> zoom_shift;
		int p = base - base_y;
		if ((p < 0) || (p>=V4_N)) {
			continue;
		}
		const uint32_t idx = ((uint32_t)((y & zoom_mask) * alpha)) >> idx_shift;
		++first_buffer[(p<<NBITS) + idx];
	}

	// final reduction
	size_t packed_size = (buffer_size * V4_N) & ~3;
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

void count_y1_lib(const PVRow row_count, const uint32_t *col_y1,
                  const Picviz::PVSelection &selection,
                  const uint64_t y_min, const uint64_t y_max, const int zoom,
                  double alpha,
                  uint32_t *buffer, const size_t buffer_size, omp_sse_v3_ctx_t &ctx,
                  PVParallelView::PVHitGraphData &lib_omp,
                  PVParallelView::PVZoneTree &zt)
{

	lib_omp.process_bg(PVParallelView::PVHitGraphData::ProcessParams(zt, col_y1, row_count, y_min, zoom, alpha, 0, V4_N));
	memcpy(buffer, lib_omp.buffer_all().buffer(), lib_omp.buffer_all().size_bytes());
}

/*****************************************************************************
 * some code to reduce code duplication
 *****************************************************************************/

template <typename RunFn, typename CleanFn>
inline void bench_median(const char *text, const RunFn& run, const CleanFn &clean,
                         size_t ele_num, size_t ele_size, int iter_num = 10)
{
	std::vector<double> measures;
	measures.reserve(iter_num);
	double mean = 0;

	for(int i = 0; i < iter_num; ++i) {
		clean();
		BENCH_START(aa);
		run();
		BENCH_STOP(aa);
		mean += measures[i] = BENCH_END_TIME(aa);
	}
	mean /= iter_num;

	std::sort(measures.begin(), measures.end());

	double median;
	int mid = iter_num / 2;
	if (iter_num & 1) {
		median = measures[mid];
	} else {
		median = 0.5 * (measures[mid] + measures[mid+1]);
	}

	size_t mean_bw = (double)(ele_num * ele_size) / mean;
	mean_bw >>= 20;

	size_t med_bw = (double)(ele_num * ele_size) / median;
	med_bw >>= 20;

	printf("  test:%-16s | mean:%7.3f bw:%5lu | median:%7.3f bw:%5lu\n",
	       text, 1000. * mean, mean_bw, 1000. * median, med_bw);
}

#define DEF_TEST_SEQ(ALGO,DATA,...)	  \
	uint32_t *buffer_ ## ALGO; \
	posix_memalign((void**)&buffer_ ## ALGO, 16, buffer_size * V4_N * sizeof(uint32_t)); \
	memset(buffer_ ## ALGO, 0, buffer_size * V4_N * sizeof(uint32_t)); \
	bench_median(#ALGO, \
	             [&]() { \
		             count_y1_ ## ALGO (row_count, DATA, selection, y_min, y_max, zoom, alpha, buffer_ ## ALGO, buffer_size, ##__VA_ARGS__); \
	             }, \
	             [&]() { \
		             memset(buffer_ ## ALGO, 0, sizeof(uint32_t) * V4_N * buffer_size); \
	             }, \
	             row_count, sizeof(uint32_t));

#define DEF_TEST_OMP(ALGO, DATA,...)	  \
	uint32_t *buffer_ ## ALGO; \
	posix_memalign((void**)&buffer_ ## ALGO, 16, buffer_size * V4_N * sizeof(uint32_t)); \
	memset(buffer_ ## ALGO, 0, buffer_size * V4_N * sizeof(uint32_t)); \
	omp_sse_v3_ctx_t ALGO ## _ctx(buffer_size * V4_N); \
	bench_median(#ALGO, \
	             [&]() { \
		             count_y1_ ## ALGO (row_count, DATA, selection, y_min, y_max, zoom, alpha, buffer_ ## ALGO, buffer_size, ALGO ## _ctx, ##__VA_ARGS__); \
	             }, \
	             [&]() { \
		             memset(buffer_ ## ALGO, 0, sizeof(uint32_t) * V4_N * buffer_size); \
		             ALGO ## _ctx.clear(); \
	             }, \
	             row_count, sizeof(uint32_t));

#define CMP_TEST(ALGO,ALGO_REF)	  \
	if (memcmp(buffer_ ## ALGO, buffer_ ## ALGO_REF, buffer_size * V4_N * sizeof(uint32_t)) != 0) { \
		std::cout << "algorithms count_y1_" #ALGO " and count_y1_" #ALGO_REF " differ" << std::endl; \
		for(int i = 0; i < buffer_size * V4_N; ++i) { \
			if (buffer_ ## ALGO [i] != buffer_ ## ALGO_REF [i]) { \
				std::cout << "  at " << i << ": ref == " << buffer_ ## ALGO_REF [i] << " buffer == " << buffer_ ## ALGO [i] << std::endl; \
			} \
		} \
	} else { \
		std::cout << "count_y1_" #ALGO " is ok" << std::endl; \
	}

#define DEL_TEST(ALGO)	\
	free(buffer_ ## ALGO);

uint32_t get_aligned(PVRow row_count)
{
	return ((row_count+3)/4)*4;
}

uint32_t *get_col(uint32_t *p, PVRow row_count, PVCol col)
{
	return p + (col*get_aligned(row_count));
}

template <typename Fn1, typename Fn2, typename Fn3>
void do_one_run(const std::string text,
                const Fn1& allocate, const Fn2& deallocate, const Fn3 &mem_modifier,
                uint32_t *orig_data,
                PVRow row_count, PVCol col_count, PVCol col,
                uint64_t y_min, uint64_t y_max, int zoom, double alpha)
{
	std::cout << text << std::endl;

	Picviz::PVSelection selection;
	int buffer_size = BUFFER_SIZE;
	size_t real_count = get_aligned(row_count) * col_count;

	uint32_t *local_data = allocate(real_count);

	if (local_data == nullptr) {
		throw std::bad_alloc();
	}

	mem_modifier(local_data, real_count);

	uint32_t *data = 0;

	data = get_col(local_data, row_count, col);

	memcpy(data, get_col(orig_data, row_count, col), sizeof(uint32_t) * row_count);

	DEF_TEST_SEQ(seq_v1, data);

	DEF_TEST_SEQ(seq_v2, data);
	if (verbose) {
		CMP_TEST(seq_v2, seq_v1);
	}

	DEF_TEST_SEQ(seq_v3, data);
	if (verbose) {
		CMP_TEST(seq_v3, seq_v1);
	}

	DEF_TEST_SEQ(seq_v4, data);
	if (verbose) {
		CMP_TEST(seq_v4, seq_v1);
	}

	DEF_TEST_SEQ(sse_v3, data);
	if (verbose) {
		CMP_TEST(sse_v3, seq_v1);
	}

	DEF_TEST_SEQ(sse_v4, data);
	if (verbose) {
		CMP_TEST(sse_v4, sse_v3);
	}

#if 0
#ifdef __AVX__
	// DEF_TEST_SEQ(avx_v3, data);
	// if (verbose) {
	// 	CMP_TEST(avx_v3, seq_v1);
	// }
	// DEL_TEST(avx_v3);
#endif
#endif

	DEF_TEST_OMP(omp_sse_v3, data);
	if (verbose) {
		CMP_TEST(omp_sse_v3, seq_v3);
	}

	DEF_TEST_OMP(omp_sse_v3_2, data);
	if (verbose) {
		CMP_TEST(omp_sse_v3_2, seq_v3);
	}

	DEF_TEST_OMP(omp_sse_v4, data);
	if (verbose) {
		CMP_TEST(omp_sse_v4, sse_v4);
	}

	PVParallelView::PVZoneTree *zt = new PVParallelView::PVZoneTree();
	PVParallelView::PVHitGraphData lib_omp(NBITS, V4_N);

	DEF_TEST_OMP(lib, data, lib_omp, *zt);
	if (verbose) {
		CMP_TEST(lib, sse_v4);
	}

	delete zt;

#if 0
	for(int i = 0; i < buffer_size * V4_N; ++i) {
		std::cout << "buffer[" << i << "] =  " << buffer_seq_v1[i] << std::endl;
	}
#endif

	DEL_TEST(seq_v1);
	DEL_TEST(seq_v2);
	DEL_TEST(seq_v3);
	DEL_TEST(seq_v4);

	DEL_TEST(sse_v3);
	DEL_TEST(sse_v4);

	DEL_TEST(omp_sse_v3);
	DEL_TEST(omp_sse_v3_2);
	DEL_TEST(omp_sse_v4);

	deallocate(local_data, real_count);
}

template <typename Fn1>
void do_one_allocator(const std::string text, const Fn1& mem_modifier,
                      uint32_t *data,
                      PVRow row_count, PVCol col_count, PVCol col,
                      uint64_t y_min, uint64_t y_max, int zoom, double alpha)
{
#if 0
	do_one_run("mem=aligned"+text,
	           [](size_t n) {
		           uint32_t *mem;
		           if (posix_memalign((void**) &mem, 16, sizeof(uint32_t) * n) < 0) {
			           perror("posix_memalign");
			           throw std::bad_alloc();
		           }
		           return mem;
	           },
	           [](uint32_t *p, size_t) {
		           free(p);
	           },
	           mem_modifier,
	           data, row_count, col_count, col, y_min, y_max, zoom, alpha);

	do_one_run("mem=mmap"+text,
	           [](size_t n) {
		           intptr_t mem = (intptr_t) mmap(NULL, sizeof(uint32_t)*n, PROT_WRITE|PROT_READ,
		                                          MAP_PRIVATE|MAP_ANONYMOUS,
		                                          -1, 0);
		           if (mem == (intptr_t)-1) {
			           perror("mmap");
			           throw std::bad_alloc();
		           }
		           return (uint32_t*)mem;
	           },
	           [](uint32_t *p, size_t n) {
		           munmap(p, sizeof(uint32_t)*n);
	           },
	           mem_modifier,
	           data, row_count, col_count, col, y_min, y_max, zoom, alpha);
#endif

#if 0
	// trop contraignant
	do_one_run("mem=hugepages"+text,
	           [](size_t n) {
		           intptr_t mem = (intptr_t) mmap(NULL, sizeof(uint32_t)*n, PROT_WRITE|PROT_READ,
		                                          MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB,
		                                          -1, 0);
		           if (mem == (intptr_t)-1) {
			           perror("mmap(HUGETLB)");
			           throw std::bad_alloc();
		           }
		           return (uint32_t*)mem;
	           },
	           [](uint32_t *p, size_t n) {
		           munmap(p, sizeof(uint32_t)*n);
	           },
	           mem_modifier,
	           data, row_count, col_count, col, y_min, y_max, zoom, alpha);

	do_one_run("mem=numa_local"+text,
	           [](size_t n) {
		           uint32_t *mem = (uint32_t*)numa_alloc_local(sizeof(uint32_t)*n);
		           if (mem == nullptr) {
			           perror("numa_alloc_local");
			           throw std::bad_alloc();
		           }
		           return mem;
	           },
	           [](uint32_t *p, size_t n) {
		           numa_free(p, sizeof(uint32_t)*n);
	           },
	           mem_modifier,
	           data, row_count, col_count, col, y_min, y_max, zoom, alpha);
#endif

	do_one_run("mem=numa_interleaved"+text,
	           [](size_t n) {
		           uint32_t *mem = (uint32_t*)numa_alloc_interleaved(sizeof(uint32_t)*n);
		           if (mem == nullptr) {
			           perror("numa_alloc_interleaved");
			           throw std::bad_alloc();
		           }
		           return mem;
	           },
	           [](uint32_t *p, size_t n) {
		           numa_free(p, sizeof(uint32_t)*n);
	           },
	           mem_modifier,
	           data, row_count, col_count, col, y_min, y_max, zoom, alpha);

#if 0
	// comme hugepages : trop contraignant
	do_one_run("mem=numa_interleaved+hugepages"+text,
	           [](size_t n) {
		           intptr_t ret = (intptr_t) mmap(NULL, sizeof(uint32_t)*n, PROT_WRITE|PROT_READ,
		                                          MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB,
		                                          -1, 0);
		           if (ret == (intptr_t)-1) {
			           perror("numa_alloc_interleaved_hugepages");
			           throw std::bad_alloc();
		           }
		           uint32_t *mem = (uint32_t*)ret;
		           unsigned long nodemask = 0;
		           int maxnode = numa_max_node();
		           for (int i = 0; i <= maxnode; ++i) {
			           nodemask |= (1<<(i));
		           }
		           std::cout << "nodemask: " << nodemask << std::endl;
		           std::cout << "maxnode: " << maxnode << std::endl;
		           if (mbind(mem, sizeof(uint32_t)*n, MPOL_INTERLEAVE, &nodemask, maxnode, 0) < 0) {
			           perror("mbind");
			           throw std::bad_alloc();
		           }
		           return mem;
	           },
	           [](uint32_t *p, size_t n) {
		           munmap(p, sizeof(uint32_t)*n);
	           },
	           mem_modifier,
	           data, row_count, col_count, col, y_min, y_max, zoom, alpha);
#endif
}

/*****************************************************************************
 * main & co.
 *****************************************************************************/

typedef enum {
	ARG_ROW_NUM = 1,
	ARG_COL_NUM,
	ARG_COL,
	ARG_MIN,
	ARG_ZOOM,
	ARG_ALPHA,
	ARG_COUNT
} EXTRA_ARG;


void usage(const char* prog)
{
	std::cout << "usage: " << basename(prog) << " [-v] row_count col_count col y_min zoom alpha" << std::endl;
}

int main(int argc, char **argv)
{
	int argv_base = 0;

	if (argc == 0) {
		usage(argv[0]);
		exit(1);
	}

	if (strncmp(argv[1], "-v", 2) == 0) {
		if (argc != (ARG_COUNT+1)) {
			usage(argv[0]);
			exit(1);
		}

		verbose = true;
		argv_base = 1;
	} else {
		if (argc != ARG_COUNT) {
			usage(argv[0]);
			exit(1);
		}

		verbose = false;
		argv_base = 0;
	}

	PVRow row_count = atol(argv[argv_base + ARG_ROW_NUM]);
	PVCol col_count = atol(argv[argv_base + ARG_COL_NUM]);
	PVCol col = atol(argv[argv_base + ARG_COL]);
	uint64_t y_min = atol(argv[argv_base + ARG_MIN]);
	int zoom = atol(argv[argv_base + ARG_ZOOM]);
	double alpha = atof(argv[argv_base + ARG_ALPHA]);

	if (verbose) {
		std::cout << "# generating random data" << std::endl;
		std::cout << "  element count: " << row_count << std::endl;
	}
	srand(0);

	const PVRow row_count_aligned = get_aligned(row_count);
	uint32_t *data = new uint32_t [row_count_aligned*col_count];

	for (PVCol j = 0; j < col_count; ++j) {
		for (PVRow i = 0; i < row_count; ++i) {
			data[j*row_count_aligned+i] = (rand() << 10) | (rand()&1023);
		}
	}

	if (verbose) {
		std::cout << "# done" << std::endl;
	}

	if (zoom == 0) {
		zoom = 1;
		std::cout << "INFO: setting zoom to 1 because block algorithm have an exception for zoom == 0"
		          << std::endl;
	}
	uint64_t y_max = y_min + (1UL << (32 - zoom));

	if (verbose) {
		std::cout << "# start test" << std::endl;
	}

	// nothing special
	do_one_allocator(std::string(""),
	                 [](uint32_t* p, size_t n) {},
	                 get_col(data, row_count, col), row_count, col_count, col, y_min, y_max, zoom, alpha);

#if 0
	// add sequential
	do_one_allocator(std::string("+seq"),
	                 [](uint32_t* p, size_t n) {
		                 if (madvise(p, sizeof(uint32_t)*n, MADV_SEQUENTIAL) < 0) {
			                 std::cout << "I: MADV_SEQUENTIAL fails" << std::endl;
		                 }
	                 },
	                 get_col(data, row_count, col), row_count, col_count, col, y_min, y_max, zoom, alpha);

	// add transparent huge pages
	do_one_allocator(std::string("+thp"),
	                 [](uint32_t* p, size_t n) {
		                 if (madvise(p, sizeof(uint32_t)*n, MADV_HUGEPAGE) < 0) {
			                 std::cout << "I: MADV_HUGEPAGE fails" << std::endl;
		                 }
	                 },
	                 get_col(data, row_count, col), row_count, col_count, col, y_min, y_max, zoom, alpha);

	// add sequential and transparent huge pages
	do_one_allocator(std::string("+seq+thp"),
	                 [](uint32_t* p, size_t n) {
		                 if (madvise(p, sizeof(uint32_t)*n, MADV_SEQUENTIAL) < 0) {
			                 std::cout << "I: MADV_SEQUENTIAL fails" << std::endl;
		                 }
		                 if (madvise(p, sizeof(uint32_t)*n, MADV_HUGEPAGE) < 0) {
			                 std::cout << "I: MADV_HUGEPAGE fails" << std::endl;
		                 }

	                 },
	                 get_col(data, row_count, col), row_count, col_count, col, y_min, y_max, zoom, alpha);
#endif

	delete [] data;

	return 0;
}
