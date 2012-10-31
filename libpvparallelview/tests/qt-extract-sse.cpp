
#include <x86intrin.h>

#include <iostream>

#include <bithacks.h>

#include <pvkernel/core/PVAllocators.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVQuadTree.h>

#define QT_MAX_VALUE (BUCKET_ELT_COUNT)

typedef PVParallelView::PVQuadTreeEntry                                    quadtree_entry_t;
typedef PVParallelView::PVQuadTree<10000, 1000, 0, PARALLELVIEW_ZZT_BBITS> quadtree_t;
typedef quadtree_t::pv_tlr_buffer_t                                        tlr_buffer_t;

/*****************************************************************************
 * about data
 */
size_t init_count(size_t num)
{
	return 2048 * num;
}

quadtree_entry_t *init_entries(size_t num)
{
	typedef PVCore::PVAlignedAllocator<quadtree_entry_t, 16> Alloc;
	quadtree_entry_t *entries = Alloc().allocate(init_count(num));
	size_t idx = 0;
	for(size_t i = 0; i < 2048; ++i) {
		for(size_t j = 0; j < num; ++j) {
			entries[i].idx = idx;
			entries[i].y1 = i;
			entries[i].y2 = (j / (double)num) * QT_MAX_VALUE;
			++idx;
		}
	}
	return entries;
}

/*****************************************************************************
 * about program's parameters
 */
enum {
	P_PROG = 0,
	P_NUM,
	P_ZOOM,
	P_MAX_VALUE
};

#define PROGNAME
void usage(const char *program)
{
	std::cerr << "usage: " << basename(program) << " num\n" << std::endl;
	std::cerr << "\tnum  : number of events for each primary coordinate event" << std::endl;
	std::cerr << "\tzoom : zoom level in [0,21]" << std::endl;
}

/*****************************************************************************
 * sequential extraction
 */
template <typename Ftest, typename Finsert>
void extract_seq(const quadtree_entry_t *entries, const size_t size,
                 const uint32_t y1_min_value, const uint32_t y1_mid_value,
                 const uint32_t y2_min_value, const uint32_t y2_mid_value,
                 const uint64_t y1_min, const uint64_t y1_max,
                 const uint32_t zoom, const uint32_t y2_count,
                 const Ftest &test_f,
                 const Finsert &insert_f,
                 uint32_t *buffer,
                 tlr_buffer_t &tlr)
{
	const uint64_t max_count = 1 << zoom;
	const uint64_t y1_orig = y1_min_value;
	const uint64_t y1_len = (y1_mid_value - y1_orig) * 2;
	const uint64_t y1_scale = y1_len / max_count;
	const uint64_t y2_orig = y2_min_value;
	const uint64_t y2_scale = ((y2_mid_value - y2_orig) * 2) / y2_count;
	const uint64_t ly1_min = (PVCore::clamp(y1_min, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
	const uint64_t ly1_max = (PVCore::clamp(y1_max, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
	const uint64_t clipped_max_count = PVCore::max(1UL, ly1_max - ly1_min);
	const size_t count_aligned = ((clipped_max_count * y2_count) + 31) / 32;
	memset(buffer, 0, count_aligned * sizeof(uint32_t));
	uint32_t remaining = clipped_max_count * y2_count;

	for(size_t i = 0; i < size; ++i) {
		const quadtree_entry_t &e = entries[i];
		if (!test_f(e, y1_min, y1_max)) {
			continue;
		}
		const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
		if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
			continue;
		}
		insert_f(e, tlr);
		B_SET(buffer[pos >> 5], pos & 31);
		--remaining;
		if (remaining == 0) {
			break;
		}
	}
}

#define NONE 0
bool test_sse(const __m128i &sse_y1, const __m128i &sse_y1_min, const __m128i &sse_y1_max, __m128i &sse_res)
{
	static const __m128i shuffle_mask      = _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF);
	static const __m128i full_mask         = _mm_set1_epi32(0xFFFFFFFF);

	/* expand 4x32b register into 2 2x64b registers
	 */

	// TODO: remplacer les shuffle par un masque pour _0 et masque + shift pour _1
	__m128i sse_y1_0t = _mm_shuffle_epi32(sse_y1, _MM_SHUFFLE(NONE, 1, NONE, 0));
	__m128i sse_y1_1t = _mm_shuffle_epi32(sse_y1, _MM_SHUFFLE(NONE, 3, NONE, 2));

	__m128i sse_y1_0 = _mm_and_si128(sse_y1_0t, shuffle_mask);
	__m128i sse_y1_1 = _mm_and_si128(sse_y1_1t, shuffle_mask);

	/* doing registers test against min
	 */
	__m128i sse_min0 = _mm_cmpgt_epi64(sse_y1_min, sse_y1_0);
	__m128i sse_min1 = _mm_cmpgt_epi64(sse_y1_min, sse_y1_1);

	/* doing registers test against max
	 */
	__m128i sse_max0 = _mm_cmpgt_epi64(sse_y1_0, sse_y1_max);
	__m128i sse_max1 = _mm_cmpgt_epi64(sse_y1_1, sse_y1_max);

	/* "unshuffle" min result into 1 register
	 */
	__m128i sse_min_r0 = _mm_shuffle_epi32(sse_min0, _MM_SHUFFLE(NONE, NONE,    2,    0));
	__m128i sse_min_r1 = _mm_shuffle_epi32(sse_min1, _MM_SHUFFLE(   2,    0, NONE, NONE));
	__m128i sse_tmin = _mm_or_si128(sse_min_r0, sse_min_r1);

	/* "unshuffle" min result into 1 register
	 */
	__m128i sse_max_r0 = _mm_shuffle_epi32(sse_max0, _MM_SHUFFLE(NONE, NONE,    2,    0));
	__m128i sse_max_r1 = _mm_shuffle_epi32(sse_max1, _MM_SHUFFLE(   2,    0, NONE, NONE));
	__m128i sse_tmax = _mm_or_si128(sse_max_r0, sse_max_r1);

	sse_res = _mm_andnot_si128(sse_tmin, sse_tmax);

	return _mm_testz_si128(sse_res, full_mask);
}
#undef NONE

/*****************************************************************************
 * vectorized extraction
 */
template <typename Ftest, typename Finsert>
void extract_sse(const quadtree_entry_t *entries, const size_t size,
                 const uint32_t y1_min_value, const uint32_t y1_mid_value,
                 const uint32_t y2_min_value, const uint32_t y2_mid_value,
                 const uint64_t y1_min, const uint64_t y1_max,
                 const uint32_t zoom, const uint32_t y2_count,
                 const Ftest &test_f,
                 const Finsert &insert_f,
                 uint32_t *buffer,
                 tlr_buffer_t &tlr)
{
	const uint64_t max_count = 1 << zoom;
	const uint64_t y1_orig = y1_min_value;
	const uint64_t y1_len = (y1_mid_value - y1_orig) * 2;
	const uint64_t y1_scale = y1_len / max_count;
	const uint64_t y1_shift = PVCore::upper_power_of_2(y1_scale);
	const uint64_t y2_orig = y2_min_value;
	const uint64_t y2_scale = ((y2_mid_value - y2_orig) * 2) / y2_count;
	const uint64_t y2_shift = PVCore::upper_power_of_2(y2_scale);
	const uint64_t ly1_min = (PVCore::clamp(y1_min, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
	const uint64_t ly1_max = (PVCore::clamp(y1_max, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
	const uint64_t clipped_max_count = PVCore::max(1UL, ly1_max - ly1_min);
	const size_t count_aligned = ((clipped_max_count * y2_count) + 31) / 32;
	memset(buffer, 0, count_aligned * sizeof(uint32_t));
	uint32_t remaining = clipped_max_count * y2_count;

	const __m128i sse_y1_min            = _mm_set1_epi32(y1_min);
	const __m128i sse_y1_max            = _mm_set1_epi32(y1_max);
	const __m128i sse_y1_orig           = _mm_set1_epi32(y1_orig);
	const __m128i sse_y1_scale          = _mm_set1_epi32(y1_scale);
	const __m128i sse_y1_shift          = _mm_set1_epi32(y1_shift);
	const __m128i sse_ly1_min           = _mm_set1_epi32(ly1_min);
	const __m128i sse_y2_orig           = _mm_set1_epi32(y2_orig);
	const __m128i sse_y2_scale          = _mm_set1_epi32(y2_scale);
	const __m128i sse_y2_shift          = _mm_set1_epi32(y2_shift);
	const __m128i sse_clipped_max_count = _mm_set1_epi32(clipped_max_count);
	const __m128i sse_all0              = _mm_set1_epi32(0); // peut sans doute faire mieux

	size_t packed_size = size & ~3;
	for(size_t i = 0; i < packed_size; i += 4) {
		const quadtree_entry_t &e0 = entries[i];
		const quadtree_entry_t &e1 = entries[i+1];
		const quadtree_entry_t &e2 = entries[i+2];
		const quadtree_entry_t &e3 = entries[i+3];

		// TODO: compact all _mm_xxxxx expressions ;-)
		__m128i sse_r0 = _mm_load_si128((const __m128i*) &e0);
		__m128i sse_r1 = _mm_load_si128((const __m128i*) &e1);
		__m128i sse_r2 = _mm_load_si128((const __m128i*) &e2);
		__m128i sse_r3 = _mm_load_si128((const __m128i*) &e3);

		/*
		 * a0 b0 c0 d0     a0 a1 a2 a3
		 * a1 b1 c1 d1     b0 b1 b2 b3
		 * a2 b2 c2 d2  => c0 c1 c2 c3
		 * a3 b3 c3 d3     d0 d1 d2 d3
		 *
		 * => sse_y1 == sse_r3
		 * => sse_y2 == sse_r2
		 */
		_MM_TRANSPOSE4_PS(sse_r0, sse_r1, sse_r2, sse_r3);

		// I know, but it is more readable :-P
		__m128i sse_y1 = sse_r3;
		__m128i sse_y2 = sse_r2;
		__m128i sse_test;

		if (!test_sse(sse_y1, sse_y1_min, sse_y1_max, sse_test)) {
			continue;
		}

		/*
		  pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min)
		        + clipped_max_count * ((e.y2 - y2_orig) / y2_scale)
		 */
		__m128i sse_0s  = _mm_sub_epi32(sse_y1, sse_y1_orig);
		__m128i sse_0sd = _mm_srl_epi32(sse_0s, sse_y1_shift);
		__m128i sse_0x  = _mm_sub_epi32(sse_0sd, sse_ly1_min);

		__m128i sse_1s  = _mm_sub_epi32(sse_y2, sse_y2_orig);
		__m128i sse_1sd = _mm_srl_epi32(sse_1s, sse_y2_shift);
		__m128i sse_1y  = _mm_mullo_epi32(sse_1sd, sse_clipped_max_count);

		__m128i sse_pos = _mm_add_epi32(sse_0x, sse_1y);

		// TODO: faire les shift de 5 + mask en SSE
		if(_mm_extract_epi32(sse_test, 0)) {
			uint32_t p = _mm_extract_epi32(sse_pos, 0);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
				insert_f(e0, tlr);
				B_SET(buffer[p >> 5], p & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		if(_mm_extract_epi32(sse_test, 1)) {
			uint32_t p = _mm_extract_epi32(sse_pos, 1);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
				insert_f(e1, tlr);
				B_SET(buffer[p >> 5], p & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		if(_mm_extract_epi32(sse_test, 2)) {
			uint32_t p = _mm_extract_epi32(sse_pos, 2);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
				insert_f(e2, tlr);
				B_SET(buffer[p >> 5], p & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		if(_mm_extract_epi32(sse_test, 3)) {
			uint32_t p = _mm_extract_epi32(sse_pos, 3);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
				insert_f(e3, tlr);
				B_SET(buffer[p >> 5], p & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}
	}

	for(size_t i = packed_size; i < size; ++i) {
		const quadtree_entry_t &e = entries[i];
		if (!test_f(e, y1_min, y1_max)) {
			continue;
		}
		const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
		if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
			continue;
		}
		insert_f(e, tlr);
		B_SET(buffer[pos >> 5], pos & 31);
		--remaining;
		if (remaining == 0) {
			break;
		}
	}
}

/*****************************************************************************
 * main
 */
int main(int argc, char **argv)
{
	if (argc != P_MAX_VALUE) {
		usage(argv[P_PROG]);
		exit(1);
	}

	size_t num = (size_t) atol(argv[P_NUM]);
	uint32_t zoom = (uint32_t) atol(argv[P_ZOOM]);

	quadtree_entry_t *entries = init_entries(num);
	uint32_t *buffer = new uint32_t [2048 * 4096];
	tlr_buffer_t *tlr_seq = new tlr_buffer_t;
	tlr_buffer_t *tlr_sse = new tlr_buffer_t;

	size_t ent_num = init_count(num);

	const uint64_t y1_min = 0;
	const uint64_t y1_max = 1UL << (32 - zoom);
	const uint64_t y1_lim = y1_max;
	std::cout << "y1_min  : " << y1_min << std::endl;
	std::cout << "y1_max  : " << y1_max << std::endl;
	std::cout << "y1_lim  : " << y1_lim << std::endl;

	tlr_seq->clear();
	BENCH_START(seq);
	extract_seq(entries, ent_num,
	            0, BUCKET_ELT_COUNT >> 1,
	            0, BUCKET_ELT_COUNT >> 1,
	            y1_min, y1_max,
	            zoom, 2048,
	            [](const quadtree_entry_t &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
	            {
		            return (e.y1 >= y1_min) && (e.y1 < y1_max);
	            },
	            [](const quadtree_entry_t &e, tlr_buffer_t &buffer)
	            {
		            // TODO: write a working insert_f :-P
	            },
	            buffer, *tlr_seq);
	BENCH_STOP(seq);
	BENCH_SHOW(seq, "sequential run", 0, 0, 0, 0);
	double seq_dt = BENCH_END_TIME(seq);

	tlr_sse->clear();
	BENCH_START(sse);
	extract_seq(entries, ent_num,
	            0, BUCKET_ELT_COUNT >> 1,
	            0, BUCKET_ELT_COUNT >> 1,
	            y1_min, y1_max,
	            zoom, 2048,
	            [](const quadtree_entry_t &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
	            {
		            return (y1_min <= e.y1) && (e.y1 < y1_max);
	            },
	            [](const quadtree_entry_t &e, tlr_buffer_t &buffer)
	            {
		            // TODO: write a working insert_f :-P
	            },
	            buffer, *tlr_sse);
	BENCH_STOP(sse);
	BENCH_SHOW(sse, "vectorized run", 0, 0, 0, 0);
	double sse_dt = BENCH_END_TIME(sse);

	std::cout << "speedup: " << seq_dt / sse_dt << std::endl;

	bool diff = false;
	int diff_count = 0;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr_seq)[i] != UINT32_MAX) {
			++diff_count;
		}
		if ((*tlr_seq)[i] != (*tlr_seq)[i]) {
			diff = true;
			break;
		}
	}

	if (diff_count) {
		std::cout << "BCI codes found!" << std::endl;
	} else{
		std::cout << "no BCI codes found!" << std::endl;
	}

	if (diff) {
		std::cout << "TLRBuffers are different" << std::endl;
	} else {
		std::cout << "TLRBuffers are equal" << std::endl;
	}

	return 0;
}

