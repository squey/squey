
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
typedef tlr_buffer_t::index_t                                              tlr_index_t;
typedef PVParallelView::PVBCICode<PARALLELVIEW_ZZT_BBITS>                  bci_code_t;
typedef PVParallelView::constants<PARALLELVIEW_ZZT_BBITS>                  constants;

constexpr static uint32_t mask_int_ycoord = constants::mask_int_ycoord;

// #define USE_SSE_BOOL_ACCESS

uint32_t tested_count_seq = 0;
uint32_t tested_count_sse = 0;
uint32_t inserted_count_seq = 0;
uint32_t inserted_count_sse = 0;


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
 * copy/paste from namespace PVParallelView
 *****************************************************************************/

static inline double get_scale_factor(uint32_t zoom)
{
	return pow(2, zoom);
}

static inline void compute_bci_projection_y2(const uint64_t y1,
                                             const uint64_t y2,
                                             const uint64_t y_min,
                                             const uint64_t y_lim,
                                             const int shift,
                                             const uint32_t mask,
                                             const uint32_t width,
                                             const float beta,
                                             bci_code_t &bci)
{
	bci.s.l = ((y1 - y_min) >> shift) & mask;

	int64_t dy = (int64_t)y2 - (int64_t)y1;
	double y2p = (double)y1 + dy * (double)beta;

	if (y2p >= y_lim) {
		bci.s.type = bci_code_t::DOWN;
		bci.s.r = ((double)width * (double)(y_lim - y1)) / (double)(y2p - y1);
	} else if (y2p < y_min) {
		bci.s.type = bci_code_t::UP;
		bci.s.r = ((double)width * (double)(y1 - y_min)) / (double)(y1 - y2p);
	} else {
		bci.s.type = bci_code_t::STRAIGHT;
		bci.s.r = (((uint32_t)(y2p - y_min)) >> shift) & mask;
	}
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
	std::cerr << "usage: " << basename(program) << " num zoom\n" << std::endl;
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
		++tested_count_seq;
		if (!test_f(e, y1_min, y1_max)) {
			continue;
		}
		const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
		if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
			continue;
		}
		insert_f(e, tlr);
		++inserted_count_seq;
		B_SET(buffer[pos >> 5], pos & 31);
		--remaining;
		if (remaining == 0) {
			break;
		}
	}
}


void _sse_p32(const char *text, const __m128i &sse_reg)
{
	std::cout << text << ": [ "
	          << _mm_extract_epi32(sse_reg, 3) << " | "
	          << _mm_extract_epi32(sse_reg, 2) << " | "
	          << _mm_extract_epi32(sse_reg, 1) << " | "
	          << _mm_extract_epi32(sse_reg, 0) << " ]" << std::endl;
}

void _sse_p64(const char *text, const __m128i &sse_reg)
{
	std::cout << text << ": [ "
	          << _mm_extract_epi64(sse_reg, 1) << " | "
	          << _mm_extract_epi64(sse_reg, 0) << " ]" << std::endl;
}

#ifdef PRINT_SSE
#define sse_p32(R) _sse_p32(#R, R)
#define sse_p64(R) _sse_p64(#R, R)
#else
#define sse_p32(R)
#define sse_p64(R)
#endif

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

bool test_sse2(const __m128i &sse_y1, const __m128i &sse_y1_min, const __m128i &sse_y1_max, __m128i &sse_res)
{
	static const __m128i sse_full_ones  = _mm_set1_epi32(0xFFFFFFFF);
	static const __m128i sse_full_zeros = _mm_set1_epi32(0);

	/* expand 4x32b register into 2 2x64b registers
	 */

#ifdef PRINT_SSE
	std::cout << "#############################################################" << std::endl;
#endif
	sse_p32(sse_y1);

	const __m128i sse_y1_0 = _mm_unpacklo_epi32 (sse_y1, sse_full_zeros);
	const __m128i sse_y1_1 = _mm_unpackhi_epi32 (sse_y1, sse_full_zeros);

	/* doing registers test against min
	 */
	__m128i sse_min0 = _mm_cmpgt_epi64(sse_y1_min, sse_y1_0);
	__m128i sse_min1 = _mm_cmpgt_epi64(sse_y1_min, sse_y1_1);

	sse_p64(sse_y1_min);
	sse_p64(sse_y1_0);
	sse_p64(sse_min0);

	sse_p64(sse_y1_min);
	sse_p64(sse_y1_1);
	sse_p64(sse_min1);

	/* doing registers test against max
	 */
	__m128i sse_max0 = _mm_cmpgt_epi64(sse_y1_max, sse_y1_0);
	__m128i sse_max1 = _mm_cmpgt_epi64(sse_y1_max, sse_y1_1);

	sse_p64(sse_y1_max);
	sse_p64(sse_y1_0);
	sse_p64(sse_max0);

	sse_p64(sse_y1_max);
	sse_p64(sse_y1_1);
	sse_p64(sse_max1);

	/*
	 * fusion des resultats (en passant ces valeurs de 32 à 64 bits)
	 *
	 * resultat sur 64b
	 * min0 = [ v01 | v00 ]
	 * min1 = [ v11 | v10 ]
	 *
	 * <=>
	 *
	 * resultat sur 32b (car 0 ou 0xFFFFFFFF
	 * min0 = [ v01 | v01 | v00 | v00 ]
	 * min1 = [ v11 | v11 | v10 | v10 ]
	 *
	 * en décalant de 4 octets vers la droite, on "regroupe" vX1 et vX0 dans
	 * le même mot de 64 bits :
	 *
	 * mins0 = _mm_srli_si128(min0, 4);
	 * mins1 = _mm_srli_si128(min1, 4);

	 * mins0 = [ 0 | v01 | v01 | v00 ]
	 * mins1 = [ 0 | v11 | v11 | v10 ]
	 */

	__m128i sse_ms0 = _mm_srli_si128(sse_min0, 4);
	__m128i sse_ms1 = _mm_srli_si128(sse_min1, 4);

	/* puis un unpacklo_epi64 pour fusionner les 2 résultats partiels
	 * dans un seul registre
	 */
	__m128i sse_tmin = _mm_unpacklo_epi64(sse_ms0, sse_ms1);
	sse_p32(sse_tmin);

	/* on fait de même pour les max
	 */
	sse_ms0 = _mm_srli_si128(sse_max0, 4);
	sse_ms1 = _mm_srli_si128(sse_max1, 4);

	__m128i sse_tmax = _mm_unpacklo_epi64(sse_ms0, sse_ms1);
	sse_p32(sse_tmax);

	sse_res = _mm_andnot_si128(sse_tmin, sse_tmax);

	int res = _mm_testz_si128(sse_res, sse_full_ones);
#ifdef PRINT_SSE
	std::cout << "res: " << res << std::endl;
#endif
	sse_p32(sse_res);

	return res;
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

	const __m128i sse_y1_min            = _mm_set1_epi64x(y1_min);
	const __m128i sse_y1_max            = _mm_set1_epi64x(y1_max);
	const __m128i sse_y1_orig           = _mm_set1_epi32(y1_orig);
	const __m128i sse_y1_shift          = _mm_set1_epi32(y1_shift);
	const __m128i sse_ly1_min           = _mm_set1_epi32(ly1_min);
	const __m128i sse_y2_orig           = _mm_set1_epi32(y2_orig);
	const __m128i sse_y2_shift          = _mm_set1_epi32(y2_shift);
	const __m128i sse_clipped_max_count = _mm_set1_epi32(clipped_max_count);

#ifdef USE_SSE_BOOL_ACCESS
	static const __m128i sse_pos_mask   = _mm_set1_epi32(31);
#endif

	size_t packed_size = size & ~3;
	for(size_t i = 0; i < packed_size; i += 4) {
		const quadtree_entry_t &e0 = entries[i];
		const quadtree_entry_t &e1 = entries[i+1];
		const quadtree_entry_t &e2 = entries[i+2];
		const quadtree_entry_t &e3 = entries[i+3];

		// TODO: compact all _mm_xxxxx expressions ;-)
		__m128i sse_r0 = _mm_loadu_si128((const __m128i*) &e0);
		__m128i sse_r1 = _mm_loadu_si128((const __m128i*) &e1);
		__m128i sse_r2 = _mm_loadu_si128((const __m128i*) &e2);
		__m128i sse_r3 = _mm_loadu_si128((const __m128i*) &e3);

		/* "transposition" partielle pour avoir les y1 dans un registre
		 * et les y2 dans un autre
		 */
		__m128i sse_tmp01 = _mm_unpacklo_epi32(sse_r0, sse_r1);
		__m128i sse_tmp23 = _mm_unpacklo_epi32(sse_r2, sse_r3);

		__m128i sse_y1 = _mm_unpacklo_epi64(sse_tmp01, sse_tmp23);

		__m128i sse_test;

		tested_count_sse += 4;
		if (test_sse2(sse_y1, sse_y1_min, sse_y1_max, sse_test)) {
			continue;
		}

		/* sse_y2 n'est pas encore utile, donc autant ne pas le
		 * calculer
		 */
		__m128i sse_y2 = _mm_unpackhi_epi64(sse_tmp01, sse_tmp23);

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

#ifdef USE_SSE_BOOL_ACCESS
		__m128i sse_block = _mm_srli_epi32(sse_pos, 5);
		__m128i sse_index = _mm_and_si128(sse_pos, sse_pos_mask);
#endif

		if(_mm_extract_epi32(sse_test, 0)) {
#ifdef USE_SSE_BOOL_ACCESS
			uint32_t b = _mm_extract_epi32(sse_block, 0);
			uint32_t i = _mm_extract_epi32(sse_index, 0);
			if (!(B_IS_SET(buffer[b], i))) {
#else
			uint32_t p = _mm_extract_epi32(sse_pos, 0);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
#endif
				insert_f(e0, tlr);
				++inserted_count_sse;
#ifdef USE_SSE_BOOL_ACCESS
				B_SET(buffer[b], i);
#else
				B_SET(buffer[p >> 5], p & 31);
#endif
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		if(_mm_extract_epi32(sse_test, 1)) {
#ifdef USE_SSE_BOOL_ACCESS
			uint32_t b = _mm_extract_epi32(sse_block, 1);
			uint32_t i = _mm_extract_epi32(sse_index, 1);
			if (!(B_IS_SET(buffer[b], i))) {
#else
			uint32_t p = _mm_extract_epi32(sse_pos, 1);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
#endif
				insert_f(e1, tlr);
				++inserted_count_sse;
#ifdef USE_SSE_BOOL_ACCESS
				B_SET(buffer[b], i);
#else
				B_SET(buffer[p >> 5], p & 31);
#endif
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		if(_mm_extract_epi32(sse_test, 2)) {
#ifdef USE_SSE_BOOL_ACCESS
			uint32_t b = _mm_extract_epi32(sse_block, 2);
			uint32_t i = _mm_extract_epi32(sse_index, 2);
			if (!(B_IS_SET(buffer[b], i))) {
#else
			uint32_t p = _mm_extract_epi32(sse_pos, 2);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
#endif
				insert_f(e2, tlr);
				++inserted_count_sse;
#ifdef USE_SSE_BOOL_ACCESS
				B_SET(buffer[b], i);
#else
				B_SET(buffer[p >> 5], p & 31);
#endif
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		if(_mm_extract_epi32(sse_test, 3)) {
#ifdef USE_SSE_BOOL_ACCESS
			uint32_t b = _mm_extract_epi32(sse_block, 3);
			uint32_t i = _mm_extract_epi32(sse_index, 3);
			if (!(B_IS_SET(buffer[b], i))) {
#else
			uint32_t p = _mm_extract_epi32(sse_pos, 3);
			if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
#endif
				insert_f(e3, tlr);
				++inserted_count_sse;
#ifdef USE_SSE_BOOL_ACCESS
				B_SET(buffer[b], i);
#else
				B_SET(buffer[p >> 5], p & 31);
#endif
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}
	}

	for(size_t i = packed_size; i < size; ++i) {
		const quadtree_entry_t &e = entries[i];
		++tested_count_sse;
		if (!test_f(e, y1_min, y1_max)) {
			continue;
		}
		const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
		if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
			continue;
		}
		insert_f(e, tlr);
		++inserted_count_sse;
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

	const uint64_t y1_min = 1;
	const uint64_t y1_max = y1_min + (1UL << (32 - zoom));
	const uint64_t y1_lim = y1_max;
	std::cout << "y1_min  : " << y1_min << std::endl;
	std::cout << "y1_max  : " << y1_max << std::endl;
	std::cout << "y1_lim  : " << y1_lim << std::endl;

	std::cout << "sizeof(quadtree_entry_t) = " << sizeof(quadtree_entry_t) << std::endl;

	const __m128i sse_y1_min = _mm_set1_epi64x(y1_min);
	const __m128i sse_y1_max = _mm_set1_epi64x(y1_max);
	__m128i sse_vec, sse_res;

	uint32_t v_out = 0;
	uint32_t v_in = 1024;

	std::cout << "test_sse with values around y1_min" << std::endl;
	for(size_t i = 0; i < 16; ++i) {
		uint32_t v0 = (i&1)?v_in:v_out;
		uint32_t v1 = (i&2)?v_in:v_out;
		uint32_t v2 = (i&4)?v_in:v_out;
		uint32_t v3 = (i&8)?v_in:v_out;

		sse_vec = _mm_set_epi32(v3, v2, v1, v0);

		int res = test_sse2(sse_vec, sse_y1_min, sse_y1_max, sse_res);

		printf ("  sse_vec: [ %4u %4u %4u %4u ] => res: %d - [ %2d %2d %2d %2d ]\n",
		        _mm_extract_epi32(sse_vec, 3),
		        _mm_extract_epi32(sse_vec, 2),
		        _mm_extract_epi32(sse_vec, 1),
		        _mm_extract_epi32(sse_vec, 0),
		        res,
		        _mm_extract_epi32(sse_res, 3),
		        _mm_extract_epi32(sse_res, 2),
		        _mm_extract_epi32(sse_res, 1),
		        _mm_extract_epi32(sse_res, 0));
	}

	v_out = 2050;
	v_in = 1024;

	std::cout << "test_sse with values around y2_min" << std::endl;
	for(size_t i = 0; i < 16; ++i) {
		uint32_t v0 = (i&1)?v_in:v_out;
		uint32_t v1 = (i&2)?v_in:v_out;
		uint32_t v2 = (i&4)?v_in:v_out;
		uint32_t v3 = (i&8)?v_in:v_out;

		sse_vec = _mm_set_epi32(v3, v2, v1, v0);
		int res = test_sse2(sse_vec, sse_y1_min, sse_y1_max, sse_res);

		printf ("  sse_vec: [ %4u %4u %4u %4u ] => res: %d - [ %2d %2d %2d %2d ]\n",
		        _mm_extract_epi32(sse_vec, 3),
		        _mm_extract_epi32(sse_vec, 2),
		        _mm_extract_epi32(sse_vec, 1),
		        _mm_extract_epi32(sse_vec, 0),
		        res,
		        _mm_extract_epi32(sse_res, 3),
		        _mm_extract_epi32(sse_res, 2),
		        _mm_extract_epi32(sse_res, 1),
		        _mm_extract_epi32(sse_res, 0));
	}

	const uint64_t y_min = y1_min;
	const uint64_t y_lim = y1_lim;
	const uint32_t shift = (32 - PARALLELVIEW_ZZT_BBITS) - zoom;
	const uint32_t width = 512;
	double beta = 1. / get_scale_factor(zoom);

	tlr_seq->clear();
	BENCH_START(seq);
	extract_seq(entries, ent_num,
	            0, BUCKET_ELT_COUNT >> 1,
	            0, BUCKET_ELT_COUNT >> 1,
	            y1_min, y1_max,
	            zoom, 4096,
	            [](const quadtree_entry_t &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
	            {
		            return (e.y1 >= y1_min) && (e.y1 < y1_max);
	            },
	            [&](const quadtree_entry_t &e, tlr_buffer_t &buffer)
	            {
		            bci_code_t bci;
		            compute_bci_projection_y2(e.y1, e.y2,
		                                      y_min, y_lim,
		                                      shift, mask_int_ycoord,
		                                      width, beta, bci);
		            tlr_index_t tlr(bci.s.type,
		                            bci.s.l,
		                            bci.s.r);
		            if (e.idx < buffer[tlr.v]) {
			            buffer[tlr.v] = e.idx;
		            }
	            },
	            buffer, *tlr_seq);
	BENCH_STOP(seq);
	BENCH_SHOW(seq, "sequential run", 0, 0, 0, 0);
	double seq_dt = BENCH_END_TIME(seq);

	tlr_sse->clear();
	BENCH_START(sse);
	extract_sse(entries, ent_num,
	            0, BUCKET_ELT_COUNT >> 1,
	            0, BUCKET_ELT_COUNT >> 1,
	            y1_min, y1_max,
	            zoom, 4096,
	            [](const quadtree_entry_t &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
	            {
		            return (y1_min <= e.y1) && (e.y1 < y1_max);
	            },
	            [&](const quadtree_entry_t &e, tlr_buffer_t &buffer)
	            {
		            bci_code_t bci;
		            compute_bci_projection_y2(e.y1, e.y2,
		                                      y_min, y_lim,
		                                      shift, mask_int_ycoord,
		                                      width, beta, bci);
		            tlr_index_t tlr(bci.s.type,
		                            bci.s.l,
		                            bci.s.r);
		            if (e.idx < buffer[tlr.v]) {
			            buffer[tlr.v] = e.idx;
		            }
	            },
	            buffer, *tlr_sse);
	BENCH_STOP(sse);
	BENCH_SHOW(sse, "vectorized run", 0, 0, 0, 0);
	double sse_dt = BENCH_END_TIME(sse);

	std::cout << "speedup: " << seq_dt / sse_dt << std::endl;

	bool diff = false;

	int seq_found_count = 0;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr_seq)[i] != UINT32_MAX) {
			++seq_found_count;
		}
	}
	std::cout << "BCI codes found in seq version: " << seq_found_count << std::endl;

	int sse_found_count = 0;
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr_sse)[i] != UINT32_MAX) {
			++sse_found_count;
		}
	}
	std::cout << "BCI codes found in sse version: " << sse_found_count << std::endl;

	//std::cout << "diff at:";
	for(size_t i = 0; i < tlr_buffer_t::length; ++i) {
		if ((*tlr_seq)[i] != (*tlr_sse)[i]) {
			diff = true;
			//std::cout << " " << i;
		}
	}
	//std::cout << std::endl;

	if (diff) {
		std::cout << "TLRBuffers are different" << std::endl;
	} else {
		std::cout << "TLRBuffers are equal" << std::endl;
	}

	std::cout << "seq:" << std::endl;
	std::cout << "  tested  : " << tested_count_seq << std::endl;
	std::cout << "  inserted: " << inserted_count_seq << std::endl;

	std::cout << "sse:" << std::endl;
	std::cout << "  tested  : " << tested_count_sse << std::endl;
	std::cout << "  inserted: " << inserted_count_sse << std::endl;

	return 0;
}

