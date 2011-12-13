#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>

// SSE-transposition functions
// Adapted from GCC's xmmintrin.h
#define MM_TRANSPOSE4_PS_DST(row0, row1, row2, row3) \
	do {                                    \
		__v4sf __r0 = (__v4sf) (row0), __r1 = (row1), __r2 = (row2), __r3 = (row3);    \
		__v4sf __t0 = __builtin_ia32_unpcklps (__r0, __r1);           \
		__v4sf __t1 = __builtin_ia32_unpcklps (__r2, __r3);           \
		__v4sf __t2 = __builtin_ia32_unpckhps (__r0, __r1);           \
		__v4sf __t3 = __builtin_ia32_unpckhps (__r2, __r3);           \
		(row0) = __builtin_ia32_movlhps (__t0, __t1);             \
		(row1) = __builtin_ia32_movhlps (__t1, __t0);             \
		(row2) = __builtin_ia32_movlhps (__t2, __t3);             \
		(row3) = __builtin_ia32_movhlps (__t3, __t2);             \
	} while (0)

// Must be specified for template-function specialisation

namespace PVCore {

void __transpose_float(float* res, float* data, uint32_t nrows, uint32_t ncols)
{
	__v4sf sse_row0, sse_row1, sse_row2, sse_row3;

	__m128i sse_4ncols = _mm_set1_epi32(4*ncols);
	__m128i sse_off = _mm_set_epi32(3*ncols, 2*ncols, 1*ncols, 0);
	__m128i sse_4 = _mm_set1_epi32(4);
	uint32_t DECLARE_ALIGN(32) loff[4];

	for (uint32_t i = 0; i < (nrows/4)*4; i += 4) {
		__m128i sse_off_col = sse_off;
		for (uint32_t j = 0; j < (ncols/4)*4; j += 4) {
			// Store indices
			_mm_store_si128((__m128i*) loff, sse_off_col);
			// Data load
			sse_row0 = _mm_loadu_ps(&data[loff[0]]);
			sse_row1 = _mm_loadu_ps(&data[loff[1]]);
			sse_row2 = _mm_loadu_ps(&data[loff[2]]);
			sse_row3 = _mm_loadu_ps(&data[loff[3]]);
			MM_TRANSPOSE4_PS_DST(sse_row0, sse_row1, sse_row2, sse_row3);
			// Store to the transposed matrix
			_mm_storeu_ps(&res[(j+0)*nrows+i], sse_row0);
			_mm_storeu_ps(&res[(j+1)*nrows+i], sse_row1);
			_mm_storeu_ps(&res[(j+2)*nrows+i], sse_row2);
			_mm_storeu_ps(&res[(j+3)*nrows+i], sse_row3);
			sse_off_col = _mm_add_epi32(sse_off_col, sse_4);
		}

		for (uint32_t j = (ncols/4)*4; j < ncols; j++) {
			__v4sf sse_final = _mm_set_ps(data[(i+3)*ncols+j], data[(i+2)*ncols+j], data[(i+1)*ncols+j], data[(i+0)*ncols+j]);
			_mm_storeu_ps(&res[j*nrows+i], sse_final);
		}
		sse_off = _mm_add_epi32(sse_off, sse_4ncols);
	}
	
	for (uint32_t i = (nrows/4)*4; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
			res[j*nrows+i] = data[i*ncols+j];
		}
	}
}

}
