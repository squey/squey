#include <code_bz/bz_compute.h>
#include <cassert>

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/picviz_libdivide.h>

#define COMPUTE_TYPE_INT(Ypl, Ypr, Ydiff, X0, X0comp, X1comp, Xdiff, d_Xdiff, bcode_l, bcode_r)\
	uint32_t tmp;\
	switch (type) {\
		case -1:\
			continue;\
		case 0:\
			bcode_l = Ypr-Y0 + libdivide_u32_do((Ydiff)*(X0comp), &d_Xdiff);\
			bcode_r = (Xdiff*(Ypl-Y0))/(Ydiff) - X0;\
			break;\
		case 1:\
			tmp = Ypr-Y0; \
			bcode_l = tmp + libdivide_u32_do((Ydiff)*(X0comp), &d_Xdiff);\
			bcode_r = tmp + libdivide_u32_do((Ydiff)*(X1comp), &d_Xdiff);\
			break;\
		case 2:\
			bcode_l = Ypr-Y0 + libdivide_u32_do((Ydiff)*(X0comp), &d_Xdiff);\
			bcode_r = (Xdiff*(Ypl-Y1))/(Ydiff) - X0;\
			break;\
		case 3:\
			bcode_l = (Xdiff*(Ypl-Y1))/(Ydiff) - X0;\
			bcode_r = Ypr-Y0 + libdivide_u32_do((Ydiff)*(X1comp), &d_Xdiff);\
			break;\
		case 4:\
			tmp = Ypr-Y0; \
			bcode_l = tmp + libdivide_u32_do((Ydiff)*(X1comp), &d_Xdiff);\
			bcode_r = tmp + libdivide_u32_do((Ydiff)*(X0comp), &d_Xdiff);\
			break;\
		case 5:\
			bcode_l = (Xdiff*(Ypl-Y0))/(Ydiff) - X0;\
			bcode_r = Ypr-Y0 + libdivide_u32_do((Ydiff)*(X1comp), &d_Xdiff);\
			break;\
	}

static int8_t types_from_line_pos[] = {-1, 2, 3, 1, -1, -1, 4, 0, -1, -1, -1, 5, -1, -1, -1, -1};

int8_t PVBZCompute::get_line_type_int(PVLineEqInt const& l, int x0, int x1, int y0, int y1) const
{
	int a = l(x0, y1) >= 0;
	int b = l(x1, y1) >= 0;
	int c = l(x1, y0) >= 0;
	int d = l(x0, y0) >= 0;
	int lpos = a | b<<1 | c<<2 | d<<3;
	int8_t type = types_from_line_pos[lpos];
	return type;
}

int PVBZCompute::compute_b_trans_int(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, int X0, int X1, int Y0, int Y1)
{
	const unsigned int Xdiff = _zoom_x;
	const unsigned int X0comp = Xdiff - X0;
	const unsigned int X1comp = Xdiff - X1;

	PVLineEqInt l;
	l.b = Xdiff;
	int idx_code = 0;
	for (PVRow i = 0; i < _nb_rows; i++) {
		const float ypl = get_plotted_trans(axis_a, i);
		const float ypr = get_plotted_trans(axis_b, i);
		//const int Ypl = (int) (ypl*_zoom_y - _trans_y);
		//const int Ypr = (int) (ypr*_zoom_y - _trans_y);
		const int Ypl = (int) (ypl*_zoom_y);
		const int Ypr = (int) (ypr*_zoom_y);

		// Line equation
		// (ypl-ypr)*x + (Xdiff)*y - Xdiff*ypl = 0
		// a*X + b*Y + c = 0
		const int Ydiff = Ypl-Ypr;
		l.a = Ydiff;
		l.c = -Xdiff*Ypl;
		
		PVBCode bcode;
		const int type = get_line_type_int(l, X0, X1, Y0, Y1);
		int bcode_l, bcode_r;
		
		switch (type) {
			case -1:
				// This line does not cross our region.
				//PVLOG_INFO("Line out of region (%f/%f)\n", ypl, ypr);
				continue;
			case 0:
				bcode_l = Ypr-Y0 + ((Ydiff)*(X0comp))/Xdiff;
				bcode_r = (Xdiff*(Ypl-Y0))/(Ydiff) - X0;
				break;
			case 1:
				bcode_l = Ypr-Y0 + ((Ydiff)*(X0comp))/Xdiff;
				bcode_r = Ypr-Y0 + ((Ydiff)*(X1comp))/Xdiff;
				break;
			case 2:
				bcode_l = Ypr-Y0 + ((Ydiff)*(X0comp))/Xdiff;
				bcode_r = (Xdiff*(Ypl-Y1))/(Ydiff) - X0;
				break;
			case 3:
				bcode_l = (Xdiff*(Ypl-Y1))/(Ydiff) - X0;
				bcode_r = Ypr-Y0 + ((Ydiff)*(X1comp))/Xdiff;
				break;
			case 4:
				bcode_l = Ypr-Y0 + ((Ydiff)*(X1comp))/Xdiff;
				bcode_r = Ypr-Y0 + ((Ydiff)*(X0comp))/Xdiff;
				break;
			case 5:
				bcode_l = (Xdiff*(Ypl-Y0))/(Ydiff) - X0;
				bcode_r = Ypr-Y0 + ((Ydiff)*(X1comp))/Xdiff;
				break;
			default:
				assert(false);
				break;
		}
		bcode.int_v = 0;
		bcode.s.type = type;
		bcode.s.l = (uint16_t) bcode_l;
		bcode.s.r = (uint16_t) bcode_r;

		codes[idx_code] = bcode;
		idx_code++;
	}
	return idx_code;
}

int PVBZCompute::compute_b_trans_int_ld(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, int X0, int X1, int Y0, int Y1)
{
	// ld == libdivide
	const unsigned int Xdiff = _zoom_x;
	const unsigned int X0comp = Xdiff - X0;
	const unsigned int X1comp = Xdiff - X1;

	PVLineEqInt l;
	l.b = Xdiff;
	//codes.reserve(_nb_rows);
	libdivide::libdivide_u32_t d_Xdiff = libdivide::libdivide_u32_gen(Xdiff);
	int idx_code = 0;
	for (PVRow i = 0; i < _nb_rows; i++) {
		const float ypl = get_plotted_trans(axis_a, i);
		const float ypr = get_plotted_trans(axis_b, i);
		const int Ypl = (int) (ypl*_zoom_y - _trans_y);
		const int Ypr = (int) (ypr*_zoom_y - _trans_y);

		// Line equation
		// (ypl-ypr)*x + (Xdiff)*y - Xdiff*ypl = 0
		// a*X + b*Y + c = 0
		const int Ydiff = Ypl-Ypr;
		l.a = Ydiff;
		l.c = -Xdiff*Ypl;
		
		PVBCode bcode;
		const int type = get_line_type_int(l, X0, X1, Y0, Y1);
		int bcode_l, bcode_r;
		
		COMPUTE_TYPE_INT(Ypl, Ypr, Ydiff, X0, X0comp, X1comp, Xdiff, d_Xdiff, bcode_l, bcode_r);

		bcode.int_v = 0;
		bcode.s.type = type;
		bcode.s.l = (uint16_t) bcode_l;
		bcode.s.r = (uint16_t) bcode_r;

		codes[idx_code] = bcode;
		idx_code++;
	}
	return idx_code;
}

int PVBZCompute::compute_b_trans_sse_int(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, int X0, int X1, int Y0, int Y1)
{
	const unsigned int Xdiff = _zoom_x;
	const unsigned int X0comp = Xdiff-X0;
	const unsigned int X1comp = Xdiff-X1;

	const __m128i sse_X0 = _mm_set1_epi32(X0);
	//const __m128i sse_X0comp = _mm_sub_epi32(_mm_set1_epi32(Xdiff), _mm_set1_epi32(X0));
	const __m128i sse_X1 = _mm_set1_epi32(X1);
	//const __m128i sse_X1comp = _mm_sub_epi32(_mm_set1_epi32(Xdiff), _mm_set1_epi32(X1));
	const __m128i sse_Y0 = _mm_set1_epi32(Y0);
	const __m128i sse_Y1 = _mm_set1_epi32(Y1);

	//const __m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	const __m128 sse_zoomy = _mm_set1_ps(_zoom_y);
	const __m128i sse_Xdiff = _mm_set1_epi32((int) _zoom_x);

	const __m128i sse_Y0Xd = _mm_mullo_epi32(sse_Xdiff, sse_Y0);
	const __m128i sse_Y1Xd = _mm_mullo_epi32(sse_Xdiff, sse_Y1);

	const libdivide::libdivide_u32_t d_Xdiff = libdivide::libdivide_u32_gen(Xdiff);

	__m128i sse_Ypl, sse_Ypr;
	int idx_code = 0;
	for (PVRow i = 0; i < (_nb_rows/4)*4; i += 4) {
		__m128 sse_ypl, sse_ypr;
		sse_ypl = _mm_load_ps(&_trans_plotted[axis_a*_nb_rows+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[axis_b*_nb_rows+i]);

		sse_Ypl = _mm_cvtps_epi32(_mm_mul_ps(sse_ypl, sse_zoomy));
		sse_Ypr = _mm_cvtps_epi32(_mm_mul_ps(sse_ypr, sse_zoomy));

		// Line equation
		// (Ypl-Ypr)*x + (Xdiff)*Y - Xdiff*Ypl = 0
		// (Ypl-Ypr)*x + (Xdiff)*Y >= Xdiff*Ypl = 0

		const __m128i sse_Ydiff = _mm_sub_epi32(sse_Ypl, sse_Ypr);
		const __m128i sse_Yplcmp = _mm_sub_epi32(_mm_mullo_epi32(sse_Xdiff, sse_Ypl), _mm_set1_epi32(1));
		const __m128i a = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(_mm_mullo_epi32(sse_Ydiff, sse_X0), sse_Y1Xd), sse_Yplcmp), _mm_set1_epi32(1));
		const __m128i b = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(_mm_mullo_epi32(sse_Ydiff, sse_X1), sse_Y1Xd), sse_Yplcmp), _mm_set1_epi32(1<<1));
		const __m128i c = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(_mm_mullo_epi32(sse_Ydiff, sse_X1), sse_Y0Xd), sse_Yplcmp), _mm_set1_epi32(1<<2));
		const __m128i d = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(_mm_mullo_epi32(sse_Ydiff, sse_X0), sse_Y0Xd), sse_Yplcmp), _mm_set1_epi32(1<<3));
		const __m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));

		/*
		int a = l(x0, y1) >= 0;
		int b = l(x1, y1) >= 0;
		int c = l(x1, y0) >= 0;
		int d = l(x0, y0) >= 0;
		int lpos = a | b<<1 | c<<2 | d<<3;
		*/

		int DECLARE_ALIGN(16) types[4];
		// Pos -> types
		types[0] = types_from_line_pos[_mm_extract_epi32(sse_pos, 0)];
		types[1] = types_from_line_pos[_mm_extract_epi32(sse_pos, 1)];
		types[2] = types_from_line_pos[_mm_extract_epi32(sse_pos, 2)];
		types[3] = types_from_line_pos[_mm_extract_epi32(sse_pos, 3)];

		// No SSE possible here, do it by hand.
		PVBCode bcode;
		bcode.int_v = 0;
		// At 4, GCC does not unroll this loop, and _mm_extract_epi32 expect a constant... :/
		for (int j = 0; j < 3; j++) {
			const int Ypl = _mm_extract_epi32(sse_Ypl, j);
			const int Ypr = _mm_extract_epi32(sse_Ypr, j);
			const int Ydiff = _mm_extract_epi32(sse_Ydiff, j);
			const int type = types[j];
			int bcode_l, bcode_r;

			COMPUTE_TYPE_INT(Ypl, Ypr, Ydiff, X0, X0comp, X1comp, Xdiff, d_Xdiff, bcode_l, bcode_r);
		
			bcode.s.type = type;
			bcode.s.l = (uint16_t) bcode_l;
			bcode.s.r = (uint16_t) bcode_r;

			codes[idx_code] = bcode;
			idx_code++;
		}

		const int Ypl = _mm_extract_epi32(sse_Ypl, 3);
		const int Ypr = _mm_extract_epi32(sse_Ypr, 3);
		const int Ydiff = _mm_extract_epi32(sse_Ydiff, 3);
		const int type = types[3];
		int bcode_l, bcode_r;

		COMPUTE_TYPE_INT(Ypl, Ypr, Ydiff, X0, X0comp, X1comp, Xdiff, d_Xdiff, bcode_l, bcode_r);
	
		bcode.s.type = type;
		bcode.s.l = (uint16_t) bcode_l;
		bcode.s.r = (uint16_t) bcode_r;

		codes[idx_code] = bcode;
		idx_code++;
	}
	return idx_code;
}

int PVBZCompute::compute_b_trans_sse4_int(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
{
	// Convert box to plotted coordinates
	const unsigned int Xdiff = _zoom_x;
	const unsigned int X0comp = Xdiff-X0;
	const unsigned int X1comp = Xdiff-X1;

	const __m128i sse_X0 = _mm_set1_epi32(X0);
	const __m128i sse_X0comp = _mm_set1_epi32(X0comp);
	const __m128i sse_X1 = _mm_set1_epi32(X1);
	const __m128i sse_X1comp = _mm_set1_epi32(X1comp);
	const __m128i sse_Y0 = _mm_set1_epi32(Y0);
	const __m128i sse_Y1 = _mm_set1_epi32(Y1);

	const __m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	const __m128 sse_zoomy = _mm_set1_ps(_zoom_y);
	const __m128i sse_Xdiff = _mm_set1_epi32((int) _zoom_x);

	const __m128i sse_Y0Xd = _mm_mullo_epi32(sse_Xdiff, sse_Y0);
	const __m128i sse_Y1Xd = _mm_mullo_epi32(sse_Xdiff, sse_Y1);

	const libdivide::libdivide_u32_t d_Xdiff = libdivide::libdivide_u32_gen(Xdiff);

	__m128i sse_Ypl, sse_Ypr;
	int idx_code = 0;
	PVRow offa = axis_a*_nb_rows;
	PVRow offb = axis_b*_nb_rows;
	for (PVRow i = 0; i < (_nb_rows/4)*4; i += 4) {
		__m128 sse_ypl, sse_ypr;
		sse_ypl = _mm_load_ps(&_trans_plotted[offa+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[offb+i]);

		sse_Ypl = _mm_cvtps_epi32(_mm_mul_ps(sse_ypl, sse_zoomy));
		sse_Ypr = _mm_cvtps_epi32(_mm_mul_ps(sse_ypr, sse_zoomy));

		// Line equation
		// (Ypl-Ypr)*x + (Xdiff)*Y - Xdiff*Ypl = 0
		// (Ypl-Ypr)*x + (Xdiff)*Y >= Xdiff*Ypl = 0

		const __m128i sse_Ydiff = _mm_sub_epi32(sse_Ypl, sse_Ypr);
		const __m128i sse_Yplcmp = _mm_sub_epi32(_mm_mullo_epi32(sse_Xdiff, sse_Ypl), _mm_set1_epi32(1));
		const __m128i sse_mul0 = _mm_mullo_epi32(sse_Ydiff, sse_X0);
		const __m128i sse_mul1 = _mm_mullo_epi32(sse_Ydiff, sse_X1);
		const __m128i a = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(sse_mul0, sse_Y1Xd), sse_Yplcmp), _mm_set1_epi32(1));
		const __m128i b = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(sse_mul1, sse_Y1Xd), sse_Yplcmp), _mm_set1_epi32(1<<1));
		const __m128i c = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(sse_mul1, sse_Y0Xd), sse_Yplcmp), _mm_set1_epi32(1<<2));
		const __m128i d = _mm_and_si128(_mm_cmpgt_epi32(_mm_add_epi32(sse_mul0, sse_Y0Xd), sse_Yplcmp), _mm_set1_epi32(1<<3));
		const __m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));

		/*
		int a = l(x0, y1) >= 0;
		int b = l(x1, y1) >= 0;
		int c = l(x1, y0) >= 0;
		int d = l(x0, y0) >= 0;
		int lpos = a | b<<1 | c<<2 | d<<3;
		*/

		int DECLARE_ALIGN(16) types[4];
		// Pos -> types
		types[0] = types_from_line_pos[_mm_extract_epi32(sse_pos, 0)];
		types[1] = types_from_line_pos[_mm_extract_epi32(sse_pos, 1)];
		types[2] = types_from_line_pos[_mm_extract_epi32(sse_pos, 2)];
		types[3] = types_from_line_pos[_mm_extract_epi32(sse_pos, 3)];


		// Special cases when all the types are the same. Let's do this in SSE !!
		if ((types[0] == types[1]) && (types [1] == types[2]) && (types[2] == types[3])) {
			__m128i sse_types = _mm_load_si128((__m128i*) types);
			int type = types[0];
			__m128i sse_bcodes_li, sse_bcodes_ri;
			__m128i sse_tmp;
			switch (type) {
				case -1:
					continue;
				case 0:
					sse_bcodes_li = _mm_add_epi32(_mm_sub_epi32(sse_Ypr, sse_Y0), libdivide_u32_do_vector(_mm_mul_epi32(sse_Ydiff, sse_X0comp), &d_Xdiff));
					//bcode_r = (Xdiff*(Ypl-Y0))/(Ydiff) - X0;
					break;
				case 1:
					sse_tmp = _mm_sub_epi32(sse_Ypr, sse_Y0);
					sse_bcodes_li = _mm_add_epi32(sse_tmp, libdivide_u32_do_vector(_mm_mul_epi32(sse_Ydiff, sse_X0comp), &d_Xdiff));
					sse_bcodes_ri = _mm_add_epi32(sse_tmp, libdivide_u32_do_vector(_mm_mul_epi32(sse_Ydiff, sse_X1comp), &d_Xdiff));
					break;
				case 2:
					break;
				case 3:
					break;
				case 4:
					break;
				case 5:
					break;
			}
			__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(sse_bcodes_li, 3),
			                                     _mm_slli_epi32(sse_bcodes_ri, 14));
			__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);
			if ((idx_code & 3) == 0) {
				// We are style 16-byte aligned
				_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}
			else {
				// Un-aligned store, performance loss... :s
				// Check the difference between this and 4 _mm_stream_si32 !
				printf("Unaligned store!\n");
				_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}

			continue;
		}

		//__m128i sse_types = _mm_set_epi32(types[3], types[2], types[1], types[0]);
		// It seems to be faster, at this point, to load the memory pointed by types !!
		__m128i sse_types = _mm_load_si128((__m128i*) types);

		// Check if one of the type is -1.
		// If not, using some SSE optimisations !
		if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(-1)), _mm_set1_epi32(0xFFFFFFFF))) {
			__m128i sse_bcodes_li;
			__m128i sse_bcodes_ri;
			
#define MANUAL_BCODE_COMPUTE(j)\
			{\
				const int Ypl = _mm_extract_epi32(sse_Ypl, j);\
				const int Ypr = _mm_extract_epi32(sse_Ypr, j);\
				const int Ydiff = _mm_extract_epi32(sse_Ydiff, j);\
				const int type = types[j];\
				int bcode_l, bcode_r;\
				COMPUTE_TYPE_INT(Ypl, Ypr, Ydiff, X0, X0comp, X1comp, Xdiff, d_Xdiff, bcode_l, bcode_r);\
				sse_bcodes_li = _mm_insert_epi32(sse_bcodes_li, bcode_l, j);\
				sse_bcodes_ri = _mm_insert_epi32(sse_bcodes_ri, bcode_r, j);\
			}
			MANUAL_BCODE_COMPUTE(0);
			MANUAL_BCODE_COMPUTE(1);
			MANUAL_BCODE_COMPUTE(2);
			MANUAL_BCODE_COMPUTE(3);
			// Use sse to set the codes.
			// Format:
			//   * 3 bits: types
			//   * 11 bits: l
			//   * 11 bits: r
			//   * free = 0
			__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(sse_bcodes_li, 3),
			                                     _mm_slli_epi32(sse_bcodes_ri, 14));
			__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);
			if ((idx_code & 3) == 0) {
				// We are still 16-byte aligned
				_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}
			else {
				// Un-aligned store, performance loss... :s
				// Check the difference between this and 4 _mm_stream_si32 !
				_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}
		}
		else {
			int bcode_l, bcode_r;
			volatile PVBCode bcode;
			bcode.int_v = 0;
#define MANUAL_BCODE_COMPUTE2(j)\
			{\
				const int Ypl = _mm_extract_epi32(sse_Ypl, j);\
				const int Ypr = _mm_extract_epi32(sse_Ypr, j);\
				const int Ydiff = _mm_extract_epi32(sse_Ydiff, j);\
				const int type = types[j];\
				COMPUTE_TYPE_INT(Ypl, Ypr, Ydiff, X0, X0comp, X1comp, Xdiff, d_Xdiff, bcode_l, bcode_r);\
				bcode.s.type = type;\
				bcode.s.l = bcode_l;\
				bcode.s.r = bcode_r;\
			}
				/*codes[idx_code+j] = bcode;\*/
			MANUAL_BCODE_COMPUTE2(0);
			MANUAL_BCODE_COMPUTE2(1);
			MANUAL_BCODE_COMPUTE2(2);
			MANUAL_BCODE_COMPUTE2(3);
			//idx_code += 4;
		}
	}
	return idx_code;
}
