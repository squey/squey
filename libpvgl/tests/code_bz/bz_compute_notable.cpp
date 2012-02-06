#include <code_bz/bz_compute.h>
#include <cassert>

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/picviz_libdivide.h>

#define _MM_INSERT_PS(r, fi, i)\
	_mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128((r)), (fi), (i)))

void PVBCode::to_pts_new(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const
{
	switch (s.type) {
	case 5:
		lx = 0; ly = s.l;
		rx = s.r; ry = 0;
		break;
	case 3:
		lx = 0; ly = s.l;
		rx = w; ry = s.r;
		break;
	case 1:
		lx = 0; ly = s.l;
		rx = s.r; ry = h;
		break;
	case 2:
		lx = s.l; ly = h;
		rx = w; ry = s.r;
		break;
	case 4:
		lx = s.l; ly = h;
		rx = s.r; ry = 0;
		break;
	case 6:
		lx = s.l; ly = 0;
		rx = w; ry = s.r;
		break;
	default:
		assert(false);
		break;
	}
}

void PVBZCompute::convert_to_points_new(uint16_t width, uint16_t height, std::vector<PVBCode> const& codes, std::vector<int>& ret)
{
	std::vector<PVBCode>::const_iterator it;
	ret.reserve(codes.size()*4);
	uint16_t lx,ly,rx,ry;
	for (it = codes.begin(); it != codes.end(); it++) {
		it->to_pts_new(width, height, lx, ly, rx, ry);
		ret.push_back(lx);
		ret.push_back(ly);
		ret.push_back(rx);
		ret.push_back(ry);
	}
}

int8_t PVBZCompute::get_line_type_notable(PVLineEq const& l, float x0, float x1, float y0, float y1) const
{
	int a = l(x0, y1) >= 0;
	int b = l(x1, y1) >= 0;
	int c = l(x1, y0) >= 0;
	int d = l(x0, y0) >= 0;
	int r0 = a & !d;
	int r1 = b & !c;
	int r2 = c | d;
	int8_t type = (r0 | r1<<1 | r2<<2) & 7;
	return type;
}

int PVBZCompute::compute_b_trans_notable(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
{
	// Convert box to plotted coordinates
	vec2 frame_p_left = frame_to_plotted(vec2(X0, Y0));
	vec2 frame_p_right = frame_to_plotted(vec2(X1, Y1));
	float x0 = frame_p_left.x;
	float y0 = frame_p_left.y;
	float x1 = frame_p_right.x;
	float y1 = frame_p_right.y;
	x0 -= (int)x0;
	x1 -= (int)x0;
	float Xnorm = x0*_zoom_x;

	PVLineEq l;
	l.b = 1.0f;
	//codes.reserve(_nb_rows);
	int idx_code = 0;
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted_trans(axis_a, i);
		float ypr = get_plotted_trans(axis_b, i);

		// Line equation
		// (ypl-ypr)*x + y - ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypl-ypr;
		l.c = -ypl;
		
		PVBCode bcode;
		int type = get_line_type_notable(l, x0, x1, y0, y1);
		float bcode_l, bcode_r;
		
		switch (type) {
			case 0:
				continue;
			case 5:
				bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 3:
				bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
				bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
				break;
			case 1:
				bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
				bcode_r = ((y1-ypl)/(ypr-ypl))*_zoom_x - Xnorm;
				break;
			case 2:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
				break;
			case 4:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 6:
				bcode_l = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
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

int PVBZCompute::compute_b_trans_sse4_notable(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
{
	// Convert box to plotted coordinates
	vec2 frame_p_left = frame_to_plotted(vec2(X0, Y0));
	vec2 frame_p_right = frame_to_plotted(vec2(X1, Y1));
	float x0 = frame_p_left.x;
	float y0 = frame_p_left.y;
	float x1 = frame_p_right.x;
	float y1 = frame_p_right.y;
	x0 -= (int)x0;
	x1 -= (int)x0;
	float Xnorm = x0*_zoom_x;

	const __m128 sse_x0 = _mm_set1_ps(x0);
	const __m128 sse_x0comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x0));
	const __m128 sse_x1 = _mm_set1_ps(x1);
	const __m128 sse_x1comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x1));
	const __m128 sse_y0 = _mm_set1_ps(y0);
	const __m128 sse_y1 = _mm_set1_ps(y1);

	const __m128i sse_Y0 = _mm_set1_epi32(Y0);
	const __m128i sse_Xnorm = _mm_set1_epi32((int) Xnorm);

	const __m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	const __m128 sse_zoomy = _mm_set1_ps(_zoom_y);

	const __m128i sse_1i = _mm_set1_epi32(1);

	__m128 sse_ypl, sse_ypr;
	int idx_code = 0;
	const PVRow offa = axis_a*_nb_rows;
	const PVRow offb = axis_a*_nb_rows;
	const PVRow end = (_nb_rows*4)/4;
#pragma omp parallel for
	for (PVRow i = 0; i < end; i += 4) {
		sse_ypl = _mm_load_ps(&_trans_plotted[offa+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[offb+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		const __m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		const __m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), sse_1i);
		const __m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), sse_1i);
		const __m128i c = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl));
		const __m128i d = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl));
		// r0 = a & !d
		// r1 = b & !c
		// r2 = c | d
		const __m128i r0 = _mm_and_si128(a, _mm_andnot_si128(d, sse_1i));
		const __m128i r1 = _mm_and_si128(b, _mm_andnot_si128(c, sse_1i));
		const __m128i r2 = _mm_and_si128(_mm_or_si128(c, d), sse_1i);
		// types = r0 | r1 << 1 | r2 << 2
		const __m128i sse_types = _mm_or_si128(r0, _mm_or_si128(_mm_slli_epi32(r1, 1),
		                                                        _mm_slli_epi32(r2, 2)));

		// Special cases when all the types are the same. Let's do this in SSE !!
		int64_t dtypes_0 = _mm_extract_epi64(sse_types, 0);
		int64_t dtypes_1 = _mm_extract_epi64(sse_types, 1);
		if (dtypes_0 == dtypes_1) {
			// Take the first type.
			const uint32_t type = _mm_extract_epi32(sse_types, 0);
			__m128i sse_bcodes_li, sse_bcodes_ri;
			__m128 tmpl,tmpr;
			switch (type) {
				case 0:
					continue;
				case 5:
					tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
					tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

					sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
					sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
					break;
				case 3:
					tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
					tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

					sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
					sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
					break;
				case 1:
					tmpl = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy);
					tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);

					sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpl), sse_Y0);
					sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
					break;
				case 2:
					tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
					tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

					sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
					sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
					break;
				case 4:
					tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx);
					tmpr = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);

					sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
					sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
					break;
				case 6:
					tmpl = _mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx);
					tmpr = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy);

					sse_bcodes_li = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Xnorm);
					sse_bcodes_ri = _mm_sub_epi32(_mm_cvtps_epi32(tmpr), sse_Y0);
					break;
			}
			__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(sse_bcodes_li, 3),
			                                     _mm_slli_epi32(sse_bcodes_ri, 14));
			volatile __m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);
			/*
			if ((idx_code & 3) == 0) {
				// We are style 16-byte aligned
				_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}
			else {
				// Un-aligned store, performance loss... :s
				// Check the difference between this and 4 _mm_stream_si32 !
				_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}*/

			continue;
		}

		// Check if one of the type is 0.
		// If not, using some SSE optimisations !
		if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFF))) {
			__m128 sse_bcodes_l;
			__m128 sse_bcodes_r;
			for (int j = 0; j < 4; j++) {
				float ypl,ypr,fydiff;
				_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
				_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
				_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
				union { int i; float f; } bcode_l, bcode_r;
				int type = _mm_extract_epi32(sse_types, j);
				switch (type) {
					case 5:
						bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
						bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 3:
						bcode_l.f = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
						bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
						break;
					case 1:
						bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
						bcode_r.f = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 2:
						bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
						bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
						break;
					case 4:
						bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
						bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 6:
						bcode_l.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						bcode_r.f = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
						break;
#ifndef NDEBUG
					default:
						assert(false);
						break;
#endif
				}
				// _mm_insert_ps does not suit our purpose.
				// This will not work without optimisations, because this loop won't be unrolled,
				// and _mm_insert_epi32 waits for a constant !
				sse_bcodes_l = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_l), bcode_l.i, j));
				sse_bcodes_r = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(sse_bcodes_r), bcode_r.i, j));
			}
			// Use sse to set the codes. Casting is also done in SSE, which gives different
			// results that the serial one (+/- 1 !).
			// Format:
			//   * 3 bits: types
			//   * 11 bits: l
			//   * 11 bits: r
			//   * free = 0
			__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_l), 3),
			                                     _mm_slli_epi32(_mm_cvtps_epi32(sse_bcodes_r), 14));
			volatile __m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);
			/*
			if ((idx_code & 3) == 0) {
				// We are style 16-byte aligned
				_mm_stream_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}
			else {
				// Un-aligned store, performance loss... :s
				// Check the difference between this and 4 _mm_stream_si32 !
				_mm_storeu_si128((__m128i*) &codes[idx_code], sse_bcodes);
				idx_code += 4;
			}*/
		}
		else {
			// No SSE possible here, do it by hand.
			volatile PVBCode bcode;
			bcode.int_v = 0;
			for (int j = 0; j < 4; j++) {
				float ypl,ypr,fydiff,bcode_l,bcode_r;
				_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
				_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
				_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
				int type = _mm_extract_epi32(sse_types, j);
				switch (type) {
					case 0:
						continue;
					case 5:
						bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
						bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 3:
						bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
						bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
						break;
					case 1:
						bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
						bcode_r = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 2:
						bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
						bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
						break;
					case 4:
						bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
						bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 6:
						bcode_l = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						bcode_r = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
						break;
					default:
						assert(false);
						break;
				}
				bcode.s.type = type;
				bcode.s.l = (uint16_t) bcode_l;
				bcode.s.r = (uint16_t) bcode_r;

				/*codes[idx_code] = bcode;
				idx_code++;*/
			}
		}
	}
	return idx_code;
}
