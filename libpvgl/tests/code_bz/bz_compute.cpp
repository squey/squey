#include <code_bz/bz_compute.h>
#include <cassert>

#include <pvkernel/core/picviz_intrin.h>

int8_t types_from_line_pos[] = {-1, 2, 3, 1, -1, -1, 4, 0, -1, -1, -1, 5, -1, -1, -1, -1};

void PVBCode::to_pts(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const
{
	switch (s.type) {
	case 0:
		lx = 0; ly = s.l;
		rx = s.r; ry = 0;
		break;
	case 1:
		lx = 0; ly = s.l;
		rx = w; ry = s.r;
		break;
	case 2:
		lx = 0; ly = s.l;
		rx = s.r; ry = h;
		break;
	case 3:
		lx = s.l; ly = h;
		rx = w; ry = s.r;
		break;
	case 4:
		lx = s.l; ly = h;
		rx = s.r; ry = 0;
		break;
	case 5:
		lx = s.l; ly = 0;
		rx = w; ry = s.r;
		break;
	default:
		assert(false);
		break;
	}
}

PVBZCompute::PVBZCompute():
	_plotted(NULL),
	_nb_cols(0),
	_nb_rows(0),
	_zoom_x(1),
	_zoom_y(1),
	_trans_x(0),
	_trans_y(0)
{
}

void PVBZCompute::set_plotted(std::vector<float> const& plotted, PVCol ncols)
{
	_plotted = &plotted;
	_nb_cols = ncols;
	_nb_rows = plotted.size()/ncols;
}

void PVBZCompute::set_trans_plotted(std::vector<float> const& plotted, PVCol ncols)
{
	_trans_plotted = &plotted;
	_nb_cols = ncols;
	_nb_rows = plotted.size()/ncols;
}

void PVBZCompute::set_zoom(float zoom_x, float zoom_y)
{
	_zoom_x = zoom_x;
	_zoom_y = zoom_y;
}

vec2 PVBZCompute::plotted_to_frame(vec2 const& p) const
{
	return vec2(p.x*_zoom_x, p.y*_zoom_y) - vec2(_trans_x, _trans_y);
}

vec2 PVBZCompute::frame_to_plotted(vec2 const& p) const
{
	return vec2((p.x + _trans_x)/_zoom_x, (p.y + _trans_y)/_zoom_y);
}

int8_t PVBZCompute::get_line_type(PVLineEq const& l, float x0, float x1, float y0, float y1) const
{
	int a = l(x0, y1) >= 0;
	int b = l(x1, y1) >= 0;
	int c = l(x1, y0) >= 0;
	int d = l(x0, y0) >= 0;
	int lpos = a | b<<1 | c<<2 | d<<3;
	int8_t type = types_from_line_pos[lpos];
	return type;
}

void PVBZCompute::convert_to_points(uint16_t width, uint16_t height, std::vector<PVBCode> const& codes, std::vector<int>& ret)
{
	std::vector<PVBCode>::const_iterator it;
	ret.reserve(codes.size()*4);
	uint16_t lx,ly,rx,ry;
	for (it = codes.begin(); it != codes.end(); it++) {
		it->to_pts(width, height, lx, ly, rx, ry);
		ret.push_back(lx);
		ret.push_back(ly);
		ret.push_back(rx);
		ret.push_back(ry);
	}
}

void PVBZCompute::compute_b(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	codes.reserve(_nb_rows);
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted(axis_a, i);
		float ypr = get_plotted(axis_b, i);

		// Line equation
		// (ypl-ypr)*x - y + ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypl-ypr;
		l.c = -ypl;
		
		PVBCode bcode;
		int type = get_line_type(l, x0, x1, y0, y1);
		float bcode_l, bcode_r;
		
		switch (type) {
			case -1:
				// This line does not cross our region.
				//PVLOG_INFO("Line out of region (%f/%f)\n", ypl, ypr);
				continue;
			case 0:
				bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 1:
				bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
				bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
				break;
			case 2:
				bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
				bcode_r = ((y1-ypl)/(ypr-ypl))*_zoom_x - Xnorm;
				break;
			case 3:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = (ypr + (1-x1)*(ypl-ypr))*_zoom_y - Y0;
				break;
			case 4:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 5:
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

		codes.push_back(bcode);
	}
}

void PVBZCompute::compute_b_type_sse(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	codes.reserve(_nb_rows);
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted(axis_a, i);
		float ypr = get_plotted(axis_b, i);

		// Line equation
		// (ypl-ypr)*x - y + ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypl-ypr;
		l.c = -ypl;
		
		PVBCode bcode;

		//int type = get_line_type(l, x0, x1, y0, y1);

		float bcode_l, bcode_r;
		
		switch (type) {
			case -1:
				// This line does not cross our region.
				//PVLOG_INFO("Line out of region (%f/%f)\n", ypl, ypr);
				continue;
			case 0:
				bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 1:
				bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
				bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
				break;
			case 2:
				bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
				bcode_r = ((y1-ypl)/(ypr-ypl))*_zoom_x - Xnorm;
				break;
			case 3:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = (ypr + (1-x1)*(ypl-ypr))*_zoom_y - Y0;
				break;
			case 4:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 5:
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

		codes.push_back(bcode);
	}
}

int PVBZCompute::compute_b_trans(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted_trans(axis_a, i);
		float ypr = get_plotted_trans(axis_b, i);

		// Line equation
		// (ypl-ypr)*x + y - ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypl-ypr;
		l.c = -ypl;
		
		PVBCode bcode;
		int type = get_line_type(l, x0, x1, y0, y1);
		float bcode_l, bcode_r;
		
		switch (type) {
			case -1:
				// This line does not cross our region.
				//PVLOG_INFO("Line out of region (%f/%f)\n", ypl, ypr);
				continue;
			case 0:
				bcode_l = (x0*ypr + (1-x0)*ypl);
				bcode_r = ((ypl-y0)/(ypl-ypr));
				bcode_l = bcode_l*_zoom_y - Y0;
				bcode_r = bcode_r*_zoom_x - Xnorm;
				break;
			case 1:
				bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
				bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
				break;
			case 2:
				bcode_l = (x0*ypr + (1-x0)*ypl);
				bcode_r = ((y1-ypl)/(ypr-ypl));
				bcode_l = bcode_l*_zoom_y - Y0;
				bcode_r = bcode_r*_zoom_x - Xnorm;
				break;
			case 3:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = (ypr + (1-x1)*(ypl-ypr))*_zoom_y - Y0;
				break;
			case 4:
				bcode_l = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
				bcode_r = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
				break;
			case 5:
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

		codes.push_back(bcode);
	}
	return 0;
}

int PVBZCompute::compute_b_trans_nobranch(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted_trans(axis_a, i);
		float ypr = get_plotted_trans(axis_b, i);

		// Line equation
		// (ypl-ypr)*x + y - ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypl-ypr;
		l.c = -ypl;
		
		PVBCode bcode;
		int type = get_line_type(l, x0, x1, y0, y1);
		float bcode_l, bcode_r;
		if (type == -1) {
			continue;
		}
		bcode.int_v = 0;
		bcode.s.type = type;

		float bcodes[4];
		bcodes[0] = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
		bcodes[1] = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
		bcodes[2] = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
		bcodes[3] = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;

		bcode_l = bcodes[((type >= 3)*3) - ((type==5)<<1)];
		type++;
		bcode_r = bcodes[type - ((type>>2)<<1) - ((type/5)<<1)];
		
		/*
		switch (type) {
			case 0:
				bcode_l = bcodes[0];
				bcode_r = bcodes[1];
				break;
			case 1:
				bcode_l = bcodes[0];
				bcode_r = bcodes[2];
				break;
			case 2:
				bcode_l = bcodes[0];
				bcode_r = bcodes[3];
				break;
			case 3:
				bcode_l = bcodes[3];
				bcode_r = bcodes[2];
				break;
			case 4:
				bcode_l = bcodes[3];
				bcode_r = bcodes[1];
				break;
			case 5:
				bcode_l = bcodes[1];
				bcode_r = bcodes[2];
				break;
			default:
				assert(false);
				break;
		}*/
		bcode.s.l = (uint16_t) bcode_l;
		bcode.s.r = (uint16_t) bcode_r;

		codes.push_back(bcode);
	}
	return 0;
}

int PVBZCompute::compute_b_trans_nobranch_sse(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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

	__m128 sse_x0 = _mm_set1_ps(x0);
	__m128 sse_x0comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x0));
	__m128 sse_x1 = _mm_set1_ps(x1);
	__m128 sse_x1comp = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_set1_ps(x1));
	__m128 sse_y0 = _mm_set1_ps(y0);
	__m128 sse_y1 = _mm_set1_ps(y1);
	__m128 sse_Xnorm = _mm_set1_ps(Xnorm);
	__m128 sse_zoomx = _mm_set1_ps(_zoom_x);
	__m128 sse_zoomy = _mm_set1_ps(_zoom_y);
	__m128 sse_Y0 = _mm_set1_ps(Y0);

	__m128 sse_ypl, sse_ypr;
	for (PVRow i = 0; i < _nb_rows; i += 4) {
		sse_ypl = _mm_loadu_ps(&(*_trans_plotted)[axis_a*_nb_rows+i]);
		sse_ypr = _mm_loadu_ps(&(*_trans_plotted)[axis_b*_nb_rows+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		__m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		__m128i a = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl));
		__m128i b = _mm_slli_epi32(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), 1);
		__m128i c = _mm_slli_epi32(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl)), 2);
		__m128i d = _mm_slli_epi32(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl)), 3);
		__m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
		/*
		int a = l(x0, y1) >= 0;
		int b = l(x1, y1) >= 0;
		int c = l(x1, y0) >= 0;
		int d = l(x0, y0) >= 0;
		int lpos = a | b<<1 | c<<2 | d<<3;
		*/

		int DECLARE_ALIGN(16) pos[4];
		int DECLARE_ALIGN(16) types[4];
		_mm_store_si128((__m128i*) pos, sse_pos);
		// Pos -> types
		types[0] = types_from_line_pos[pos[0]];
		types[1] = types_from_line_pos[pos[1]];
		types[2] = types_from_line_pos[pos[2]];
		types[3] = types_from_line_pos[pos[3]];

		// Compute indexes for bcodes coordinates
		__m128i sse_idxes_left = _mm_load_si128((__m128i*) types);
		__m128i sse_idxes_right = sse_idxes_left;
		sse_idxes_left = _mm_mul_epi32(_mm_cmpgt_epi32(sse_idxes_left, _mm_set1_epi32(2)), _mm_set1_epi32(3));
		sse_idxes_left = _mm_sub_epi32(sse_idxes_left, _mm_srli_epi32(_mm_cmpeq_epi32(sse_idxes_right, _mm_set1_epi32(5)), 1));
		sse_idxes_right = _mm_add_epi32(sse_idxes_right, _mm_set1_epi32(1));
		sse_idxes_right = _mm_sub_epi32(sse_idxes_right, _mm_add_epi32(_mm_slli_epi32(_mm_srli_epi32(sse_idxes_right, 2), 1),
																					  _mm_slli_epi32(_mm_cmpgt_epi32(sse_idxes_right, _mm_set1_epi32(4)),1)));

		/*
		float DECLARE_ALIGN(16) bcodes_c[4];
		_mm_store_ps(bcodes_c, sse_bcodes_c);

		bcode_l = bcodes[((type >= 3)*3) - ((type==5)<<1)];
		type++;
		bcode_r = bcodes[type - ((type>>2)<<1) - ((type/5)<<1)];
		*/
		
		__m128 sse_bcodes_c[4];
		sse_bcodes_c[0] = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x0comp), _mm_mul_ps(sse_ypr, sse_x0)), sse_zoomy), sse_Y0);
		sse_bcodes_c[1] = _mm_sub_ps(_mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y0), ydiff), sse_zoomx), sse_Xnorm);
		sse_bcodes_c[2] = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(sse_ypl, sse_x1comp), _mm_mul_ps(sse_ypr, sse_x1)), sse_zoomy), sse_Y0);
		sse_bcodes_c[3] = _mm_sub_ps(_mm_mul_ps(_mm_div_ps(_mm_sub_ps(sse_ypl, sse_y1), ydiff), sse_zoomx), sse_Xnorm);
		/*
		float bcodes[4];
		bcodes[0] = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
		bcodes[1] = ((ypl-y0)/(ypl-ypr))*_zoom_x - Xnorm;
		bcodes[2] = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
		bcodes[3] = ((ypl-y1)/(ypl-ypr))*_zoom_x - Xnorm;
		*/

		PVBCode bcode;
		bcode.int_v = 0;
		for (int j = 0; j < 4; j++) {
			bcode.s.type = types[j];

			float DECLARE_ALIGN(16) bcodes_c[4];
			_mm_store_ps(bcodes_c, sse_bcodes_c[j]);
			float bcode_l = bcodes_c[_mm_extract_epi32(sse_idxes_left, j)];
			float bcode_r = bcodes_c[_mm_extract_epi32(sse_idxes_right, j)];
		
			bcode.s.l = (uint16_t) bcode_l;
			bcode.s.r = (uint16_t) bcode_r;

			codes.push_back(bcode);
		}

	}
	return 0;
}

int PVBZCompute::compute_b_trans_sse(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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

	__m128 sse_x0 = _mm_set1_ps(x0);
	__m128 sse_x1 = _mm_set1_ps(x1);
	__m128 sse_y0 = _mm_set1_ps(y0);
	__m128 sse_y1 = _mm_set1_ps(y1);

	__m128 sse_ypl, sse_ypr;
	PVBCode code;
#pragma omp parallel for private(code)
	for (PVRow i = 0; i < _nb_rows; i += 4) {
		sse_ypl = _mm_loadu_ps(&(*_trans_plotted)[axis_a*_nb_rows+i]);
		sse_ypr = _mm_loadu_ps(&(*_trans_plotted)[axis_b*_nb_rows+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		__m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		__m128i a = _mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl));
		__m128i b = _mm_slli_epi32(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), 1);
		__m128i c = _mm_slli_epi32(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl)), 2);
		__m128i d = _mm_slli_epi32(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl)), 3);
		__m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
		/*
		int a = l(x0, y1) >= 0;
		int b = l(x1, y1) >= 0;
		int c = l(x1, y0) >= 0;
		int d = l(x0, y0) >= 0;
		int lpos = a | b<<1 | c<<2 | d<<3;
		*/

		int types[4];
		// Pos -> types
		types[0] = types_from_line_pos[_mm_extract_epi32(sse_pos, 0)];
		types[1] = types_from_line_pos[_mm_extract_epi32(sse_pos, 0)];
		types[2] = types_from_line_pos[_mm_extract_epi32(sse_pos, 0)];
		types[3] = types_from_line_pos[_mm_extract_epi32(sse_pos, 0)];

		PVBCode bcode;
		bcode.int_v = 0;
		for (int j = 0; j < 4; j++) {
			float ypl = _mm_extract_ps(sse_ypl, j);
			float ypr = _mm_extract_ps(sse_ypl, j);
			float fydiff = _mm_extract_ps(ydiff, j);
			float bcode_l, bcode_r;
			switch (types[j]) {
				case -1:
					continue;
				case 0:
					bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
					bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
					break;
				case 1:
					bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
					bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
					break;
				case 2:
					bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
					bcode_r = ((y1-ypl)/(ypr-ypl))*_zoom_x - Xnorm;
					break;
				case 3:
					bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
					bcode_r = (ypr + (1-x1)*(fydiff))*_zoom_y - Y0;
					break;
				case 4:
					bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
					bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
					break;
				case 5:
					bcode_l = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
					bcode_r = (x1*ypr + (1-x1)*ypl)*_zoom_y - Y0;
					break;
				default:
					assert(false);
					break;
			}
			bcode.s.type = types[j];
			bcode.s.l = (uint16_t) bcode_l;
			bcode.s.r = (uint16_t) bcode_r;
			code = bcode;
			//codes.push_back(bcode);
		}

	}
	return code.int_v;
}
