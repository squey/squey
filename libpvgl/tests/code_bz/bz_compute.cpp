#include <code_bz/bz_compute.h>
#include <cassert>

#include <pvkernel/core/picviz_intrin.h>

int8_t types_from_line_pos[] = {-1, 2, 3, 1, -1, -1, 4, 0, -1, -1, -1, 5, -1, -1, -1, -1};

#define _MM_INSERT_PS(r, fi, i)\
	_mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128((r)), (fi), (i)))

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

void PVBZCompute::set_plotted(Picviz::PVPlotted::plotted_table_t const& plotted, PVCol ncols)
{
	_plotted = &plotted[0];
	_nb_cols = ncols;
	_nb_rows = plotted.size()/ncols;
}

void PVBZCompute::set_trans_plotted(Picviz::PVPlotted::plotted_table_t const& plotted, PVCol ncols)
{
	_trans_plotted = &plotted[0];
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

void PVBZCompute::compute_b(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted(axis_a, i);
		float ypr = get_plotted(axis_b, i);

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

		codes[i] = bcode;
	}
}

int PVBZCompute::compute_b_trans(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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

		codes[i] = bcode;
	}
	return 0;
}

int PVBZCompute::compute_b_trans_nobranch(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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

		codes[i] = bcode;
	}
	return 0;
}

int PVBZCompute::compute_b_trans_nobranch_sse(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
		sse_ypl = _mm_load_ps(&_trans_plotted[axis_a*_nb_rows+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[axis_b*_nb_rows+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		__m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		__m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), _mm_set1_epi32(1));
		__m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), _mm_set1_epi32(1<<1));
		__m128i c = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl)), _mm_set1_epi32(1<<2));
		__m128i d = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl)), _mm_set1_epi32(1<<3));
		__m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
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

		// Compute indexes for bcodes coordinates
		__m128i sse_idxes_left = _mm_load_si128((__m128i*) types);
		__m128i sse_idxes_right = sse_idxes_left;
		sse_idxes_left = _mm_and_si128(_mm_cmpgt_epi32(sse_idxes_left, _mm_set1_epi32(2)), _mm_set1_epi32(3));
		sse_idxes_left = _mm_sub_epi32(sse_idxes_left, _mm_and_si128(_mm_cmpeq_epi32(sse_idxes_right, _mm_set1_epi32(5)), _mm_set1_epi32(2)));
		sse_idxes_right = _mm_add_epi32(sse_idxes_right, _mm_set1_epi32(1));
		sse_idxes_right = _mm_sub_epi32(sse_idxes_right, _mm_add_epi32(_mm_slli_epi32(_mm_srli_epi32(sse_idxes_right, 2), 1),
																					  _mm_and_si128(_mm_cmpgt_epi32(sse_idxes_right, _mm_set1_epi32(4)),_mm_set1_epi32(2))));

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

			codes[i+j] = bcode;
		}

	}
	return 0;
}

int PVBZCompute::compute_b_trans_sse(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	//PVBCode code;
	for (PVRow i = 0; i < _nb_rows; i += 4) {
		sse_ypl = _mm_load_ps(&_trans_plotted[axis_a*_nb_rows+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[axis_b*_nb_rows+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		__m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		__m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), _mm_set1_epi32(1));
		__m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), _mm_set1_epi32(1<<1));
		__m128i c = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl)), _mm_set1_epi32(1<<2));
		__m128i d = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl)), _mm_set1_epi32(1<<3));
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
		types[1] = types_from_line_pos[_mm_extract_epi32(sse_pos, 1)];
		types[2] = types_from_line_pos[_mm_extract_epi32(sse_pos, 2)];
		types[3] = types_from_line_pos[_mm_extract_epi32(sse_pos, 3)];

		__m128i sse_bcodes;
		PVBCode bcode;
		bcode.int_v = 0;
		for (int j = 0; j < 4; j++) {
			float ypl,ypr,fydiff;
			_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
			_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
			_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
			float bcode_l, bcode_r;
			switch (types[j]) {
				case -1:
					continue;
				case 0:
					bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
					bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
					break;
				case 1:
					bcode_l = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
					bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
					break;
				case 2:
					bcode_l = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
					bcode_r = ((y1-ypl)/(fydiff))*_zoom_x - Xnorm;
					break;
				case 3:
					bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
					bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
					break;
				case 4:
					bcode_l = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
					bcode_r = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
					break;
				case 5:
					bcode_l = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
					bcode_r = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
					break;
#ifndef NDEBUG
				default:
					assert(false);
					break;
#endif
			}
			bcode.s.type = types[j];
			bcode.s.l = (uint16_t) bcode_l;
			bcode.s.r = (uint16_t) bcode_r;
			sse_bcodes = _mm_insert_epi32(sse_bcodes, bcode.int_v, j);
			//code = bcode;
			//codes.push_back(bcode);
		}
		_mm_storeu_si128((__m128i*) &codes[i], sse_bcodes);
	}
	return 0;
}

int PVBZCompute::compute_b_trans_sse2(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	//PVBCode code;
	for (PVRow i = 0; i < _nb_rows; i += 4) {
		sse_ypl = _mm_load_ps(&_trans_plotted[axis_a*_nb_rows+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[axis_b*_nb_rows+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		__m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		__m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), _mm_set1_epi32(1));
		__m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), _mm_set1_epi32(1<<1));
		__m128i c = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl)), _mm_set1_epi32(1<<2));
		__m128i d = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl)), _mm_set1_epi32(1<<3));
		__m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
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

		__m128i sse_types = _mm_load_si128((__m128i*) types);

		// Check if one of the type is -1.
		// If not, using some SSE optimisations !
		if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(-1)), _mm_set1_epi32(0xFFFFFFFF))) {
			__m128 sse_bcodes_l;
			__m128 sse_bcodes_r;
			for (int j = 0; j < 4; j++) {
				float ypl,ypr,fydiff;
				_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
				_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
				_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
				union { int i; float f; } bcode_l, bcode_r;
				switch (types[j]) {
					case 0:
						bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
						bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 1:
						bcode_l.f = (ypl*(1-x0)+ypr*x0)*_zoom_y - Y0;
						bcode_r.f = (ypl*(1-x1)+ypr*x1)*_zoom_y - Y0;
						break;
					case 2:
						bcode_l.f = (x0*ypr + (1-x0)*ypl)*_zoom_y - Y0;
						bcode_r.f = ((y1-ypl)/(ypr-ypl))*_zoom_x - Xnorm;
						break;
					case 3:
						bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
						bcode_r.f = (ypr + (1-x1)*(fydiff))*_zoom_y - Y0;
						break;
					case 4:
						bcode_l.f = ((ypl-y1)/(fydiff))*_zoom_x - Xnorm;
						bcode_r.f = ((ypl-y0)/(fydiff))*_zoom_x - Xnorm;
						break;
					case 5:
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
			__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);
			_mm_store_si128((__m128i*) &codes[i], sse_bcodes);
		}
		else {
			PVLOG_WARN("One of the type is -1 !\n");
		}
	}
	return 0;
}

int PVBZCompute::compute_b_trans_sse3(PVBCode_ap codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
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
	//PVBCode code;
	for (PVRow i = 0; i < _nb_rows; i += 4) {
		sse_ypl = _mm_load_ps(&_trans_plotted[axis_a*_nb_rows+i]);
		sse_ypr = _mm_load_ps(&_trans_plotted[axis_b*_nb_rows+i]);

		// Line equation
		// a*X + b*Y + c = 0
		//
		// (ypl-ypr)*x + y - ypl = 0
		// or...
		// (ypl-ypr)*x + y >= ypl

		__m128 ydiff = _mm_sub_ps(sse_ypl, sse_ypr);
		__m128i a = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y1), sse_ypl)), _mm_set1_epi32(1));
		__m128i b = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y1), sse_ypl)), _mm_set1_epi32(1<<1));
		__m128i c = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x1), sse_y0), sse_ypl)), _mm_set1_epi32(1<<2));
		__m128i d = _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(_mm_add_ps(_mm_mul_ps(ydiff, sse_x0), sse_y0), sse_ypl)), _mm_set1_epi32(1<<3));
		__m128i sse_pos = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
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

		__m128i sse_types = _mm_load_si128((__m128i*) types);

		// Check if one of the type is -1.
		// If not, using some SSE optimisations !
		if (_mm_testz_si128(_mm_cmpeq_epi32(sse_types, _mm_set1_epi32(-1)), _mm_set1_epi32(0xFFFFFFFF))) {
			__m128 sse_bcodes_l;
			__m128 sse_bcodes_r;
			//__m128 sse_zoom_l, sse_zoom_r;
			//__m128i sse_trans_l, sse_trans_r;
			float DECLARE_ALIGN(16) zooms_l[4], zooms_r[4];
			int DECLARE_ALIGN(16) transs_l[4], transs_r[4];
			for (int j = 0; j < 4; j++) {
				float ypl,ypr,fydiff;
				union { int i; float f; } zoom_l, zoom_r;
				int trans_l, trans_r;
				_MM_EXTRACT_FLOAT(ypl, sse_ypl, j);
				_MM_EXTRACT_FLOAT(ypr, sse_ypr, j);
				_MM_EXTRACT_FLOAT(fydiff, ydiff, j);
				union { int i; float f; } bcode_l, bcode_r;
				switch (types[j]) {
					case 0:
						bcode_l.f = (x0*ypr + (1-x0)*ypl);
						bcode_r.f = ((ypl-y0)/(fydiff));
						zoom_l.f = _zoom_y; zoom_r.f = _zoom_x;
						trans_l = Y0; trans_r = Xnorm;
						break;
					case 1:
						bcode_l.f = (ypl*(1-x0)+ypr*x0);
						bcode_r.f = (ypl*(1-x1)+ypr*x1);
						zoom_l.f = _zoom_y; zoom_r.f = _zoom_y;
						trans_l = Y0; trans_r = Y0;
						break;
					case 2:
						bcode_l.f = (x0*ypr + (1-x0)*ypl);
						bcode_r.f = ((y1-ypl)/(ypr-ypl));
						zoom_l.f = _zoom_y; zoom_r.f = _zoom_x;
						trans_l = Y0; trans_r = Xnorm;
						break;
					case 3:
						bcode_l.f = ((ypl-y1)/(fydiff));
						bcode_r.f = (ypr + (1-x1)*(fydiff));
						zoom_l.f = _zoom_x; zoom_r.f = _zoom_y;
						trans_l = Xnorm; trans_r = Y0;
						break;
					case 4:
						bcode_l.f = ((ypl-y1)/(fydiff));
						bcode_r.f = ((ypl-y0)/(fydiff));
						zoom_l.f = _zoom_x; zoom_r.f = _zoom_y;
						trans_l = Xnorm; trans_r = Y0;
						break;
					case 5:
						bcode_l.f = ((ypl-y0)/(fydiff));
						bcode_r.f = (x1*ypr + (1-x1)*ypl);
						zoom_l.f = _zoom_x; zoom_r.f = _zoom_y;
						trans_l = Xnorm; trans_r = Y0;
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
				/*sse_zoom_l = _MM_INSERT_PS(sse_zoom_l, zoom_l.i, j);
				sse_zoom_r = _MM_INSERT_PS(sse_zoom_l, zoom_r.i, j);
				sse_trans_l = _mm_insert_epi32(sse_trans_l, trans_l, j);
				sse_trans_r = _mm_insert_epi32(sse_trans_r, trans_r, j);*/
				zooms_l[j] = zoom_l.f; zooms_r[j] = zoom_r.f;
				transs_l[j] = trans_l; transs_r[j] = trans_r;
			}
			// Compute zoom and translation
			//
			// Use sse to set the codes. Casting is also done in SSE, which gives different
			// results that the serial one (+/- 1 !).
			// Format:
			//   * 3 bits: types
			//   * 11 bits: l
			//   * 11 bits: r
			//   * free = 0
			__m128i sse_bcodes_lr = _mm_or_si128(_mm_slli_epi32(_mm_sub_epi32(_mm_cvtps_epi32(_mm_mul_ps(sse_bcodes_l, _mm_load_ps(zooms_l))), _mm_load_si128((__m128i*) transs_l)), 3),
			                                     _mm_slli_epi32(_mm_sub_epi32(_mm_cvtps_epi32(_mm_mul_ps(sse_bcodes_r, _mm_load_ps(zooms_r))), _mm_load_si128((__m128i*) transs_r)), 14));
			__m128i sse_bcodes = _mm_or_si128(sse_types, sse_bcodes_lr);
			_mm_store_si128((__m128i*) &codes[i], sse_bcodes);
		}
		else {
			PVLOG_WARN("One of the type is -1 !\n");
		}
	}
	return 0;
}
