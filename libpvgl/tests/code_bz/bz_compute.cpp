#include <code_bz/bz_compute.h>
#include <cassert>

int8_t types_from_line_pos[] = {-1, 2, 3, 1, -1, -1, 4, 0, -1, -1, -1, 5, -1, -1, -1, -1};

void PVBCode::to_pts(uint16_t w, uint16_t h, uint16_t& lx, uint16_t& ly, uint16_t& rx, uint16_t& ry) const
{
	switch (type) {
	case 0:
		lx = 0; ly = l;
		rx = r; ry = 0;
		break;
	case 1:
		lx = 0; ly = l;
		rx = w; ry = r;
		break;
	case 2:
		lx = 0; ly = l;
		rx = r; ry = h;
		break;
	case 3:
		lx = l; ly = h;
		rx = w; ry = r;
		break;
	case 4:
		lx = l; ly = h;
		rx = r; ry = 0;
		break;
	case 5:
		lx = l; ly = 0;
		rx = w; ry = r;;
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

	PVLOG_INFO("x0=%f | x1=%f | y0=%f | y1=%f\n", x0, x1, y0, y1);

	PVLineEq l;
	l.b = 1.0f;
	codes.reserve(_nb_rows);
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted(axis_a, i);
		float ypr = get_plotted(axis_b, i);

		// Line equation
		// -(ypr-ypl)*x + y - ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypl-ypr;
		l.c = -ypl;
		
		PVBCode bcode;
		int type = get_line_type(l, x0, x1, y0, y1);
		bcode.type = type;
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
		bcode.l = (uint16_t) bcode_l;
		bcode.r = (uint16_t) bcode_r;

		codes.push_back(bcode);

		/*
		uint16_t lx,ly,rx,ry;
		bcode.to_pts(1024,1024,lx,ly,rx,ry);
		PVLOG_INFO("bzcode for %f %f: %d/%d/%d %d,%d/%d,%d\n", ypl, ypr, bcode_l, bcode_r, bcode.type, bcode.l, bcode.r, lx,ly, rx,ry);*/
	}
}
