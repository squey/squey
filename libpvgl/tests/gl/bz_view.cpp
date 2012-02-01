#include <gl/bz_view.h>


int8_t types_from_line_pos = {-1, 2, 3, -1, 1, -1, 4, 0, -1, -1, -1, 5, -1, -1, -1, -1};

PVBZView::PVBZView():
	_zoom_x(1),
	_zoom_y(1),
	_trans_x(0),
	_trans_y(0),
	_nb_cols(0),
	_plotted(NULL)
{
}

void PVBZView::set_plotted(std::vector<float> const& plotted, PVCol ncols)
{
	_plotted = &plotted;
	_nb_cols = ncols;
	_nb_rows = plotted.size()/ncols;
}

void PVBZView::set_zoom(float zoom_x, float zoom_y)
{
	_zoom_x = zoom_x;
	_zoom_y = zoom_y;
}

vec2 PVBZView::plotted_to_frame(vec2 const& p) const
{
	return vec2(p.x*_zoom_x, p.y*_zoom_y) - vec2(_trans_x, _trans_y);
}

vec2 PVBZView::frame_to_plotted(vec2 const& p) const
{
	return vec2((p.x + _trans_x)/_zoom_x, (p.y + _trans_y)/_zoom_y);
}

int8_t PVBZView::get_line_type(PVLineEq const& l, float x0, float x1, float y0, float y1) const
{
	int a = l(x0, y0) >= 0;
	int b = l(x0, y1) >= 0;
	int c = l(x1, y0) >= 0;
	int d = l(x1, y1) >= 0;
	int lpos = a | b<<1 | c<<2 | d<<3;
	int8_t type = types_from_line_pos[lpos];
	return type;
}

void PVBZView::compute_b(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1)
{
	// Convert box to plotted coordinates
	vec2 frame_p_left = frame_to_plotted(vec2(X0, Y0));
	vec2 frame_p_right = frame_to_plotted(vec2(X0, Y0));
	float x0 = frame_p_left.x;
	float y0 = frame_p_left.y;
	float x1 = frame_p_right.x;
	float y1 = frame_p_right.y;
	x0 -= (int)x0;
	y0 -= (int)x0;

	PVLineEq l;
	l.b = -1.0f;
	for (PVRow i = 0; i < _nb_rows; i++) {
		float ypl = get_plotted(axis_a, i);
		float ypr = get_plotted(axis_b, i);

		// Line equation
		// (ypr-ypl)*x - y + ypl = 0
		// a*X + b*Y + c = 0
		l.a = ypr-ypl;
		l.c = ypl;
		
		PVBCode bcode;
		int type = get_line_type(l, x0, x1, y0, y1);
		bcode.type = type;
		float bcode_x, bcode_y;
		switch (type) {
			case -1:
				// This line does not cross our region.
				continue;
			case 0:
				bcode_x = x0*(ypr-ypl) + ypl;
				bcode_y = (ypl-y0)/(ypl-ypr);
				break;
			case 1:
				bcode_x = (ypl-ypr)*(1.0f-x0)+ypr;
				bcode_y = (ypl-ypr)*(1.0f-x1)+ypr;
				break;
			case 2:
				bcode_x = (ypl-y1)/(ypr-ypl);
				bcode_y = x0*ypr + (1-x0)*ypl;
				break;
			case 3:
				bcode_x = (ypl-y1)/(ypl-ypr);
				bcode_y = ypr + (1-x1)*(ypl-ypr);
				break;
			case 4:
				bcode_x = (ypl-y1)/(ypl-ypr);
				bcode_y = (ypl-y0)/(ypl-ypr);
				break;
			case 5:
				bcode_x = (ypl-y0)/(ypl-ypr);
				bcode_y = x1*ypr + (1-x1)*ypl;
				break;
			default:
				assert(false);
				break;
		}
		bcode.x = (uint16_t) (bcode_x*_zoom_x);
		bcode.y = (uint16_t) (bcode_y*_zoom_y);

		PVLOG_DEBUG("bzcode for %f %f: %d/%d/%d\n", ypl, ypr, bcode.type, bcode.x, bcode.y);
	}
}
