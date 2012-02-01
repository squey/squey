#ifndef PVGL_BZVIEW_H
#define PVGL_BZVIEW_H

#include <pvkernel/core/general.h>
#include <vector>


#pragma pack(push)
#pragma pack(4)
struct PVBCode
{
	uint16_t x;
	uint16_t y: 11;
	uint16_t type: 5;
};
#pragma pack(pop)

struct PVLineEq
{
	float a;
	float b;
	float c;
	inline float operator()(float x, float y) const { return a*x+b*y+c; }
};

class PVBZView
{
public:
	PVBZView();
public:
	void set_plotted(std::vector<float> const& plotted, PVCol ncols);
	void set_zoom(float zoom_x, float zoom_y);
	void compute_b(PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);

private:
	inline float get_plotted(PVCol col, PVRow row) const { return (*_plotted)[col*_nb_rows+row]; }
	vec2 plotted_to_frame(vec2 const& p) const;
	vec2 frame_to_plotted(vec2 const& f) const;
	inline int16_t get_distance_axes() const { return _zoom_x; }
	int get_line_type(PVLineEq const& l, int x0, int x1, int y0, int y1) const;

private:
	std::vector<float> const* _plotted;
	PVCol _nb_cols;
	PVRow _nb_rows;
	uint32_t _zoom_x;
	uint32_t _zoom_y;
	uint32_t _trans_x;
	uint32_t _trans_y;
};

#endif
