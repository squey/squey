#ifndef PVGL_BZVIEW_H
#define PVGL_BZVIEW_H

#include <pvkernel/core/general.h>
#include "types.h"
#include <vector>

struct PVLineEq
{
	float a;
	float b;
	float c;
	inline float operator()(float x, float y) const { return a*x+b*y+c; }
};

class PVBZCompute
{
public:
	PVBZCompute();
public:
	void set_plotted(std::vector<float> const& plotted, PVCol ncols);
	void set_trans_plotted(std::vector<float> const& trans_plotted, PVCol ncols);
	void set_zoom(float zoom_x, float zoom_y);
	void compute_b(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	int compute_b_trans(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	int compute_b_trans_nobranch(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	int compute_b_trans_nobranch_sse(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	int compute_b_trans_sse(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	int compute_b_trans_sse2(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	int compute_b_trans_sse3(std::vector<PVBCode>& codes, PVCol axis_a, PVCol axis_b, float X0, float X1, float Y0, float Y1);
	void convert_to_points(uint16_t width, uint16_t height, std::vector<PVBCode> const& codes, std::vector<int>& ret);
	inline PVRow get_nrows() const { return _nb_rows; }

private:
	inline float get_plotted(PVCol col, PVRow row) const { return (*_plotted)[row*_nb_cols+col]; }
	inline float get_plotted_trans(PVCol col, PVRow row) const { return (*_trans_plotted)[col*_nb_rows+row]; }
	vec2 plotted_to_frame(vec2 const& p) const;
	vec2 frame_to_plotted(vec2 const& f) const;
	inline int16_t get_distance_axes() const { return _zoom_x; }
	int8_t get_line_type(PVLineEq const& l, float x0, float x1, float y0, float y1) const;

private:
	std::vector<float> const* _plotted;
	std::vector<float> const* _trans_plotted;
	PVCol _nb_cols;
	PVRow _nb_rows;
	uint32_t _zoom_x;
	uint32_t _zoom_y;
	uint32_t _trans_x;
	uint32_t _trans_y;
};

#define BCOMPUTE_BENCH_START

#endif
