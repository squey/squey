#include <common/common.h>
#include <common/bench.h>
#include <gl/simple_lines_float_view.h>
#include <gl/simple_lines_int_view.h>
#include <code_bz/types.h>
#include <code_bz/init.h>
#include <picviz/PVPlotted.h>

#include <QApplication>
#include <QMainWindow>

#define MAX_INT ((int32_t)((1UL<<30) - 1))

typedef std::vector<uint32_t> int_plotted_t;
typedef std::vector<int32_t> pts_t;

void norm_int_plotted(Picviz::PVPlotted::plotted_table_t const& plotted, int_plotted_t& res)
{
	res.reserve(plotted.size());
	for (size_t i = 0; i < plotted.size(); i++) {
		res.push_back((int32_t) (plotted[i] * (float)MAX_INT));
	}

}

template <class F>
void filter_norm_plotted(pts_t& res, int_plotted_t plotted, PVRow nrows, PVCol axis_a, PVCol axis_b, F const& f)
{
	pts_t::value_type p1,p2;
	res.clear();
	res.reserve(nrows*2);
	for (size_t i = 0; i < nrows; i++) {
		p1 = plotted[axis_a*nrows + i];
		p2 = plotted[axis_b*nrows + i];

		if (f(p1, p2)) {
			res.push_back(0);
			res.push_back(p1);
			res.push_back(1);
			res.push_back(p2);
		}
	}
}

// From http://graphics.stanford.edu/~seander/bithacks.html
uint32_t interleave(uint16_t i1, uint16_t i2)
{
	static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
	static const unsigned int S[] = {1, 2, 4, 8};

	unsigned int z;
	unsigned int x = i1;
	unsigned int y = i2;
	// Interleave lower 16 bits of x and y, so the bits of x
	// are in the even positions and bits from y in the odd;
	// z gets the resulting 32-bit Morton Number.  
	// x and y must initially be less than 65536.

	x = (x | (x << S[3])) & B[3];
	x = (x | (x << S[2])) & B[2];
	x = (x | (x << S[1])) & B[1];
	x = (x | (x << S[0])) & B[0];

	y = (y | (y << S[3])) & B[3];
	y = (y | (y << S[2])) & B[2];
	y = (y | (y << S[1])) & B[1];
	y = (y | (y << S[0])) & B[0];

	z = x | (y << 1);
	return z;
}

inline uint64_t interleave(uint32_t i1, uint32_t i2)
{
	// Use the 16-bit version of interleave
	uint64_t low = interleave((uint16_t) i1, (uint16_t) i2);
	uint64_t high = interleave((uint16_t) (i1 >> 16), (uint16_t) (i2 >> 16));
	return low | high<<32;
}

bool f_filter_true(uint32_t /*p1*/, uint32_t /*p2*/)
{
	return true;
}

class FilterBandUp
{
public:
	// This will returns true for:
	//  * y1 >= k and y2 >= k
	//  * y1 >=k and y2 < k
	//  * y2 >=k and y1 < k
	FilterBandUp(uint32_t n): _n(n) { }
public:
	inline bool operator()(uint32_t y1, uint32_t y2) const
	{
		return (y1 >= _n && y2 >=_n) ||
		       (y1 > _n && y2 < _n) ||
			   (y2 > _n && y1 < _n);
	}
	inline uint32_t limit() const { return _n; }
private:
	uint32_t _n;
};

class InterleavedFilterBandUp
{
public:
	// This should returns true in the same conditions as FilterBandUp,
	// with n = 2**k.
	// The filtering is done here w/ y1 and y2 interlaced
	InterleavedFilterBandUp(uint32_t k)
	{
		assert(k >= 1);
		_mask = ~((1ULL<<(2*k))-1);
		_n = 1U<<k;
	}
public:
	inline bool operator()(uint32_t y1, uint32_t y2) const
	{
		uint64_t yi = interleave(y1, y2);
		return (yi & _mask) != 0;
	}
	inline uint32_t limit() const { return _n; }
private:
	uint32_t _n;
	uint64_t _mask;
};

class GenericInterleavedFilterBandUp
{
public:
	GenericInterleavedFilterBandUp(uint32_t n):
		_n(n)
	{
		_mask = interleave(n, 0);
	}
public:
	inline bool operator()(uint32_t y1, uint32_t y2) const
	{
		uint64_t yi = interleave(y1, y2);
		return yi >= _mask;
	}
	inline uint32_t limit() const { return _n; }
private:
	uint32_t _n;
	uint64_t _mask;
};

class InterleavedFilterBand
{
public:
	InterleavedFilterBand(uint32_t k1, uint32_t k2)
	{
		_n1 = 1ULL<<(2*k1);
		_n2 = 1ULL<<(2*k2) | 1ULL<<(2*k2+1);
		_k1 = k1;
		_k2 = k2;
		/*_mask_odd = interleave(0xFFFFFFFF, 0);
		_mask_even = interleave(0, 0xFFFFFFFF);
		_mask_sup = interleave(~((1U<<(k1))-1), ~((1U<<(k2))-1));*/
		_mask_k1 = interleave(~((1U<<(k1))-1), 0);
		_mask_k2 = interleave(0, ~((1U<<(k2))-1));
	}
public:
	inline bool operator()(uint32_t y1, uint32_t y2) const
	{
		uint64_t yi = interleave(y1, y2);
		/*uint64_t m = yi & _mask_sup;
		bool ca = (m & (_mask_odd)) != 0;
		bool cb = (m & (_mask_even)) == 0;*/
		bool ca = (m & (_mask_k1)) != 0;
		bool cb = (m & (_mask_k2)) == 0;
		return !(ca ^ cb);
	}
	inline uint32_t min() const { return 1U<<_k1; }
	inline uint32_t max() const { return 1U<<_k2; }
private:
	uint64_t _n1;
	uint64_t _n2;
	uint32_t _k1;
	uint32_t _k2;
	uint64_t _mask_sup;
	uint64_t _mask_even;
	uint64_t _mask_odd;
	uint64_t _mask_k1;
	uint64_t _mask_k2;
};

class FilterBand
{
public:
	FilterBand(uint32_t n1, uint32_t n2): _n1(n1), _n2(n2) { }
public:
	inline bool operator()(uint32_t y1, uint32_t y2) const
	{
		return (y1 > _n1 && y2 < _n2) ||
		       (y1 < _n2 && y2 > _n1) ||
			   (y1 >= _n1 && y1 <= _n2 && y2 >= _n1 && y2 <= _n2);
	}
	inline uint32_t min() const { return _n1; }
	inline uint32_t max() const { return _n2; }
private:
	uint32_t _n1;
	uint32_t _n2;
};

void add_horizontal_line(pts_t& pts, int32_t y)
{
	pts.push_back(0); pts.push_back(y);
	pts.push_back(1); pts.push_back(y);
}

void show_res(QString const& name, pts_t const& pts)
{
	// Showing filtering result
	QMainWindow *window = new QMainWindow();
	window->setWindowTitle(name);
	SLIntView *v = new SLIntView(window);

	v->set_size(1024, 1024);
	v->set_ortho(1, MAX_INT);
	v->set_points(pts);

	window->setCentralWidget(v);
	window->resize(v->sizeHint());
	window->show();
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " plotted_file" << std::endl;
		return 1;
	}
	QApplication app(argc, argv);

	PVCol ncols;
	Picviz::PVPlotted::plotted_table_t plotted;
	if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
		std::cerr << "Unable to load plotted !" << std::endl;
		return 1;
	}

	int_plotted_t norm_plotted;
	norm_int_plotted(plotted, norm_plotted);

	PVRow nrows = plotted.size()/ncols;
	pts_t pts_all, pts_band_up, pts_band_up_inter, pts_band_up_inter_gen, pts_band, pts_band_inter;
	// All points (test)
	filter_norm_plotted(pts_all, norm_plotted, nrows, 2, 3, f_filter_true);

	// Filter band up
	FilterBandUp fup(1<<29);
	//FilterBandUp fup(MAX_INT - MAX_INT/4);
	filter_norm_plotted(pts_band_up, norm_plotted, nrows, 2, 3, fup);
	add_horizontal_line(pts_band_up, fup.limit());

	// Filter band up, interleaved
	InterleavedFilterBandUp fup_inter(29);
	filter_norm_plotted(pts_band_up_inter, norm_plotted, nrows, 2, 3, fup_inter);
	add_horizontal_line(pts_band_up_inter, fup_inter.limit());

	//GenericInterleavedFilterBandUp fup_inter_gen(1<<29);
	GenericInterleavedFilterBandUp fup_inter_gen(1<<29);
	filter_norm_plotted(pts_band_up_inter_gen, norm_plotted, nrows, 2, 3, fup_inter_gen);
	add_horizontal_line(pts_band_up_inter_gen, fup_inter_gen.limit());

	FilterBand fb(1<<24, 1<<27);
	filter_norm_plotted(pts_band, norm_plotted, nrows, 2, 3, fb);
	add_horizontal_line(pts_band, fb.min());
	add_horizontal_line(pts_band, fb.max());

	InterleavedFilterBand fb_inter(24, 27);
	filter_norm_plotted(pts_band_inter, norm_plotted, nrows, 2, 3, fb_inter);
	add_horizontal_line(pts_band_inter, fb_inter.min());
	add_horizontal_line(pts_band_inter, fb_inter.max());

	show_res("all lines", pts_all);
	//show_res("band up (2**29)", pts_band_up);
	//show_res("band up interleaved (2**29)", pts_band_up_inter);
	//show_res("band up interleaved generic (2**29)", pts_band_up_inter_gen);
	show_res("band (2**29 -> 2**30)", pts_band);
	show_res("band inter (2**29 -> 2**30)", pts_band_inter);

	return app.exec();
}
