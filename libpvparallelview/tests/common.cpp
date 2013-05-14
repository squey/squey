#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <stdlib.h>
#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "common.h"

static Picviz::PVView_sp g_fake_view;
static Picviz::PVPlotted::uint_plotted_table_t g_norm_plotted;
static const char *extra_param_text = nullptr;
static int extra_param_num = 0;
static bool input_is_file = false;

static void init_rand_plotted(Picviz::PVPlotted::uint_plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	const PVRow nrows_aligned = ((nrows+3)/4)*4;
	p.resize(nrows_aligned*ncols);
	for (PVCol j = 0; j < ncols; j++) {
		for (PVRow i = 0; i < nrows; i++) {
			p[j*nrows_aligned+i] = (rand() << 1) | (rand()&1);
		}
	}
}

static void init_qt_plotted(Picviz::PVPlotted::uint_plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	p.clear();
	const PVRow nrows_aligned = ((nrows+3)/4)*4;
	p.resize(nrows_aligned*ncols);
	for (PVCol j = 0; j < (ncols/2)*2; j += 2) {
		for (PVRow i = 0; i < nrows; i++) {
			p[j*nrows_aligned+i] = 1<<31;;
			//p[j*nrows_aligned+i] = (1023-(i&1023))*(1<<22)+4;
		}
		for (PVRow i = 0; i < nrows; i++) {
			p[(j+1)*nrows_aligned+i] = i;
		}
	}
}

static void init_gauss_plotted(Picviz::PVPlotted::uint_plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	static boost::random::mt19937 rng;
	p.clear();
	const PVRow nrows_aligned = ((nrows+3)/4)*4;
	p.resize(nrows_aligned*ncols);
	for (PVCol j = 0; j < ncols; j++) {
		boost::random::normal_distribution<double> normd(1U<<31, 1U<<25);
		for (PVRow i = 0; i < nrows; i++) {
			p[j*nrows_aligned+i] = normd(rng);
		}
	}
}

void set_extra_param(int num, const char* usage_text)
{
	extra_param_num = num;
	extra_param_text = usage_text;
}

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " (plotted_file (0|1) | (0|1|2) nrows ncols)";
	if (extra_param_num != 0) {
		std::cerr << " " << extra_param_text;
	}
	std::cerr << std::endl;
}

int extra_param_start_at()
{
	if (input_is_file) {
		return 3;
	} else {
		return 4;
	}
}

bool input_is_a_file()
{
	return input_is_file;
}

Picviz::PVView_sp& get_view_sp() { return g_fake_view; }


bool create_plotted_table_from_args(Picviz::PVPlotted::uint_plotted_table_t &norm_plotted,
                                    PVRow &nrows, PVCol &ncols, int argc, char** argv)
{
	QString fplotted(argv[1]);
	if ((fplotted == "0") || (fplotted == "1") || (fplotted == "2")) {
		if (argc < (4 + extra_param_num)) {
			usage(argv[0]);
			return false;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return false;
		}

		ncols = atol(argv[3]);

		if (fplotted == "0") {
			init_rand_plotted(norm_plotted, nrows, ncols);
			//Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
		} else if (fplotted == "1") {
			init_qt_plotted(norm_plotted, nrows, ncols);
		} else {
			init_gauss_plotted(norm_plotted, nrows, ncols);
		}
		input_is_file = false;
	}
	else
	{
		if (argc < (3 + extra_param_num)) {
			usage(argv[0]);
			return false;
		}

		if (argv[2][0] == '1') {
			if (!Picviz::PVPlotted::load_buffer_from_file(norm_plotted, nrows, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return false;
			}
		}
		else {
			Picviz::PVPlotted::plotted_table_t plotted;
			if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return false;
			}
			nrows = plotted.size()/ncols;
			Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
		}

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return false;
		}

		input_is_file = true;
	}

	return true;
}

PVParallelView::PVLibView* create_lib_view_from_args(int argc, char** argv)
{
	PVCol ncols;
	PVRow nrows;

	Picviz::PVPlotted::uint_plotted_table_t &norm_plotted = g_norm_plotted;

	if (!create_plotted_table_from_args(norm_plotted, nrows, ncols, argc, argv)) {
		return NULL;
	}

	//PVCore::PVHSVColor* colors = PVCore::PVHSVColor::init_colors(nrows);

	g_fake_view.reset(new Picviz::PVView());
	g_fake_view->reset_layers();
	g_fake_view->set_fake_axes_comb(ncols);

	PVParallelView::PVLibView* plib_view = PVParallelView::common::get_lib_view(*g_fake_view, norm_plotted, nrows, ncols);

	return plib_view;
}
