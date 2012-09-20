#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <stdlib.h>
#include <iostream>

#include "common.h"

static Picviz::PVView_sp g_fake_view;
static Picviz::PVPlotted::uint_plotted_table_t g_norm_plotted;


static void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

Picviz::PVView_sp& get_view_sp() { return g_fake_view; }

PVParallelView::PVLibView* create_lib_view_from_args(int argc, char** argv)
{
	PVCol ncols;
	PVRow nrows;

	Picviz::PVPlotted::uint_plotted_table_t &norm_plotted = g_norm_plotted;
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		if (argc < 4) {
			usage(argv[0]);
			return NULL;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return NULL;
		}

		ncols = atol(argv[3]);

		Picviz::PVPlotted::plotted_table_t plotted;
		init_rand_plotted(plotted, nrows, ncols);
		Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
	}
	else
	{
		bool plotted_uint = false;
		if (argc >= 3) {
			plotted_uint = (argv[2][0] == '1');
		}

		if (plotted_uint) {
			if (!Picviz::PVPlotted::load_buffer_from_file(norm_plotted, nrows, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return NULL;
			}
		}
		else {
			Picviz::PVPlotted::plotted_table_t plotted;
			if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return NULL;
			}
			nrows = plotted.size()/ncols;
			Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
		}

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return NULL;
		}
	}

	//PVCore::PVHSVColor* colors = PVCore::PVHSVColor::init_colors(nrows);

	g_fake_view.reset(new Picviz::PVView());
	g_fake_view->reset_layers();
	g_fake_view->set_fake_axes_comb(ncols);

	PVParallelView::PVLibView* plib_view = PVParallelView::common::get_lib_view(*g_fake_view, norm_plotted, nrows, ncols);

	return plib_view;
}
