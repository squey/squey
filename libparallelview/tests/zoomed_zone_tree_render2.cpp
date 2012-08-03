/**
 * \file zoomed_zone_tree_render2.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVTools.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <QApplication>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void fdprintf(int fd, const char *format, ...)
{
	char buffer [2048];

	va_list ap;
	va_start(ap, format);
	(void) vsnprintf (buffer, 2048, format, ap);
	va_end(ap);

	(void) write (fd, buffer, strlen(buffer));
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
#if 1
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
#else
	for (PVCol j = 0; j < ncols; ++j) {
		for (PVRow i = 0; i < nrows; i++) {
			p.push_back((32. * i) / 1024.);
			// p.push_back((32. * i + .25) / 1024.);
			// p.push_back((32. * i + 0.5) / 1024.);
		}
	}
#endif
}

void show_qimage(QString const& title, QImage const& img)
{
	// QPainter p(img);
	// p.drawLine(QPoint(0, 0), QPoint(0, 512));
	// p.end();

	QDialog* dlg = new QDialog();
	dlg->setWindowTitle(title);
	QVBoxLayout* layout = new QVBoxLayout();
	QLabel* limg = new QLabel();
	limg->setPixmap(QPixmap::fromImage(img));
	layout->addWidget(limg);
	dlg->setLayout(layout);
	dlg->show();
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVCol ncols, nrows;
	Picviz::PVPlotted::plotted_table_t plotted;
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);
		ncols = atol(argv[3]);

		if (ncols < 3) {
			ncols = 3;
		}

		init_rand_plotted(plotted, nrows, ncols);
	}
	else
	{
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;
	}

	PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(nrows);

	for(int i = 0; i < nrows; ++i) {
		colors[i] = (i*20) % 192;
	}
	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	PVParallelView::PVZonesManager &zm = *(new PVParallelView::PVZonesManager());
	zm.set_uint_plotted(norm_plotted, nrows, ncols);
	zm.update_all();

	PVParallelView::PVBCIDrawingBackendCUDA backend_cuda;
	PVParallelView::PVZonesDrawing &zones_drawing = *(new PVParallelView::PVZonesDrawing(zm, backend_cuda, *colors));

	PVParallelView::PVBCIBackendImage_p dst_img1 = zones_drawing.create_image(1920);

	uint32_t p = 0;
	uint32_t z = 0;

	for (int i = 0; i < 4; ++i) {
		PVParallelView::PVBCIBackendImage_p dst_img = zones_drawing.create_image(512);
		std::cout << "drawing area: " << p << " (" << z << ")" << std::endl;

		BENCH_START(col);
#if 1
		zones_drawing.draw_zoomed_zone(*dst_img, p, z, 0,
		                               &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1);
#else
		PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree = zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(0);
		zones_drawing.draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
			(zoomed_zone_tree, *dst_img, 0, 512,
			 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
			     PVParallelView::PVHSVColor const* colors,
			     PVParallelView::PVBCICode<NBITS_INDEX>* codes)
			 {
				 size_t num = zoomed_zone_tree.browse_tree_bci_by_y1(p, z, colors, codes);
				 std::cout << "ZZT-0: num of codes: " << num << std::endl;
				 return num;
			 });
#endif
		BENCH_END(col, "render col", 1, 1, 1, 1);
		show_qimage("test - zoomed zone tree" + QString::number(i), dst_img->qimage());
		++z;
	}

	app.exec();


	return 0;
}
