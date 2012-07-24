/**
 * \file zoomed_zone_tree_render.cpp
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
	zm.set_zone_width(0, 256);
	zm.set_zone_width(1, 256);
	zm.set_zone_width(2, 256);
	zm.set_zone_width(3, 256);

	PVParallelView::PVBCIDrawingBackendCUDA backend_cuda;
	PVParallelView::PVZonesDrawing &zones_drawing = *(new PVParallelView::PVZonesDrawing(zm, backend_cuda, *colors));

	PVParallelView::PVBCIBackendImage_p dst_img1 = zones_drawing.create_image(1920);

#if 0
	// zones_drawing.draw_zone<PVParallelView::PVZoneTree>(*dst_img1, zm.get_zone_absolute_pos(0), 0, &PVParallelView::PVZoneTree::browse_tree_bci);
	// zones_drawing.draw_zone<PVParallelView::PVZoneTree>(*dst_img1, zm.get_zone_absolute_pos(1), 1, &PVParallelView::PVZoneTree::browse_tree_bci);

	zones_drawing.draw_zone_lambda<PVParallelView::PVZoneTree>
		(*dst_img1, zm.get_zone_absolute_pos(0), 0,
		 [&](PVParallelView::PVZoneTree const& zone_tree,
		     PVParallelView::PVHSVColor const* colors,
		     PVParallelView::PVBCICode* codes)
		 {
			 size_t num = zone_tree.browse_tree_bci(colors, codes);
			 std::cout << "ZT-0: num of codes: " << num << std::endl;
			 // for (unsigned i = 0; i < num; ++i) {
			 // 	 fdprintf(3, "%u %u %u %u\n", codes[i].s.l, codes[i].s.r, codes[i].s.idx, codes[i].s.color);
			 // }
			 return num;
		 });

	zones_drawing.draw_zone_lambda<PVParallelView::PVZoneTree>
		(*dst_img1, zm.get_zone_absolute_pos(1), 1,
		 [&](PVParallelView::PVZoneTree const& zone_tree,
		     PVParallelView::PVHSVColor const* colors,
		     PVParallelView::PVBCICode* codes)
		 {
			 size_t num = zone_tree.browse_tree_bci(colors, codes);
			 std::cout << "ZT-1: num of codes: " << num << std::endl;
			 // for (unsigned i = 0; i < num; ++i) {
			 // 	 fdprintf(3, "%u %u %u %u\n", codes[i].s.l, codes[i].s.r, codes[i].s.idx, codes[i].s.color);
			 // }
			 return num;
		 });

	show_qimage("test - zone tree", dst_img1->qimage());
#endif

#if 1
	uint32_t a = 0;
	uint32_t b = 0;

	// std::cout << "drawing area [" << a << ", " << b << "]" << std::endl;

	PVParallelView::PVBCIBackendImage_p dst_img2 = zones_drawing.create_image(1920);

	BENCH_START(col1);
	PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree = zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(0);
	zones_drawing.draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
		(zoomed_zone_tree, *dst_img2, zm.get_zone_absolute_pos(0), 256,
		 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
		     PVParallelView::PVHSVColor const* colors,
		     PVParallelView::PVBCICode* codes)
		 {
			 size_t num = zoomed_zone_tree.browse_tree_bci_by_y1(a, b, colors, codes);
			 std::cout << "ZZT-0: num of codes: " << num << std::endl;
			 // for (unsigned i = 0; i < num; ++i) {
			 // 	 printf("%u %u %u %u\n", codes[i].s.l, codes[i].s.r, codes[i].s.idx, codes[i].s.color);
			 // }
			 return num;
		 });
	BENCH_END(col1, "render col1", 1, 1, 1, 1);

	// "1 +" because position must be a power of 2
	a = 1 + (UINT32_MAX / 2);
	b = 0;

	// std::cout << "drawing area [" << a << ", " << b << "]" << std::endl;

	BENCH_START(col2);
	zones_drawing.draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
		(zoomed_zone_tree, *dst_img2, zm.get_zone_absolute_pos(1), 256,
		 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
		     PVParallelView::PVHSVColor const* colors,
		     PVParallelView::PVBCICode* codes)
		 {
			 size_t num = zoomed_zone_tree.browse_tree_bci_by_y1(a, b, colors, codes);
			 std::cout << "ZZT-1: num of codes: " << num << std::endl;
			 // for (unsigned i = 0; i < num; ++i) {
			 // 	 printf("%u %u %u %u\n", codes[i].s.l, codes[i].s.r, codes[i].s.idx, codes[i].s.color);
			 // }
			 return num;
		 });
	BENCH_END(col2, "render col2", 1, 1, 1, 1);

	a = 0;
	b = 2;

	// std::cout << "drawing area [" << a << ", " << b << "]" << std::endl;

	BENCH_START(col3);
	zones_drawing.draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
		(zoomed_zone_tree, *dst_img2, zm.get_zone_absolute_pos(2), 256,
		 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
		     PVParallelView::PVHSVColor const* colors,
		     PVParallelView::PVBCICode* codes)
		 {
			 size_t num = zoomed_zone_tree.browse_tree_bci_by_y1(a, b, colors, codes);
			 std::cout << "ZZT-1: num of codes: " << num << std::endl;
			 // for (unsigned i = 0; i < num; ++i) {
			 // 	 printf("%u %u %u %u\n", codes[i].s.l, codes[i].s.r, codes[i].s.idx, codes[i].s.color);
			 // }
			 return num;
		 });
	BENCH_END(col3, "render col3", 1, 1, 1, 1);

	a = 0;
	b = 3;

	// std::cout << "drawing area [" << a << ", " << b << "]" << std::endl;

	BENCH_START(col4);
	zones_drawing.draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
		(zoomed_zone_tree, *dst_img2, zm.get_zone_absolute_pos(3), 256,
		 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
		     PVParallelView::PVHSVColor const* colors,
		     PVParallelView::PVBCICode* codes)
		 {
			 size_t num = zoomed_zone_tree.browse_tree_bci_by_y1(a, b, colors, codes);
			 std::cout << "ZZT-1: num of codes: " << num << std::endl;
			 // for (unsigned i = 0; i < num; ++i) {
			 // 	 printf("%u %u %u %u\n", codes[i].s.l, codes[i].s.r, codes[i].s.idx, codes[i].s.color);
			 // }
			 return num;
		 });
	BENCH_END(col4, "render col4", 1, 1, 1, 1);

	show_qimage("test - zoomed zone tree", dst_img2->qimage());
#endif

	/*
	PVParallelView::PVLinesView lv(zones_drawing, 4);
	lv.render_all_imgs(400);
	lv.translate(100, 400);
	lv.translate(260, 400);
	lv.translate(260*2, 400);
	lv.render_all(260*4, 400);
	lv.translate(0, 400);
	*/

	// Test concurrent drawing
	// std::vector<PVParallelView::PVBCIBackendImage_p> imgs;
	// std::vector<PVParallelView::PVBCIBackendImage*> imgs_p;
	// imgs.resize(20); imgs_p.resize(20);
	// for (int i = 0; i < 20; i++) {
	// 	PVParallelView::PVBCIBackendImage_p img = zones_drawing.create_image(1024);
	// 	imgs[i] = img;
	// 	imgs_p[i] = img.get();
	// }

	// QFuture<void> the_future_is_here = zones_drawing.draw_zones_futur<PVParallelView::PVZoneTree>(imgs.begin(), 0, 20, &PVParallelView::PVZoneTree::browse_tree_bci);
	// the_future_is_here.waitForFinished();

	// for (int i = 0; i < 10; i++) {
	// 	show_qimage(QString::number(i), imgs_p[i]->qimage());
	// }

	app.exec();


	return 0;
}
