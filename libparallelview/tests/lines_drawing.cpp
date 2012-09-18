/**
 * \file lines_drawing.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVHSVColor.h>
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

typedef PVParallelView::PVZonesDrawing<10> zones_drawing_t;

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

void show_qimage(QString const& title, QImage const& img)
{
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

	PVCore::PVHSVColor* colors = PVCore::PVHSVColor::init_colors(nrows);

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	PVParallelView::PVZonesManager &zm = *(new PVParallelView::PVZonesManager());
	zm.set_uint_plotted(norm_plotted, nrows, ncols);
	zm.update_all();
	//zm.set_zone_width(0, 128);
	//zm.set_zone_width(1, 128);
	//zm.set_zone_width(2, 128);

	PVParallelView::PVBCIDrawingBackendCUDA<10> backend_cuda;
	zones_drawing_t &zones_drawing = *(new zones_drawing_t(zm, backend_cuda, *colors));

	zones_drawing_t::backend_image_p_t dst_img = zones_drawing.create_image(1024);
	zones_drawing.draw_zone(*dst_img, 0, 0, 128, &PVParallelView::PVZoneTree::browse_tree_bci);
	zones_drawing.draw_zone(*dst_img, 128+3, 1, 128, &PVParallelView::PVZoneTree::browse_tree_bci);
	zones_drawing.draw_zone(*dst_img, 128*2+3, 2, 128, &PVParallelView::PVZoneTree::browse_tree_bci);

	show_qimage("test", dst_img->qimage());

	/*PVParallelView::PVRenderingJob render_job;
	PVParallelView::PVLinesView lv(zones_drawing, 4);
	QFuture<void> the_future_is_here = lv.render_all_imgs(400, render_job);*/


	/*
	lv.translate(100, 400);
	lv.translate(260, 400);
	lv.translate(260*2, 400);
	lv.render_all(260*4, 400);
	lv.translate(0, 400);*/

	// Test concurrent drawing
	/*
	std::vector<PVParallelView::PVBCIBackendImage_p> imgs;
	std::vector<PVParallelView::PVBCIBackendImage*> imgs_p;
	imgs.resize(20); imgs_p.resize(20);
	for (int i = 0; i < 20; i++) {
		PVParallelView::PVBCIBackendImage_p img = zones_drawing.create_image(1024);
		imgs[i] = img;
		imgs_p[i] = img.get();
	}

	QFuture<void> the_future_is_here = zones_drawing.draw_zones_futur<PVParallelView::PVZoneTree>(imgs.begin(), 0, 20, &PVParallelView::PVZoneTree::browse_tree_bci);
	the_future_is_here.waitForFinished();

	for (int i = 0; i < 10; i++) {
		show_qimage(QString::number(i), imgs_p[i]->qimage());
	}*/

	app.exec();


	return 0;
}
