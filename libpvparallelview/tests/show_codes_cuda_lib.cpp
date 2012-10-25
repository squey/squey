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

#define WIDTH 1024
#define BBITS 10

typedef PVParallelView::PVZonesDrawing<BBITS> zones_drawing_t;

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
		std::cerr << "Usage: " << argv[0] << " nlines" << " [width]" << std::endl;
		return 1;
	}


	size_t width = WIDTH;
	if (argc >= 3) {
		width = atoll(argv[2]);
	}

	size_t n = atoll(argv[1]);

	PVCuda::init_cuda();
	PVParallelView::PVBCIDrawingBackendCUDA& backend_cuda = PVParallelView::PVBCIDrawingBackendCUDA::get();

	PVParallelView::PVBCICode<BBITS>* codes = PVParallelView::PVBCICode<BBITS>::allocate_codes(n);
	//PVParallelView::PVBCICode<BBITS>* codes;
	//picviz_verify_cuda(cudaHostAlloc(&codes, n*sizeof(PVParallelView::PVBCICode<BBITS>*), cudaHostAllocPortable | cudaHostAllocWriteCombined));
	PVParallelView::PVBCICode<BBITS>::init_random_codes(codes, n);

	PVParallelView::PVBCIBackendImage_p dst_img = backend_cuda.create_image(width, BBITS);
	PVParallelView::PVBCIBackendImage_p dst_img2 = backend_cuda.create_image(width, BBITS);

	backend_cuda(*dst_img, 0, width, (PVParallelView::PVBCICodeBase*) codes, n);
	backend_cuda(*dst_img2, 0, width, (PVParallelView::PVBCICodeBase*) codes, n);

	BENCH_START(b);
	backend_cuda.wait_all();
	BENCH_END(b, "wait_all", 1, 1, 1, 1);

	QImage img(dst_img->qimage());
	write(4, img.constBits(), img.height() * img.width() * sizeof(uint32_t));

	/*QApplication app(argc, argv);
	show_qimage("test", dst_img->qimage());
	app.exec();*/

	//picviz_verify_cuda(cudaFreeHost(codes));

	return 0;
}
