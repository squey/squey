/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>

#include <iostream>
#include <chrono>

#include <QApplication>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>

#include "bci_helpers.h"

#define BBITS 10
#define WIDTH 1024
#define HEIGHT (1 << BBITS)

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

template <size_t Bbits>
PVParallelView::PVBCIBackendImage_p do_test(size_t n, size_t width, int pattern)
{
	PVParallelView::PVBCIDrawingBackendOpenCL& backend =
	    PVParallelView::PVBCIDrawingBackendOpenCL::get();

	PVParallelView::PVBCICode<Bbits>* codes = PVParallelView::PVBCICode<Bbits>::allocate_codes(n);
	PVParallelView::PVBCIPatterns<Bbits>::init_codes_pattern(codes, n, pattern);

	PVParallelView::PVBCIBackendImage_p dst_img = backend.create_image(width, Bbits);

	auto start = std::chrono::steady_clock::now();
	backend.render(dst_img, 0, width, (PVParallelView::PVBCICodeBase*)codes, n);
	backend.wait_all();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

	return dst_img;
}

int main(int argc, char** argv)
{
	size_t width = WIDTH;
	int pattern = 0;
	int bbits = BBITS;

#ifdef INSPECTOR_BENCH
	// we just want the measured time, not the backend information
	setenv("INENDI_DEBUG_LEVEL", "FATAL", 1);
	(void)argc;
	(void)argv;

	size_t n = HEIGHT * HEIGHT;
#else
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nlines"
		          << " [width] [pattern] [bbits]" << std::endl;
		const char* const* patterns = PVParallelView::PVBCIPatterns<BBITS>::get_patterns_string();
		std::cerr << "where pattern is one of the following:" << std::endl;
		for (int i = 0; i < PVParallelView::PVBCIPatterns<BBITS>::get_number_patterns(); i++) {
			std::cerr << i << "\t-\t" << patterns[i] << std::endl;
		}
		std::cerr << "and bbits is 10 (1024 image height) or 11 (2048 image height)." << std::endl;
		return 1;
	}

	if (argc >= 3) {
		width = atoll(argv[2]);
	}
	if (argc >= 4) {
		pattern = atoi(argv[3]);
	}
	if (argc >= 5) {
		bbits = atoi(argv[4]);
		if (bbits != 10 && bbits != 11) {
			std::cerr << "bbits must be 10 or 11 !" << std::endl;
			return 1;
		}
	}

	size_t n = atoll(argv[1]);
#endif

	PVParallelView::PVBCIBackendImage_p dst_img;
	switch (bbits) {
	case 10:
		dst_img = do_test<10>(n, width, pattern);
		break;
	case 11:
		dst_img = do_test<11>(n, width, pattern);
		break;
	}

#ifndef INSPECTOR_BENCH
	QImage img(dst_img->qimage());
	write(4, img.constBits(), img.height() * img.width() * sizeof(uint32_t));

	QApplication app(argc, argv);
	show_qimage("test", dst_img->qimage());
	app.exec();
#endif

	return 0;
}
