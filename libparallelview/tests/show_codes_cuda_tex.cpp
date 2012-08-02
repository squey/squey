/**
 * \file show_codes_cuda_tex.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/cuda/common.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVBCICode.h>

#define USE_GL_WITH_CUDA
#include "bci_cuda.h"

#include <pvparallelview/lines_buffer_view.h>

#include <GL/gl.h>

#include <iostream>

#define WIDTH 1024

#include <QApplication>
#include <QMainWindow>

#include <QGLBuffer>

void init_codes(LBView* v, PVParallelView::PVBCICode* codes, size_t n)
{
	v->set_size(WIDTH, 1024);
	v->set_ortho(1, 1024);

	std::vector<int32_t>& pts = *(new std::vector<int32_t>);
	std::vector<PVRGB>& colors = *(new std::vector<PVRGB>);
	pts.reserve(n*4);
	colors.reserve(n);
	PVRGB rgb;
	rgb.int_v = 0;
	for (size_t i = 0; i < n; i++) {
		PVParallelView::PVBCICode c = codes[i];
		pts.push_back(0); pts.push_back(c.s.l);
		pts.push_back(1); pts.push_back(c.s.r);

		PVParallelView::PVHSVColor hsv(c.s.color);
		hsv.to_rgb((uint8_t*) &rgb);
		colors.push_back(rgb);
	}
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

	srand(0);

	QApplication app(argc, argv);

	QMainWindow* mw = new QMainWindow();
	mw->setWindowTitle("codes");
	LBView* v = new LBView(mw, width, IMAGE_HEIGHT);

	PVCuda::init_gl_cuda();

	size_t n = atoll(argv[1]);

	PVParallelView::PVBCICode* codes = PVParallelView::PVBCICode::allocate_codes(n);
	PVParallelView::PVBCICode::init_random_codes(codes, n);

	v->set_size(WIDTH, 1024);
	v->set_ortho(1, 1024);

	show_codes_cuda(codes, n, width, v->get_buffer_id());

	cudaDeviceReset();

	init_codes(v, codes, n);
	mw->setCentralWidget(v);
	mw->resize(v->sizeHint());
	mw->show();


	app.exec();

	PVParallelView::PVBCICode::free_codes(codes);
	return 0;
}
