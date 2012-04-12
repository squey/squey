#include <pvkernel/cuda/common.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include "bci_cuda.h"
#include <pvparallelview/simple_lines_int_view.h>

#include <iostream>

#define WIDTH 1024

#include <QApplication>
#include <QMainWindow>

void show_codes(PVParallelView::PVBCICode* codes, size_t n)
{
	QMainWindow* mw = new QMainWindow();
	mw->setWindowTitle("codes");
	SLIntView* v = new SLIntView(mw);
	v->set_size(1024, 1024);
	v->set_ortho(1, 1024);

	std::vector<int32_t>& pts = *(new std::vector<int32_t>);
	pts.reserve(n*4);
	for (size_t i = 0; i < n; i++) {
		PVParallelView::PVBCICode c = codes[i];
		pts.push_back(0); pts.push_back(c.s.l);
		pts.push_back(1); pts.push_back(c.s.r);
	}
	v->set_points(pts);
	mw->setCentralWidget(v);
	mw->resize(v->sizeHint());
	mw->show();
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nlines" << std::endl;
		return 1;
	}

	QApplication app(argc, argv);
	PVCuda::init_cuda();

	size_t n = atoll(argv[1]);

	PVParallelView::PVBCICode* codes = PVParallelView::PVBCICode::allocate_codes(n);
	PVParallelView::PVBCICode::init_random_codes(codes, n);

	uint32_t* img = new uint32_t[WIDTH*IMAGE_HEIGHT];

	show_codes_cuda(codes, n, WIDTH, img);

	write(4, img, WIDTH*IMAGE_HEIGHT*sizeof(uint32_t));
	delete [] img;

	show_codes(codes, n);
	app.exec();

	PVParallelView::PVBCICode::free_codes(codes);
	return 0;
}
