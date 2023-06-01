//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/common.h>

#include <QApplication>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>

#include <iostream>
#include <chrono>

#include "common.h"

void show_qimage(QString const& title, QImage const& img)
{
	auto* dlg = new QDialog();
	dlg->setWindowTitle(title);
	auto* layout = new QVBoxLayout();
	auto* limg = new QLabel();
	limg->setPixmap(QPixmap::fromImage(img));
	layout->addWidget(limg);
	dlg->setLayout(layout);
	dlg->show();
}

PVParallelView::PVZoneRenderingBCI_p<10> new_zr(PVParallelView::PVBCIDrawingBackend& backend,
                                                size_t n,
                                                PVParallelView::PVBCIBackendImage_p& dst_img)
{
	dst_img = backend.create_image(20, 10);
	PVParallelView::PVZoneRenderingBCI_p<10> zr(new PVParallelView::PVZoneRenderingBCI<10>(
	    PVZoneID(0, 1),
	    [n](PVZoneID, PVCore::PVHSVColor const* colors_, PVParallelView::PVBCICode<10>* codes) {
		    for (size_t i = 0; i < n; i++) {
			    codes[i].int_v = 0;
			    codes[i].s.l = 0;
			    codes[i].s.r = i & MASK_INT_YCOORD;
			    codes[i].s.color = colors_[i].h();
			    codes[i].s.idx = i;
		    }
		    return n;
	    },
	    dst_img, 0, 20));
	return zr;
}

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

int main(int argc, char** argv)
{
#ifdef SQUEY_BENCH
	setenv("SQUEY_DEBUG_LEVEL", "FATAL", 1);

	/* 1K take 2 seconds on proto-03 with the CPU backend while 1M take 2.2 seconds on the same
	 * computer but using the GPU backend.
	 */

	size_t n = 1000;

	if (argc == 2) {
		n = std::atoll(argv[1]);
	}

#else
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nlines" << std::endl;
		return 1;
	}

	size_t n = std::atoll(argv[1]);
#endif

	PVParallelView::common::RAII_backend_init resources;
	TestEnv env(filename, fileformat);
	PVParallelView::PVLibView* pv = env.get_lib_view();
	PVParallelView::PVZonesManager& zm = pv->get_zones_manager();

	auto& backend = PVParallelView::common::backend();
	auto& pipeline = PVParallelView::common::pipeline();

	PVCore::PVHSVColor* colors = std::allocator<PVCore::PVHSVColor>().allocate(n);
	for (size_t i = 0; i < n; i++) {
		colors[i] = PVCore::PVHSVColor((i % (HSV_COLOR_RED.h() - HSV_COLOR_GREEN.h())) +
		                               HSV_COLOR_GREEN.h());
	}

	PVParallelView::PVZonesProcessor p = pipeline.declare_processor(
	    [](PVZoneID z) {
#ifndef SQUEY_BENCH
		    std::cout << "Preprocess for zone " << z << std::endl;
#else
		    (void)z;
#endif
	    },
	    colors, zm);

#define NJOBS 40
	std::vector<PVParallelView::PVZoneRenderingBCI_p<10>> zrs;
	std::vector<PVParallelView::PVBCIBackendImage_p> dimgs;
	zrs.reserve(NJOBS);
	dimgs.reserve(NJOBS);

#ifdef SQUEY_BENCH
	auto start = std::chrono::steady_clock::now();
#endif

	for (size_t i = 0; i < NJOBS; i++) {
		PVParallelView::PVBCIBackendImage_p dst_img;
		PVParallelView::PVZoneRenderingBCI_p<10> zr(new_zr(backend, n, dst_img));
		zrs.push_back(zr);
		dimgs.push_back(dst_img);
		p.add_job(zr);
	}

	for (size_t i = 0; i < NJOBS; i++) {
		zrs[i]->wait_end();
	}

#ifdef SQUEY_BENCH
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();
#endif

	return 0;
}
