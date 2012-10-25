#include <pvkernel/cuda/common.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>

#include <QApplication>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>

#include <tbb/tick_count.h>

#include <iostream>

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

PVParallelView::PVZoneRendering<10>* new_zr(PVParallelView::PVBCIDrawingBackend& backend, size_t n, PVParallelView::PVBCIBackendImage_p& dst_img)
{
	dst_img = backend.create_image(1024, 10);
	PVParallelView::PVZoneRendering<10>* zr = new PVParallelView::PVZoneRendering<10>(
		0,
		[n](PVCore::PVHSVColor* colors_, PVParallelView::PVBCICode<10>* codes)
		{
			for (size_t i = 0; i < n; i++) {
				codes[i].int_v = 0;
				codes[i].s.l = 0;
				codes[i].s.r = i & MASK_INT_YCOORD;
				codes[i].s.color = colors_[i].h();
				codes[i].s.idx = i;
			}
			return n;
		},
		*dst_img,
		0,
		1024);
	return zr;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nlines" << std::endl;
		return 1;
	}

	size_t n = picviz_min(atoll(argv[1]), PVParallelView::MaxBciCodes);

	PVCuda::init_cuda();
	PVParallelView::PVBCIDrawingBackendCUDA& backend = PVParallelView::PVBCIDrawingBackendCUDA::get();
	PVParallelView::PVRenderingPipeline* pipeline = new PVParallelView::PVRenderingPipeline(backend);

	PVCore::PVHSVColor* colors = std::allocator<PVCore::PVHSVColor>().allocate(n);
	for (size_t i = 0; i < n; i++) {
		colors[i] = (i%(HSV_COLOR_RED-HSV_COLOR_GREEN))+HSV_COLOR_GREEN;
	}

	PVParallelView::PVZonesProcessor p = pipeline->declare_processor([](PVZoneID z) { std::cout << "Preprocess for zone " << z << std::endl; }, colors, 2);

#define NJOBS 19
	std::vector<PVParallelView::PVZoneRendering<10>*> zrs;
	std::vector<PVParallelView::PVBCIBackendImage_p> dimgs;
	zrs.reserve(NJOBS);
	for (size_t i = 0; i < NJOBS; i++) {
		PVParallelView::PVBCIBackendImage_p dst_img;
		PVParallelView::PVZoneRendering<10>* zr = new_zr(backend, n, dst_img);
		zrs.push_back(zr);
		dimgs.push_back(dst_img);
		p.add_job(*zr);
	}

	for (size_t i = 0; i < NJOBS; i++) {
		zrs[i]->wait_end();
	}
	/*double time_cancel = 0.0;
	for (size_t i = NJOBS/2; i < NJOBS; i++) {
		zrs[i]->cancel();
		std::cout << zrs[i] << " canceled." << std::endl;
	}
	for (size_t i = 0; i < NJOBS; i++) {
		std::cout << "Waiting for " << zrs[i] << " to finished" << std::endl;
		tbb::tick_count start = tbb::tick_count::now();
		zrs[i]->wait_end();
		tbb::tick_count end = tbb::tick_count::now();
		time_cancel += (end-start).seconds();
	}
	std::cout << "Average cancelation time: " << (time_cancel*1000.0)/((double)NJOBS) << " ms." << std::endl;*/

	/*
	QApplication app(argc, argv);
	for (size_t i = 0; i < 15; i++) {
		PVParallelView::PVZoneRendering<10>* zr = zrs[i];
		if (zr->wait_end()) {
			show_qimage(QString::number(i), dimgs[i]->qimage());
		}
	}
	app.exec();*/

	// The pipeline must be deleted before the backend !
	delete pipeline;
	PVParallelView::PVBCIDrawingBackendCUDA::release();

	return 0;
}
