
#include <pvkernel/cuda/common.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>

#include <iostream>


#include <QApplication>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>

#include "zone_drawing_events.h"

static void show_qimage(QString const& title, QImage const& img)
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

void LinesViewMw::zr_sel_finished(int zid)
{
	PVParallelView::PVZoneRenderingBase* zr = qobject_cast<PVParallelView::PVZoneRenderingBase*>(sender());
	assert(zr);
	PVLOG_INFO("Sel zone rendering finished for zone %d\n", zr->zid());
	assert(zr->zid() == zid);
	show_qimage(QString("Sel %1").arg(zr->zid()), zr->dst_img().qimage());
}

void LinesViewMw::zr_bg_finished(int zid)
{
	PVParallelView::PVZoneRenderingBase* zr = qobject_cast<PVParallelView::PVZoneRenderingBase*>(sender());
	assert(zr);
	PVLOG_INFO("Bg zone rendering finished for zone %d\n", zr->zid());
	assert(zr->zid() == zid);
	show_qimage(QString("Bg %1").arg(zr->zid()), zr->dst_img().qimage());
}

static void init_rand_plotted(Picviz::PVPlotted::uint_plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	const PVRow nrows_aligned = ((nrows+3)/4)*4;
	p.resize(nrows_aligned*ncols);
	for (PVCol j = 0; j < ncols; j++) {
		for (PVRow i = 0; i < nrows; i++) {
			p[j*nrows_aligned+i] = (rand() << 1) | (rand()&1);
		}
	}
}

static void init_qt_plotted(Picviz::PVPlotted::uint_plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	p.clear();
	const PVRow nrows_aligned = ((nrows+3)/4)*4;
	p.resize(nrows_aligned*ncols);
	for (PVCol j = 0; j < (ncols/2)*2; j += 2) {
		for (PVRow i = 0; i < nrows; i++) {
			p[j*nrows_aligned+i] = 1<<22;
		}
		for (PVRow i = 0; i < nrows; i++) {
			p[(j+1)*nrows_aligned+i] = (i&1023)*(1<<22)+4;
		}
	}
}

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

int main(int argc, char** argv)
{
	PVCol ncols;
	PVRow nrows;

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	QString fplotted(argv[1]);
	if ((fplotted == "0") || (fplotted == "1")) {
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}

		ncols = atol(argv[3]);

		if (fplotted == "0") {
			init_rand_plotted(norm_plotted, nrows, ncols);
			//Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
		} else {
			init_qt_plotted(norm_plotted, nrows, ncols);
		}
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
				return 1;
			}
		}
		else {
			Picviz::PVPlotted::plotted_table_t plotted;
			if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return 1;
			}
			nrows = plotted.size()/ncols;
			Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
		}

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}
	}

	PVParallelView::PVZonesManager zm;
	zm.set_uint_plotted(norm_plotted, nrows, ncols);
	zm.update_all();

	PVCuda::init_cuda();
	PVParallelView::PVBCIDrawingBackendCUDA& backend = PVParallelView::PVBCIDrawingBackendCUDA::get();
	PVParallelView::PVRenderingPipeline* pipeline = new PVParallelView::PVRenderingPipeline(backend);

	PVCore::PVHSVColor* colors = std::allocator<PVCore::PVHSVColor>().allocate(nrows);
	for (size_t i = 0; i < nrows; i++) {
		colors[i] = (i%(HSV_COLOR_RED-HSV_COLOR_GREEN))+HSV_COLOR_GREEN;
	}

	Picviz::PVSelection sel;
	sel.select_none();
	for (size_t i = 0; i < 100; i++) {
		sel.set_bit_fast(i);
	}

	PVParallelView::PVZonesProcessor sel_processor = PVParallelView::PVZonesProcessor::declare_processor_zm_sel(*pipeline, zm, colors, sel);
	PVParallelView::PVZonesProcessor bg_processor = PVParallelView::PVZonesProcessor::declare_processor_direct(*pipeline, colors);

	PVParallelView::PVBCIBackendImage_p dst_img = backend.create_image(1024, 10);
	PVParallelView::PVZoneRendering<10>* zr = new PVParallelView::PVZoneRendering<10>(
		0,
		[&](PVZoneID z, PVCore::PVHSVColor const* colors_, PVParallelView::PVBCICode<10>* codes)
		{
			return zm.get_zone_tree<PVParallelView::PVZoneTree>(z).browse_tree_bci_sel(colors_, codes);
		},
		*dst_img,
		0,
		1024);

	PVParallelView::PVZoneRendering<10>* zr_bg = new PVParallelView::PVZoneRendering<10>(
		0,
		[&](PVZoneID z, PVCore::PVHSVColor const* colors_, PVParallelView::PVBCICode<10>* codes)
		{
			return zm.get_zone_tree<PVParallelView::PVZoneTree>(z).browse_tree_bci(colors_, codes);
		},
		*dst_img,
		0,
		1024);

	sel_processor.add_job(*zr);
	zr->wait_end();

	PVParallelView::PVBCIBackendImage_p dst_img2 = backend.create_image(1024, 10);
	zr->set_dst_img(*dst_img2);
	zr->reset();
	sel_processor.add_job(*zr);
	zr->wait_end();

	PVParallelView::PVBCIBackendImage_p dst_img4 = backend.create_image(1024, 10);
	sel.select_all();
	zr->set_dst_img(*dst_img4);
	zr->reset();
	sel_processor.invalidate_zone_preprocessing(0);
	sel_processor.add_job(*zr);
	zr->wait_end();

	PVParallelView::PVBCIBackendImage_p dst_img5 = backend.create_image(1024, 10);
	zr_bg->reset();
	zr_bg->set_dst_img(*dst_img5);
	bg_processor.add_job(*zr_bg);
	zr_bg->wait_end();

	QApplication app(argc, argv);

	LinesViewMw* lvmw = new LinesViewMw();
	PVParallelView::PVLinesView lines_view(backend, zm, sel_processor, bg_processor, lvmw);
	lines_view.render_all_zones_all_imgs(0, 1900, 1.0f);

	/*show_qimage("even1", dst_img->qimage());
	show_qimage("even2", dst_img2->qimage());
	show_qimage("all", dst_img4->qimage());
	show_qimage("all_direct", dst_img5->qimage());*/
	app.exec();

	delete pipeline;
	PVParallelView::PVBCIDrawingBackendCUDA::release();

}
