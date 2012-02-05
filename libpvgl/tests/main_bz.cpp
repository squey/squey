#include <common/common.h>
#include <common/bench.h>
#include <code_bz/bz_compute.h>
#include <code_bz/serial_bcodecb.h>
#include <gl/bccb_view.h>
#include <gl/simple_lines_float_view.h>
#include <gl/simple_lines_int_view.h>

#include <picviz/PVPlotted.h>

#include <tbb/tick_count.h>

#include <QApplication>
#include <QMainWindow>

#define W_FRAME 1024
#define H_FRAME 1024

#define X_START 0
#define Y_START 0

void extract_plotted(PVRow ncols, std::vector<float> const& plotted, std::vector<float>& ret, PVCol axis_a, PVCol axis_b)
{
	PVRow nrows = plotted.size()/ncols;
	ret.reserve(nrows*2);
	for (PVRow i = 0; i < nrows; i++) {
		ret.push_back(0.0f);
		ret.push_back(plotted[i*ncols+axis_a]);
		ret.push_back(1.0f);
		ret.push_back(plotted[i*ncols+axis_b]);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " plotted_file" << std::endl;
		return 1;
	}

	if (sizeof(PVBCode) != sizeof(int)) {
		std::cerr << "sizeof PVBCode is different from sizeof(int) !!" << std::endl;
		return 1;
	}

	PVCol ncols;
	Picviz::PVPlotted::plotted_table_t plotted;
	if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, false, QString(argv[1]))) {
		std::cerr << "Unable to load plotted !" << std::endl;
		return 1;
	}

	PVBZCompute bz;
	bz.set_plotted(plotted, ncols);
	bz.set_zoom(1024, 1024);
	
	std::cout << "Start BCode computation..." << std::endl;
	std::vector<PVBCode> codes;
	codes.reserve(bz.get_nrows());
	BENCH_START(bcode);
	bz.compute_b(codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode, "BCode computation", plotted.size()*2, sizeof(float), codes.size(), sizeof(PVBCode));

	// Reduction
	BCodeCB bc_cb = allocate_BCodeCB();
	std::cout << "Start BCode reduction..." << std::endl;
	serial_bcodecb(&codes[0], codes.size(), bc_cb);

	//codes.clear();
	//bcode_cb_to_bcodes(codes, bc_cb);

	QApplication app(argc, argv);
	std::vector<float> p_ext;
	std::vector<int> bz_pts;
	
	/*
	{
		QMainWindow *window = new QMainWindow();
		window->setWindowTitle("bz - red code bz");
		BCCBView *v = new BCCBView(window);

		v->set_size(W_FRAME, H_FRAME);
		v->set_ortho(W_FRAME, H_FRAME);
		v->set_bccb(bc_cb);

		window->setCentralWidget(v);
		window->resize(v->sizeHint());
		window->show();
	}

	{
		QMainWindow *window = new QMainWindow();
		SLFloatView *v = new SLFloatView(window);

		extract_plotted(ncols, plotted, p_ext, 0, 1);
		v->set_size(1024,1024);
		v->set_ortho(1.0, 1.0);
		v->set_points(p_ext);

		window->setCentralWidget(v);
		window->resize(v->sizeHint());
		window->show();
	}*/
	

	{
		QMainWindow *window = new QMainWindow();
		window->setWindowTitle("bz - code bz");
		SLIntView *v = new SLIntView(window);

		bz.convert_to_points(W_FRAME, H_FRAME, codes, bz_pts);
		v->set_size(W_FRAME, H_FRAME);
		v->set_ortho(W_FRAME, H_FRAME);
		v->set_points(bz_pts);

		window->setCentralWidget(v);
		window->resize(v->sizeHint());
		window->show();
	}

	return app.exec();
}
