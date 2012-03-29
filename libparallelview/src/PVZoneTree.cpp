#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/simple_lines_float_view.h>

#include <cassert>

#include <QMainWindow>

void PVParallelView::PVZoneTreeBase::set_trans_plotted(plotted_int_t const& plotted, PVRow nrows, PVCol ncols)
{
	assert(_col_a < ncols);
	assert(_col_b < ncols);

	_plotted = &plotted;
	_ncols = ncols;
	_nrows = nrows;
	_nrows_aligned = ((_nrows+3)/4)*4;

	/*for (PVRow i = 0; i < NBUCKETS; i++) {
		_tree[i].reserve(_nrows);
	}*/
}

void PVParallelView::PVZoneTreeBase::display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted)
{
	QMainWindow *window = new QMainWindow();
	window->setWindowTitle(name);
	SLFloatView *v = new SLFloatView(window);

	v->set_size(1024, 1024);
	v->set_ortho(1.0f, 1.0f);

	pts_t *pts = new pts_t();
	get_float_pts(*pts, org_plotted);
	v->set_points(*pts);

	window->setCentralWidget(v);
	window->resize(v->sizeHint());
	window->show();
}
