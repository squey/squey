#include <QDialog>

#include <algorithm> // std::max_element

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>
#include <pvguiqt/PVCountByStringsDlg.h>
#include <pvguiqt/PVQNraw.h>

bool PVGuiQt::PVQNraw::show_unique_values(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol c, Picviz::PVSelection const& sel, QWidget* parent, QDialog** dialog /*= nullptr*/)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);
	PVRush::PVNraw::unique_values_t values;
	uint64_t min;
	uint64_t max;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();

	bool ret_pbox = PVCore::PVProgressBox::progress([&,c] {
		nraw.get_unique_values(c, values, min, max, *((PVCore::PVSelBitField const*) &sel), &ctxt);
	}, ctxt, pbox);

	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, c, values, sel.get_number_of_selected_lines_in_range(0, nraw.get_number_rows()), min, max, parent);
	dlg->setWindowTitle("Distinct values of axis '" + nraw.get_axis_name(c) +"'");
	dlg->show();

	if (dialog) { // Keep dialog pointer to allow display toggle.
		*dialog = (QDialog*) dlg;
	}

	return true;
}

bool PVGuiQt::PVQNraw::show_count_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);
	PVRush::PVNraw::count_by_t values;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	size_t v2_unique_values_count;
	size_t min;
	size_t max;
	bool ret_pbox = PVCore::PVProgressBox::progress([&,col1,col2] { nraw.count_by(col1, col2, values, min, max, *((PVCore::PVSelBitField const*) &sel), v2_unique_values_count, &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVListUniqStringsDlg takes ownership of strings inside `values'
	PVCountByStringsDlg* dlg = new PVCountByStringsDlg(view, col1, col2, values, v2_unique_values_count, min, max, parent);
	dlg->setWindowTitle("Count by of axes '" + nraw.get_axis_name(col1) + "' and '" + nraw.get_axis_name(col2)+ "'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_sum_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);
	PVRush::PVNraw::sum_by_t values;
	uint64_t min;
	uint64_t max;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	uint64_t sum;
	bool ret_pbox = PVCore::PVProgressBox::progress([&,col1,col2] { nraw.sum_by(col1, col2, values, min, max, *((PVCore::PVSelBitField const*) &sel), sum, &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVSumByStringsDlg takes ownership of strings inside `values'
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, col1, values, sum, min, max, parent);
	dlg->setWindowTitle("Sum by of axes '" + nraw.get_axis_name(col1) + "' and '" + nraw.get_axis_name(col2)+ "'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_max_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);
	PVRush::PVNraw::max_by_t values;
	uint64_t min;
	uint64_t max;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	bool ret_pbox = PVCore::PVProgressBox::progress([&,col1,col2] { nraw.max_by(col1, col2, values, min, max, *((PVCore::PVSelBitField const*) &sel), &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVSumByStringsDlg takes ownership of strings inside `values'
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, col1, values, max, min, max, parent);
	dlg->setWindowTitle("Max by of axes '" + nraw.get_axis_name(col1) + "' and '" + nraw.get_axis_name(col2)+ "'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_min_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);
	PVRush::PVNraw::min_by_t values;
	uint64_t min;
	uint64_t max;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	bool ret_pbox = PVCore::PVProgressBox::progress([&,col1,col2] { nraw.min_by(col1, col2, values, min, max, *((PVCore::PVSelBitField const*) &sel), &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVSumByStringsDlg takes ownership of strings inside `values'
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, col1, values, max, min, max, parent);
	dlg->setWindowTitle("Min by of axes '" + nraw.get_axis_name(col1) + "' and '" + nraw.get_axis_name(col2)+ "'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_avg_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);
	PVRush::PVNraw::avg_by_t values;
	uint64_t min;
	uint64_t max;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	bool ret_pbox = PVCore::PVProgressBox::progress([&,col1,col2] { nraw.avg_by(col1, col2, values, min, max, *((PVCore::PVSelBitField const*) &sel), &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVSumByStringsDlg takes ownership of strings inside `values'
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, col1, values, max, min, max, parent);
	dlg->setWindowTitle("Average by of axes '" + nraw.get_axis_name(col1) + "' and '" + nraw.get_axis_name(col2)+ "'");
	dlg->show();

	return true;
}
