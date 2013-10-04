#include <algorithm> // std::max_element

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>
#include <pvguiqt/PVCountByStringsDlg.h>
#include <pvguiqt/PVQNraw.h>


bool PVGuiQt::PVQNraw::show_unique_values(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol c, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(false);
	PVRush::PVNraw::unique_values_t values;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	bool ret_pbox = PVCore::PVProgressBox::progress([&,c] { nraw.get_unique_values_for_col_with_sel(c, values, *((PVCore::PVSelBitField const*) &sel), &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVListUniqStringsDlg takes ownership of strings inside `values'
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, c, values, sel.get_number_of_selected_lines_in_range(0, nraw.get_number_rows()), parent);
	dlg->setWindowTitle("Unique values of axis '" + nraw.get_axis_name(c) +"'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_count_by(Picviz::PVView_sp& view, PVRush::PVNraw const& nraw, PVCol col1, PVCol col2, Picviz::PVSelection const& sel, QWidget* parent)
{
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(false);
	PVRush::PVNraw::count_by_t values;
	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();
	size_t v2_unique_values_count;
	bool ret_pbox = PVCore::PVProgressBox::progress([&,col1,col2] { nraw.count_by_with_sel(col1, col2, values, *((PVCore::PVSelBitField const*) &sel), v2_unique_values_count, &ctxt); }, ctxt, pbox);
	if (!ret_pbox || values.size() == 0) {
		return false;
	}

	// PVListUniqStringsDlg takes ownership of strings inside `values'
	PVCountByStringsDlg* dlg = new PVCountByStringsDlg(view, col1, col2, values, v2_unique_values_count, parent);
	dlg->setWindowTitle("Count by of axes '" + nraw.get_axis_name(col1) + "' and '" + nraw.get_axis_name(col2)+ "'");
	dlg->show();

	return true;
}
