#include <algorithm> // std::max_element

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>
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

	typedef PVRush::PVNraw::unique_values_container_t elem_t;
	size_t max_e = (*std::max_element(values.begin(), values.end(), [](const elem_t &lhs, const elem_t &rhs) { return lhs.second < rhs.second; } )).second;

	// PVListUniqStringsDlg takes ownership of strings inside `values'
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view, c, values, sel.get_number_of_selected_lines_in_range(0, nraw.get_number_rows()), max_e, parent);
	dlg->setWindowTitle("Unique values of axis '" + nraw.get_axis_name(c) +"'");
	dlg->show();

	return true;
}
