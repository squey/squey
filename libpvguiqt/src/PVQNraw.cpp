#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>
#include <pvguiqt/PVQNraw.h>

bool PVGuiQt::PVQNraw::show_unique_values(PVRush::PVNraw const& nraw, PVCol c, Picviz::PVSelection const& sel, QWidget* parent)
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
	PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(values, parent);
	dlg->exec();

	return true;
}
