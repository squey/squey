/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVGroupByStringsDlg.h>

#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/core/PVProgressBox.h>

#include <pvguiqt/PVStatsModel.h>

#include <pvcop/db/algo.h>

#include <numeric>

PVGuiQt::PVStatsModel* PVGuiQt::PVGroupByStringsDlg::details_create_model(
    const Inendi::PVView& view, PVCol, Inendi::PVSelection const&)
{
	const PVRush::PVNraw& nraw = view.get_rushnraw_parent();
	Inendi::PVSelection const& indexes = model().current_selection();

	if (not indexes.is_empty()) {
		pvcop::db::array col1_out;
		pvcop::db::array col2_out;
		pvcop::db::array minmax;
		pvcop::db::array sum;

		// We did the col1_in by col2_in computation
		const pvcop::db::array& col1_in = nraw.column(_col);
		const pvcop::db::array& col2_in = nraw.column(_col2);

		int row_id = indexes.find_next_set_bit(0, col1_in.size()); // We can only get the
		                                                           // details of the first
		                                                           // selected value
		// Get it from value_col which is col2_in but without duplication
		const QString value = QString::fromStdString(model().value_col().at(row_id));

		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox& pbox) {
			    pbox.set_enable_cancel(true);
			    pvcop::db::algo::op_by_details(col1_in, col2_in, value.toStdString(), col1_out,
			                                   col2_out, _sel);

			    minmax = pvcop::db::algo::minmax(col2_out);
			    sum = pvcop::db::algo::sum(col2_out);
			},
		    QObject::tr("Computing values..."), parentWidget());

		QString col2_name =
		    lib_view()->get_parent<Inendi::PVSource>().get_format().get_axes().at(_col2).get_name();

		return new PVStatsModel("Count", col2_name, QString(), std::move(col1_out),
		                        std::move(col2_out), std::move(sum), std::move(minmax));
	}

	return nullptr;
}

/******************************************************************************
 *
 * PVGuiQt::PVGroupByStringsDlg
 *
 *****************************************************************************/
bool PVGuiQt::PVGroupByStringsDlg::process_context_menu(QAction* act)
{
	if (act && act == _act_details) {
		PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(
		    *lib_view(), _col2, [this](auto&&... args) { return details_create_model(args...); },
		    parentWidget());
		dlg->move(x() + width() + 10, y());
		dlg->show();
		return true;
	}

	return PVAbstractListStatsDlg::process_context_menu(act);
}
