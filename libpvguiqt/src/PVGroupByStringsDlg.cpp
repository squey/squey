/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVGroupByStringsDlg.h>

#include <pvkernel/core/PVProgressBox.h>

#include <inendi/PVView.h>

#include <pvcop/db/algo.h>

#include <numeric>

/******************************************************************************
 *
 * PVGuiQt::PVGroupByStringsDlg
 *
 *****************************************************************************/
bool PVGuiQt::PVGroupByStringsDlg::process_context_menu(QAction* act)
{
	if (act && act == _act_details) {
		Inendi::PVSelection const& indexes = model().current_selection();

		if (not indexes.is_empty()) {
			double sum;
			double min;
			double max;
			pvcop::db::array col1_out;
			pvcop::db::array col2_out;

			PVRush::PVNraw const& nraw = lib_view()->get_rushnraw_parent();

			// We did the col1_in by col2_in computation
			const pvcop::db::array col1_in = nraw.collection().column(_col);
			const pvcop::db::array col2_in = nraw.collection().column(_col2);

			int row_id = indexes.find_next_set_bit(0, col1_in.size()); // We can only get the
			                                                           // details of the first
			                                                           // selected value
			// Get it from value_col which is col2_in but without duplication
			const QString value = QString::fromStdString(model().value_col().at(row_id));

			auto ret = PVCore::PVProgressBox::progress(
			    [&](PVCore::PVProgressBox& pbox) {
				    pbox.set_enable_cancel(true);
				    pvcop::db::algo::op_by_details(col1_in, col2_in, value.toStdString(), col1_out,
				                                   col2_out, _sel);

				    pvcop::db::array minmax = pvcop::db::algo::minmax(col2_out);

				    std::string min_str = minmax.at(0);
				    std::istringstream min_buf(min_str);
				    min_buf >> min;

				    std::string max_str = minmax.at(1);
				    std::istringstream max_buf(max_str);
				    max_buf >> max;

				    sum = pvcop::db::algo::sum(col2_out);
				},
			    QObject::tr("Computing values..."), parentWidget());

			if (ret == PVCore::PVProgressBox::CancelState::CONTINUE) {
				PVListUniqStringsDlg* dlg =
				    new PVListUniqStringsDlg(*lib_view(), _col2, std::move(col1_out),
				                             std::move(col2_out), sum, min, max, parentWidget());
				dlg->setWindowTitle("Details of value '" + value + "'");
				dlg->move(x() + width() + 10, y());
				dlg->show();
				return true;
			}
		}

		return false;
	}

	return PVAbstractListStatsDlg::process_context_menu(act);
}
