/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVGroupByStringsDlg.h>

#include <pvkernel/core/PVProgressBox.h>

#include <pvcop/db/algo.h>

#include <numeric>

/******************************************************************************
 *
 * PVGuiQt::PVCountByStringsDlg
 *
 *****************************************************************************/
bool PVGuiQt::PVGroupByStringsDlg::process_context_menu(QAction* act)
{
	if (act && act == _act_list_v2) {
		bool ret = false;
		QModelIndexList indexes = _values_view->selectionModel()->selectedIndexes();
		if (indexes.size() > 0 && indexes[0].isValid()) {

			double count;
			double min;
			double max;
			pvcop::db::array col1_out;
			pvcop::db::array col2_out;

			Inendi::PVView_sp view_sp = _view.shared_from_this();
			PVRush::PVNraw const& nraw = view_sp->get_rushnraw_parent();

			const pvcop::db::array col1_in = nraw.collection().column(_col);
			const pvcop::db::array col2_in = nraw.collection().column(_col2);

			const QModelIndex& idx = indexes[0];
			const std::string& value = idx.data().toString().toStdString();

			tbb::task_group_context ctxt(tbb::task_group_context::isolated);
			ctxt.reset();

			PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(QObject::tr("Computing values..."), parentWidget());
			pbox->set_enable_cancel(true);

			ret = PVCore::PVProgressBox::progress([&]
			{
				pvcop::db::algo::op_by_details(col1_in, col2_in, value, col1_out, col2_out);

				std::string min_str = pvcop::db::algo::min(col2_out).at(0);
				std::istringstream min_buf(min_str);
				min_buf >> min;

				std::string max_str = pvcop::db::algo::max(col2_out).at(0);
				std::istringstream max_buf(max_str);
				max_buf >> max;

				count = col1_out.size();
			}, ctxt, pbox);

			if (ret) {
				PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view_sp, _col2, std::move(col1_out), std::move(col2_out), count, min, max, parentWidget());
				dlg->setWindowTitle("Details of value '" + idx.data().toString()+ "'");
				dlg->move(x()+width()+10, y());
				dlg->show();
			}
		}

		return ret;
	}

	return PVAbstractListStatsDlg::process_context_menu(act);
}
