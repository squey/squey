/**
 * \file PVCountByStringsDlg.cpp
 *
 * Copyright (C) Picviz Labs 2012-2013
 */

#include <pvguiqt/PVCountByStringsDlg.h>
#include <pvguiqt/PVStringSortProxyModel.h>

#include <numeric>

/******************************************************************************
 *
 * PVGuiQt::PVCountByStringsDlg
 *
 *****************************************************************************/
void PVGuiQt::PVCountByStringsDlg::process_context_menu(QAction* act)
{
	PVAbstractListStatsDlg::process_context_menu(act);
	if (act && act == _act_list_v2) {
		QModelIndexList indexes = _values_view->selectionModel()->selectedIndexes();
		if (indexes.size() > 0 && indexes[0].isValid()) {
			size_t row = proxy_model()->mapToSource(indexes[0]).row();

			assert(row < get_model()->_values.size());
			__impl::PVCountByStringsModel::v1_v2_pair_t& v1_v2_pair  = get_model()->_values[row];

			typedef PVRush::PVNraw::unique_values_container_t elem_t;
			size_t total_count = std::accumulate(v1_v2_pair.second.begin(), v1_v2_pair.second.end(), 0,
				[](const size_t& rhs, const elem_t& lhs)
				{
					return rhs + lhs.second;
				}
			);

			Picviz::PVView_sp view_sp = _view.shared_from_this();
			PVListUniqStringsDlg* dlg = new PVListUniqStringsDlg(view_sp, _col2, v1_v2_pair.second, total_count, parentWidget());
			dlg->setWindowTitle("Details of value '" + QString(v1_v2_pair.first.first.c_str())+ "'");
			dlg->move(x()+width()+10, y());
			dlg->show();
		}
	}
}

PVGuiQt::__impl::PVCountByStringsModel* PVGuiQt::PVCountByStringsDlg::get_model()
{
	PVGuiQt::__impl::PVCountByStringsModel* m = static_cast<PVGuiQt::__impl::PVCountByStringsModel*>(model());
	assert(m);
	return m;
}

void PVGuiQt::PVCountByStringsDlg::init_max_element()
{
	_max_e = get_model()->get_max_element();
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVCountByStringsModel
 *
 *****************************************************************************/
int PVGuiQt::__impl::PVCountByStringsModel::rowCount(QModelIndex const& parent) const
{
	if (parent.isValid()) {
		return 0;
	}

	return _values.size();
}

QVariant PVGuiQt::__impl::PVCountByStringsModel::data(QModelIndex const& index, int role) const
{
	assert((size_t) index.row() < _values.size());

	if (role == Qt::DisplayRole) {
		switch (index.column()) {
			case 0:
			{
				std::string_tbb const& str = _values[index.row()].first.first;
				return QVariant(QString::fromUtf8(str.c_str(), str.size()));
			}
			break;
		}
	}
	else if (role == Qt::UserRole) {
		switch (index.column()) {
			case 1:
			{
				return QVariant::fromValue(_values[index.row()].first.second);
			}
		}
	}

	return QVariant();
}

size_t PVGuiQt::__impl::PVCountByStringsModel::get_max_element()
{
	return (*std::max_element(_values.begin(), _values.end(), [](const v1_v2_pair_t &lhs, const v1_v2_pair_t &rhs)
	{
		return lhs.first.second < rhs.first.second;
	})).first.second;
}