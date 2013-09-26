/**
 * \file PVCountByStringsDlg.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
#define __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__

#include <pvguiqt/PVAbstractListStatsDlg.h>

#include <QAbstractListModel>

namespace PVGuiQt
{

namespace __impl
{
template <typename T>
class PVCountByStringsModel;
}

class PVCountByStringsDlg : public PVAbstractListStatsDlg
{
public:
	template <typename T>
	PVCountByStringsDlg(Picviz::PVView_sp& view, PVCol col1, PVCol col2, T& values, size_t selection_count, QWidget* parent = nullptr) :
		PVAbstractListStatsDlg(view, col1, (QAbstractListModel*) new __impl::PVCountByStringsModel<T>(values, parent), selection_count, parent)
	{
		__impl::PVCountByStringsModel<T>* m = static_cast<__impl::PVCountByStringsModel<T>*>(model());
		assert(m);
		_max_e = m->get_max_element();

		// FIXME: this is ugly
		PVRush::PVNraw::unique_values_t unique_values_col2;
		view->get_rushnraw_parent().get_unique_values_for_col_with_sel(col2, unique_values_col2, *view->get_selection_visible_listing());
		_selection_count = unique_values_col2.size();
	}
};

namespace __impl {

template <typename T>
class PVCountByStringsModel: public PVGuiQt::__impl::PVAbstractListStatsModel
{

public:
	PVCountByStringsModel(T& values, QWidget* parent = nullptr) : PVGuiQt::__impl::PVAbstractListStatsModel(parent)
	{
		size_t v1_count = values.size();
		_values.reserve(v1_count);
		for (auto& v1 : values) {
			auto& v2_values = v1.second;

			// v2 vector
			vector_v2_count_t vector_v2;
			size_t v2_count = v2_values.size();
			vector_v2.reserve(v2_count);
			for (auto& v2 : v2_values) {
				vector_v2.emplace_back(std::move(v2.first), v2.second);
			}

			// v1 count
			string_count_t v1_count_pair(std::move(v1.first) , v2_count);

			_values.emplace_back(std::move(v1_count_pair), std::move(vector_v2));
		}
	}

	int rowCount(QModelIndex const& parent) const
	{
		if (parent.isValid()) {
			return 0;
		}

		return _values.size();
	}

	QVariant data(QModelIndex const& index, int role) const
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

public:
	size_t get_max_element()
	{
		return (*std::max_element(_values.begin(), _values.end(), [](const v1_v2_pair_t &lhs, const v1_v2_pair_t &rhs)
		{
			return lhs.first.second < rhs.first.second;
		})).first.second;
	}

private:
	typedef std::pair<std::string_tbb, size_t> string_count_t;
	typedef std::vector<string_count_t> vector_v2_count_t;
	typedef std::pair<string_count_t, vector_v2_count_t> v1_v2_pair_t;
	typedef std::vector<v1_v2_pair_t> count_by_t;

	count_by_t _values;
};

}

}

#endif // __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
