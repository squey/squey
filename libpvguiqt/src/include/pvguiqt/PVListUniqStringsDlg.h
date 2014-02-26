/**
 * \file PVListUniqStringsDlg.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
#define __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__

#include <pvguiqt/PVAbstractListStatsDlg.h>

#include <picviz/PVView_types.h>

#include <QAbstractListModel>

#include <tbb/parallel_reduce.h>

namespace PVGuiQt
{

namespace __impl
{
template <typename T>
class PVListUniqStringsModel;
}


class PVListUniqStringsDlg : public PVAbstractListStatsDlg
{
public:
	template <typename T>
	PVListUniqStringsDlg(Picviz::PVView_sp& view, PVCol c, T& values, size_t selection_count, QWidget* parent = nullptr) :
		PVAbstractListStatsDlg(view, c, new __impl::PVListUniqStringsModel<T>(values), selection_count, parent)
	{
		__impl::PVListUniqStringsModel<T>* m = static_cast<__impl::PVListUniqStringsModel<T>*>(model());
		assert(m);
		set_max_element(m->max_element());
	}
};

namespace __impl {

template <typename T>
class PVListUniqStringsModel: public PVGuiQt::__impl::PVAbstractListStatsModel
{
public:
	PVListUniqStringsModel(T& values, QWidget* parent = nullptr) : PVGuiQt::__impl::PVAbstractListStatsModel(parent), _values(std::move(values))
	{
	}

public:
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
					std::string_tbb const& str = _values[index.row()].first;
					return QVariant(QString::fromUtf8(str.c_str(), str.size()));
				}
				break;
			}
		}
		else if (role == Qt::UserRole) {
			switch (index.column()) {
				case 1:
				{
					return QVariant::fromValue(_values[index.row()].second);
				}
			}
		}

		return QVariant();
	}

	inline size_t max_element()
	{
		typedef PVRush::PVNraw::unique_values_value_t value_type;
		return std::max_element(_values.begin(), _values.end(), [](const value_type &lhs, const value_type &rhs) { return lhs.second < rhs.second; } )->second;
	}

private:
	T _values;
};

}

}

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
