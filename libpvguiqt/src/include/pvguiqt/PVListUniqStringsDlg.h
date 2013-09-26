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
		PVAbstractListStatsDlg(view, c, (QAbstractListModel*) new __impl::PVListUniqStringsModel<T>(values, parent), selection_count)
	{
		typedef PVRush::PVNraw::unique_values_container_t elem_t;
		_max_e = (*std::max_element(values.begin(), values.end(), [](const elem_t &lhs, const elem_t &rhs) { return lhs.second < rhs.second; } )).second;
	}
};

namespace __impl {

template <typename T>
class PVListUniqStringsModel: public PVGuiQt::__impl::PVAbstractListStatsModel
{

public:
	PVListUniqStringsModel(T& values, QWidget* parent = nullptr) : PVGuiQt::__impl::PVAbstractListStatsModel(parent)
	{
		for (auto& v : values) {
			_values.emplace_back(std::move(v.first), v.second);
		}
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

private:
	typedef std::pair<std::string_tbb, size_t> pair_t;
	std::vector<pair_t> _values;
};

}

}

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
