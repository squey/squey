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

namespace __impl {

class PVListUniqStringsModel: public PVGuiQt::__impl::PVAbstractListStatsModel
{
	typedef typename PVRush::PVNraw::unique_values_t unique_values_t;

public:
	PVListUniqStringsModel(unique_values_t& values, QWidget* parent = nullptr) : PVGuiQt::__impl::PVAbstractListStatsModel(parent), _values(std::move(values))
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

private:
	unique_values_t _values;
};

}

class PVListUniqStringsDlg : public PVAbstractListStatsDlg
{
	typedef typename PVRush::PVNraw::unique_values_t unique_values_t;

public:
	PVListUniqStringsDlg(Picviz::PVView_sp& view, PVCol c, unique_values_t& values, size_t abs_max, size_t rel_max, QWidget* parent = nullptr) :
		PVAbstractListStatsDlg(view, c, new __impl::PVListUniqStringsModel(values), abs_max, rel_max, parent)
	{
	}
};

}

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
