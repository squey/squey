/**
 * \file PVListUniqStrings.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>

// PVlistColNrawDlg
//

PVGuiQt::PVListUniqStringsDlg::PVListUniqStringsDlg(PVRush::PVNraw::unique_values_t& values, QWidget* parent):
	PVListDisplayDlg(new __impl::PVListUniqStringsModel(values), parent)
{
}

PVGuiQt::PVListUniqStringsDlg::~PVListUniqStringsDlg()
{
	// Force deletion so that the internal std::vector is destroyed!
	model()->deleteLater();
}

// Private implementation of PVListColNrawModel
//

PVGuiQt::__impl::PVListUniqStringsModel::PVListUniqStringsModel(PVRush::PVNraw::unique_values_t& values, QWidget* parent):
	QAbstractListModel(parent)
{
	_values.reserve(values.size());
	for (std::string_tbb const& s: values) {
		_values.push_back(std::move(s));
	}
}

int PVGuiQt::__impl::PVListUniqStringsModel::rowCount(QModelIndex const& parent) const
{
	if (parent.isValid()) {
		return 0;
	}

	return _values.size();
}

QVariant PVGuiQt::__impl::PVListUniqStringsModel::data(QModelIndex const& index, int role) const
{
	if (role == Qt::DisplayRole) {
		assert((size_t) index.row() < _values.size());
		std::string_tbb const& str = _values[index.row()];
		return QVariant(QString::fromUtf8(str.c_str(), str.size()));
	}

	return QVariant();
}

QVariant PVGuiQt::__impl::PVListUniqStringsModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole) {
		if (orientation == Qt::Horizontal) {
			return QVariant();
		}
		
		return QVariant(QString().setNum(section));
	}

	return QVariant();
}
