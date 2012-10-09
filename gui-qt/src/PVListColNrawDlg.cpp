/**
 * \file PVListColNrawDlg.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVNraw.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVProgressBox.h>
#include <PVListColNrawDlg.h>

#include <QClipboard>
#include <QMessageBox>

// PVlistColNrawDlg
//

PVInspector::PVListColNrawDlg::PVListColNrawDlg(Picviz::PVView const& view, std::vector<int> const& idxes, size_t nvalues, PVCol view_col, QWidget* parent):
	PVListDisplayDlg(new __impl::PVListColNrawModel(&view.get_rushnraw_parent(), &idxes, nvalues, view.get_original_axis_index(view_col)), parent)
{
}

// Private implementation of PVListColNrawModel
//

PVInspector::__impl::PVListColNrawModel::PVListColNrawModel(PVRush::PVNraw const* nraw, std::vector<int> const* idxes, size_t nvalues, PVCol nraw_col, QWidget* parent):
	QAbstractListModel(parent),
	_nraw(nraw),
	_nraw_col(nraw_col),
	_idxes(idxes),
	_nvalues(nvalues)
{
	assert(nraw);
	assert(nraw_col < nraw->get_number_cols());
	assert((size_t) idxes->size() >= nvalues);
}

int PVInspector::__impl::PVListColNrawModel::rowCount(QModelIndex const& parent) const
{
	if (parent.isValid()) {
		return 0;
	}

	return _nvalues;
}

QVariant PVInspector::__impl::PVListColNrawModel::data(QModelIndex const& index, int role) const
{
	if (role == Qt::DisplayRole) {
		assert((size_t) index.row() < _nvalues);
		return QVariant(_nraw->at(_idxes->at(index.row()), _nraw_col));
	}

	return QVariant();
}

QVariant PVInspector::__impl::PVListColNrawModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole) {
		if (orientation == Qt::Horizontal) {
			return QVariant();
		}
		
		return QVariant(QString().setNum(section));
	}

	return QVariant();
}
