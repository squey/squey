//! \file PVArgumentListModel.cpp
//! $Id: PVArgumentListModel.cpp 3106 2011-06-11 14:00:20Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2012

#include <PVArgumentListModel.h>

#include <QStandardItemModel>


/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::PVArgumentListModel
 *
 *****************************************************************************/
PVInspector::PVArgumentListModel::PVArgumentListModel(QObject* parent):
	QAbstractTableModel(parent),
	_args(NULL)
{
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::PVArgumentListModel
 *
 *****************************************************************************/
PVInspector::PVArgumentListModel::PVArgumentListModel(PVCore::PVArgumentList &args, QObject* parent):
	QAbstractTableModel(parent),
	_args(&args)
{
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::columnCount
 *
 *****************************************************************************/
int PVInspector::PVArgumentListModel::columnCount(const QModelIndex& parent) const
{
	// Same as above
	if (_args == NULL || parent.isValid())
		return 0;

	return 1;
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVArgumentListModel::data(const QModelIndex& index, int role) const
{
	// We check if we have no args, and then restrict to the cases of Qt::DisplayRole and Qt::EditRole
	if (_args == NULL || (role != Qt::DisplayRole && role != Qt::EditRole))
		return QVariant();

	// We get an iterator for the Arguments
	PVCore::PVArgumentList::iterator it = _args->begin();
	// We jump to the row given by the Index
	std::advance(it, index.row());

	// We return the value of tha argument at that position
	return it.value();
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVInspector::PVArgumentListModel::flags(const QModelIndex& index) const
{
	// nothing to say if we have no Arguments
	if (_args == NULL) {
		return QAbstractTableModel::flags(index);
	}

	// We prepare an empty ItemFlags
	Qt::ItemFlags ret;

	// We set the flags in case we are in the first column
	if (index.column() == 0) {
		ret |= Qt::ItemIsEnabled | Qt::ItemIsEditable;
	}

	return ret;
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVArgumentListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	return QAbstractTableModel::headerData(section, orientation, role);
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVArgumentListModel::rowCount(const QModelIndex &parent) const
{
	// Cf. QAbstractTableModel's documentation. This is for a table view.
	if (_args == NULL || parent.isValid())
		return 0;

	return _args->size();
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::set_args
 *
 *****************************************************************************/
void PVInspector::PVArgumentListModel::set_args(PVCore::PVArgumentList& args)
{
	beginResetModel();
	_args = &args;
	endResetModel();
}




/******************************************************************************
 *
 * PVInspector::PVArgumentListModel::setData
 *
 *****************************************************************************/
bool PVInspector::PVArgumentListModel::setData(const QModelIndex& index, const QVariant &value, int role)
{
	if (_args == NULL || index.column() != 0 || role != Qt::EditRole)
		return false;

	PVCore::PVArgumentList::iterator it = _args->begin();
	std::advance(it, index.row());
	if (it == _args->end())
		return false; // Should never happen !

	it.value() = value;

	emit dataChanged(index, index);

	return true;
}



