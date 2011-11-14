//! \file PVArgumentListModel.cpp
//! $Id: PVArgumentListModel.cpp 3106 2011-06-11 14:00:20Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <PVArgumentListModel.h>

#include <QStandardItemModel>

PVInspector::PVArgumentListModel::PVArgumentListModel(PVCore::PVArgumentList &args, QObject* parent):
	QAbstractTableModel(parent),
	_args(args)
{
}

int PVInspector::PVArgumentListModel::rowCount(const QModelIndex &parent) const
{
	// Cf. QAbstractTableModel's documentation. This is for a table view.
	if (parent.isValid())
		return 0;

	return _args.size();
}

int PVInspector::PVArgumentListModel::columnCount(const QModelIndex& parent) const
{
	// Same as above
	if (parent.isValid())
		return 0;

	return 1;
}

QVariant PVInspector::PVArgumentListModel::data(const QModelIndex& index, int role) const
{
	if (role != Qt::DisplayRole && role != Qt::EditRole)
		return QVariant();

	PVCore::PVArgumentList::iterator it = _args.begin();
	std::advance(it, index.row());

	return it.value();
}

bool PVInspector::PVArgumentListModel::setData(const QModelIndex& index, const QVariant &value, int role)
{
	if (index.column() != 0 || role != Qt::EditRole)
		return false;

	PVCore::PVArgumentList::iterator it = _args.begin();
	std::advance(it, index.row());
	if (it == _args.end())
		return false; // Should never happen !

	it.value() = value;

	emit dataChanged(index, index);

	return true;
}

Qt::ItemFlags PVInspector::PVArgumentListModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags ret;

	if (index.column() == 0) {
		ret |= Qt::ItemIsEnabled | Qt::ItemIsEditable;
	}

	return ret;
}

QVariant PVInspector::PVArgumentListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	return QAbstractTableModel::headerData(section, orientation, role);
}
