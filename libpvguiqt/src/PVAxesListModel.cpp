/**
 * \file axes-comb_model.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

#include <pvguiqt/PVAxesListModel.h>

// Model

PVGuiQt::PVAxesListModel::PVAxesListModel(Picviz::PVView_sp& view_p, QObject* parent):
		QAbstractListModel(parent),
		_view_deleted(false),
		_view_observer(this)
{
	// PVView observer signal
	PVHive::PVHive::get().register_observer(
			view_p,
			_view_observer
		);
	_view_observer.connect_about_to_be_deleted(this, SLOT(about_to_be_deleted_slot(PVHive::PVObserverBase*)));
}

int PVGuiQt::PVAxesListModel::rowCount(const QModelIndex &parent) const
{
	if (parent.isValid())
		return 0;

	return rowCount();
}

int PVGuiQt::PVAxesListModel::rowCount() const
{
	if (!_view_deleted) {
		return picviz_view().get_original_axes_count();
	}
	return 0;
}

QVariant PVGuiQt::PVAxesListModel::data(const QModelIndex &index, int role) const
{
	if (index.row() < 0 || index.row() >= rowCount())
		return QVariant();

	if (role == Qt::DisplayRole) {
		return QVariant(picviz_view().get_original_axis_name(index.row()));
	}

	return QVariant();
}

Qt::ItemFlags PVGuiQt::PVAxesListModel::flags(const QModelIndex &index) const
{
	if (!index.isValid())
		return QAbstractItemModel::flags(index);

	return QAbstractItemModel::flags(index);
}

void PVGuiQt::PVAxesListModel::about_to_be_deleted_slot(PVHive::PVObserverBase*)
{
	beginResetModel();
	_view_deleted = true;
	endResetModel();
}

void PVGuiQt::PVAxesListModel::refresh_slot(PVHive::PVObserverBase*)
{
	reset();
}

