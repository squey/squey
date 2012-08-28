/**
 * \file axes-comb_model.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>
#include <pvhive/waxes/waxes.h>

#include <pvguiqt/PVAxesCombinationModel.h>

// Observers implementation

void PVGuiQt::__impl::remove_column_Observer::about_to_be_updated(const arguments_type& args) const
{
	int axis_index = std::get<0>(args);

	PVLOG_INFO("remove_column_Observer::about_to_be_updated %d\n", axis_index);

	_model->beginRemoveRow(axis_index);
}

void PVGuiQt::__impl::remove_column_Observer::update(const arguments_type& /*args*/) const
{
	PVLOG_INFO("remove_column_Observer::update\n");

	_model->endRemoveRow();
}

void PVGuiQt::__impl::axis_append_Observer::about_to_be_updated(const arguments_type& /*args*/) const
{
	int axis_index = ((Picviz::PVView*)get_object())->get_axes_count();
	PVLOG_INFO("axis_append_Observer::about_to_be_updated %d\n", axis_index);

	_model->beginInsertRow(axis_index);
}

void PVGuiQt::__impl::axis_append_Observer::update(const arguments_type& /*args*/) const
{
	PVLOG_INFO("axis_append_Observer::update\n");

	_model->endInsertRow();
}

void PVGuiQt::__impl::set_axis_name_Observer::update(arguments_type const& args) const
{
	int axis_index = std::get<0>(args);

	PVLOG_INFO("set_axis_name_Observer::update %d\n", axis_index);

	emit const_cast<PVGuiQt::PVAxesCombinationModel*>(_model)->dataChanged(_model->index(axis_index, 0), _model->index(axis_index, 0));
}

void PVGuiQt::__impl::move_axis_to_new_position_Observer::update(arguments_type const& args) const
{
	int old_index = std::get<0>(args);
	int new_index = std::get<1>(args);

	PVLOG_INFO("move_axis_to_new_position_Observer::update %d <-> %d \n", old_index, new_index);

	_model->beginRemoveRow(old_index);
	_model->endRemoveRow();

	_model->beginInsertRow(new_index);
	_model->endInsertRow();
}

// Model

PVGuiQt::PVAxesCombinationModel::PVAxesCombinationModel(Picviz::PVView_sp& view_p, QObject* parent):
		QAbstractListModel(parent),
		_view_deleted(false),
		_set_axis_name_observer(this),
		_view_observer(this),
		_remove_column_observer(this),
		_axis_append_observer(this),
		_move_axis_to_new_position_observer(this)

{
	// PVView observer signal
	PVHive::PVHive::get().register_observer(
			view_p,
			_view_observer
		);
	_view_observer.connect_about_to_be_deleted(this, SLOT(about_to_be_deleted_slot(PVHive::PVObserverBase*)));

	// PVView::set_axis_name function observer
	PVHive::get().register_func_observer(
			view_p,
			_set_axis_name_observer
		);

	// PVView::remove_column function observer
	PVHive::get().register_func_observer(
			view_p,
			_remove_column_observer
		);

	// PVView::axis_append function observer
	PVHive::get().register_func_observer(
			view_p,
			_axis_append_observer
		);

	// PVView::move_axis_to_new_position function observer
	PVHive::get().register_func_observer(
			view_p,
			_move_axis_to_new_position_observer
		);

	// Register view actor
	PVHive::get().register_actor(view_p, _actor);
}

int PVGuiQt::PVAxesCombinationModel::rowCount(const QModelIndex &parent) const
{
	if (parent.isValid())
		return 0;

	return rowCount();
}

int PVGuiQt::PVAxesCombinationModel::rowCount() const
{
	if (!_view_deleted) {
		return picviz_view().get_axes_count();
	}
	return 0;
}

QVariant PVGuiQt::PVAxesCombinationModel::data(const QModelIndex &index, int role) const
{
	if (index.row() < 0 || index.row() >= rowCount())
		return QVariant();

	QString axis_name = picviz_view().get_axis_name(index.row());
	if (role == Qt::DisplayRole)
		return QVariant(axis_name);

	return QVariant();
}

Qt::ItemFlags PVGuiQt::PVAxesCombinationModel::flags(const QModelIndex &index) const
{
	if (!index.isValid())
		return QAbstractItemModel::flags(index) | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;

	return QAbstractItemModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;
}

bool PVGuiQt::PVAxesCombinationModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	PVLOG_INFO("setData\n");
	if (index.row() >= 0 && index.row() < rowCount()) {
		if (role == Qt::EditRole) {
			_actor.call<FUNC(Picviz::PVView::set_axis_name)>(index.row(), value.toString());
			//emit dataChanged(index, index);
			return true;
		}
	}
	return false;
}

void PVGuiQt::PVAxesCombinationModel::beginInsertRow(int row)
{
	beginInsertRows(QModelIndex(), row, row);
}

void PVGuiQt::PVAxesCombinationModel::endInsertRow()
{
	endInsertRows();
}

void PVGuiQt::PVAxesCombinationModel::beginRemoveRow(int row)
{
	beginRemoveRows(QModelIndex(), row, row);
}

void PVGuiQt::PVAxesCombinationModel::endRemoveRow()
{
	endRemoveRows();
}

void PVGuiQt::PVAxesCombinationModel::about_to_be_deleted_slot(PVHive::PVObserverBase*)
{
	std::cout << "PVGuiQt::PVAxesCombinationModel::about_to_be_deleted (" << boost::this_thread::get_id() << ")" << std::endl;
	beginResetModel();
	_view_deleted = true;
	endResetModel();
}

void PVGuiQt::PVAxesCombinationModel::refresh_slot(PVHive::PVObserverBase*)
{
	PVLOG_INFO("PVGuiQt::PVAxesCombinationModel::refresh\n");
	reset();
}

