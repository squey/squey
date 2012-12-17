/**
 * \file PVLayerStackModel.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVLayerStack.h>
#include <picviz/PVView.h>

#include <pvhive/PVHive.h>

#include <pvguiqt/PVCustomQtRoles.h>
#include <pvguiqt/PVLayerStackModel.h>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::PVLayerStackModel
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackModel::PVLayerStackModel(Picviz::PVView_sp& lib_view, QObject* parent):
	QAbstractTableModel(parent),
	select_brush(QColor(255,240,200)),
	unselect_brush(QColor(180,180,180)),
	_obs(this),
	_ls_valid(true)
{
	PVLOG_DEBUG("PVGuiQt::PVLayerStackModel::%s : Creating object\n", __FUNCTION__);

	select_font.setBold(true);

	PVHive::get().register_actor(lib_view, _actor);
	PVHive::get().register_observer(lib_view, [=](Picviz::PVView& view) { return &view.get_layer_stack(); }, _obs);

	_obs.connect_about_to_be_deleted(this, SLOT(layer_stack_about_to_be_deleted(PVHive::PVObserverBase*)));
	_obs.connect_about_to_be_refreshed(this, SLOT(layer_stack_about_to_be_refreshed(PVHive::PVObserverBase*)));
	_obs.connect_refresh(this, SLOT(layer_stack_refreshed(PVHive::PVObserverBase*)));
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVLayerStackModel::columnCount(const QModelIndex& /*index*/) const
{
	return 2;
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::data
 *
 *****************************************************************************/
QVariant PVGuiQt::PVLayerStackModel::data(const QModelIndex &index, int role) const
{
	// AG: this comment is kept for the sake of history...
	/* We prepare a direct acces to the total number of layers */
	int layer_count = lib_layer_stack().get_layer_count();
	/* We create and store the true index of the layer in the lib */
	int lib_index = layer_count -1 - index.row();

	switch (role) {
		case Qt::DecorationRole:
			switch (index.column()) {
			case 0:
				if (lib_layer_stack().get_layer_n(lib_index).get_visible()) {
					return QPixmap(":/layer-active.png");
				} else {
					return QPixmap(":/layer-inactive.png");
				}
				break;
			}
			break;

		case (Qt::BackgroundRole):
			if (lib_layer_stack().get_selected_layer_index() == lib_index) {
				return QBrush(QColor(205,139,204));
			}
			break;
			/*
				if (parent_widget && parent_widget->get_layer_stack_widget() && parent_widget->get_layer_stack_widget()->get_layer_stack_view()) {
					PVLayerStackView *layer_stack_view = parent_widget->get_layer_stack_widget()->get_layer_stack_view();
					if (lib_layer_stack().get_selected_layer_index() == lib_index) {
						peturn QBrush(QColor(205,139,204));
					}
					if (layer_stack_view->mouse_hover_layer_index == index.row()) {
						return QBrush(QColor(200,200,200));
					}
					return QBrush(QColor(255,255,255));
				}*/

		case (Qt::DisplayRole):
			switch (index.column()) {
				/*case 1:
					return (int)lib_layer_stack().get_layer_n(lib_index).get_locked();*/

				case 1:
					return lib_layer_stack().get_layer_n(lib_index).get_name();
			}
			break;

		case (Qt::EditRole):
			switch (index.column()) {
				case 1:
					return lib_layer_stack().get_layer_n(lib_index).get_name();
			}
			break;

		case (Qt::TextAlignmentRole):
			switch (index.column()) {
				case 0:
					return (Qt::AlignCenter + Qt::AlignVCenter);

				default:
					return (Qt::AlignLeft + Qt::AlignVCenter);
			}
			break;
	}
	return QVariant();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVGuiQt::PVLayerStackModel::flags(const QModelIndex &index) const
{
	switch (index.column()) {
		case 0:
			//return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
			return Qt::ItemIsEditable | Qt::ItemIsEnabled;

		default:
			return (Qt::ItemIsEditable | Qt::ItemIsEnabled);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::headerData
 *
 *****************************************************************************/
QVariant PVGuiQt::PVLayerStackModel::headerData(int /*section*/, Qt::Orientation /*orientation*/, int role) const
{
	// FIXME : this should not be used : delegate...
	switch (role) {
		case (Qt::SizeHintRole):
			return QSize(37,37);
			break;
	}
	
	return QVariant();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::rowCount
 *
 *****************************************************************************/
int PVGuiQt::PVLayerStackModel::rowCount(const QModelIndex &/*index*/) const
{
	if (!_ls_valid) {
		return 0;
	}

	return lib_layer_stack().get_layer_count();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::setData
 *
 *****************************************************************************/
bool PVGuiQt::PVLayerStackModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	int layer_count = lib_layer_stack().get_layer_count();
	/* We create and store the true index of the layer in the lib */
	int lib_index = layer_count -1 - index.row();

	switch (role) {
		case (Qt::EditRole):
			switch (index.column()) {
				case 0:
					_actor.call<FUNC(Picviz::PVView::toggle_layer_stack_layer_n_visible_state)>(lib_index);
					_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
					return true;

				/*case 1:
					_actor.call<FUNC(Picviz::PVView::toggle_layer_stack_layer_n_locked_state)>(lib_index);
					return true;*/

				case 1:
					_actor.call<FUNC(Picviz::PVView::set_layer_stack_layer_n_name)>(lib_index, value.toString());
					return true;

				default:
					return QAbstractTableModel::setData(index, value, role);
			}

		case (PVCustomQtRoles::RoleSetSelectedItem): {
			const bool is_sel = value.toBool();
			if (is_sel) {
				_actor.call<FUNC(Picviz::PVView::set_layer_stack_selected_layer_index)>(lib_index);
			}
			return true;
		}

		default:
			return QAbstractTableModel::setData(index, value, role);
	}

	return false;
}

void PVGuiQt::PVLayerStackModel::layer_stack_about_to_be_deleted(PVHive::PVObserverBase* /*o*/)
{
	beginResetModel();
	_ls_valid = false;
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::layer_stack_about_to_be_refreshed(PVHive::PVObserverBase* /*o*/)
{
	beginResetModel();
}

void PVGuiQt::PVLayerStackModel::add_new_layer_from_file(const QString& path)
{
	_actor.call<FUNC(Picviz::PVView::add_new_layer_from_file)>(path);
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}

void PVGuiQt::PVLayerStackModel::reset_layer(const int idx)
{
	Picviz::PVLayerStack& layerstack = lib_layer_stack();
	Picviz::PVLayer& layer = layerstack.get_layer_n(idx);
	layer.reset_to_full_and_default_color();
}

void PVGuiQt::PVLayerStackModel::layer_stack_refreshed(PVHive::PVObserverBase* /*o*/)
{
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::add_new_layer()
{
	_actor.call<FUNC(Picviz::PVView::add_new_layer)>();
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}

void PVGuiQt::PVLayerStackModel::move_selected_layer_up()
{
	beginResetModel();
	_actor.call<FUNC(Picviz::PVView::move_selected_layer_up)>();
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::move_selected_layer_down()
{
	beginResetModel();
	_actor.call<FUNC(Picviz::PVView::move_selected_layer_down)>();
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::delete_selected_layer()
{
	_actor.call<FUNC(Picviz::PVView::delete_selected_layer)>();
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}

void PVGuiQt::PVLayerStackModel::delete_layer_n(const int idx)
{
	assert(idx < rowCount());
	_actor.call<FUNC(Picviz::PVView::delete_layer_n)>(idx);
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}

void PVGuiQt::PVLayerStackModel::load_from_file(const QString& file)
{
	_actor.call<FUNC(Picviz::PVView::load_from_file)>(file);
	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}
