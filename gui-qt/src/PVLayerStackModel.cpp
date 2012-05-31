//! \file PVLayerStackModel.cpp
//! $Id: PVLayerStackModel.cpp 3091 2011-06-09 06:13:05Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

// FIXME PhS : suppress that one
#include <pvkernel/core/PVAxisIndexType.h>



#include <PVLayerStackModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::PVLayerStackModel
 *
 *****************************************************************************/
PVInspector::PVLayerStackModel::PVLayerStackModel(PVMainWindow *mw, PVTabSplitter *parent) :
	QAbstractTableModel(parent),
	main_window(mw),
	parent_widget(parent),
	lib_view(parent_widget->get_lib_view()),
	lib_layer_stack(&lib_view->layer_stack)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackModel::%s : Creating object\n", __FUNCTION__);

	select_brush = QBrush(QColor(255,240,200));
	unselect_brush = QBrush(QColor(180,180,180));

	select_font = QFont();
	select_font.setBold(true);

	unselect_font = QFont();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::columnCount
 *
 *****************************************************************************/
int PVInspector::PVLayerStackModel::columnCount(const QModelIndex &index) const
{
	PVLOG_HEAVYDEBUG("PVInspector::PVLayerStackModel::%s : at row %d and column %d\n", __FUNCTION__, index.row(), index.column());
	return 3;
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVLayerStackModel::data(const QModelIndex &index, int role) const
{
	PVLOG_HEAVYDEBUG("PVInspector::PVLayerStackModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

	/* We prepare a direct acces to the total number of layers */
	int layer_count = lib_layer_stack->get_layer_count();
	/* We create and store the true index of the layer in the lib */
	int lib_index = layer_count -1 - index.row();

	switch (role) {
		case (Qt::CheckStateRole):
			switch (index.column()) {
				case 0:
					if (lib_view->get_layer_stack_layer_n_visible_state(lib_index)) {
						return Qt::Checked;
					}
					return Qt::Unchecked;
			}
			break;

		case (Qt::BackgroundRole):
				if (parent_widget && parent_widget->get_layer_stack_widget() && parent_widget->get_layer_stack_widget()->get_layer_stack_view()) {
					PVLayerStackView *layer_stack_view = parent_widget->get_layer_stack_widget()->get_layer_stack_view();
					/* testing */
					if (lib_layer_stack->get_selected_layer_index() == lib_index) {
						return QBrush(QColor(205,139,204));
					}
					if (layer_stack_view->mouse_hover_layer_index == index.row()) {
						return QBrush(QColor(200,200,200));
					}
					return QBrush(QColor(255,255,255));
				}

		case (Qt::DisplayRole):
			switch (index.column()) {
				case 0:
					return (int)lib_view->get_layer_stack_layer_n_visible_state(lib_index);

				case 1:
					return (int)lib_view->get_layer_stack_layer_n_locked_state(lib_index);

				case 2:
					return /*(char *)*/lib_view->get_layer_stack_layer_n_name(lib_index);
				// FIXME PhS : testing purposes only
				//case 3:
				//	return QTime(9, 50);
			}
			break;

		case (Qt::EditRole):
			switch (index.column()) {
				case 2:
					return /*(char *)*/lib_view->get_layer_stack_layer_n_name(lib_index);
			}
			break;

		case (Qt::TextAlignmentRole):
			return (Qt::AlignLeft + Qt::AlignVCenter);
			break;
	}
	return QVariant();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::emit_layoutChanged
 *
 *****************************************************************************/
void PVInspector::PVLayerStackModel::emit_layoutChanged()
{
	emit layoutChanged();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVInspector::PVLayerStackModel::flags(const QModelIndex &index) const
{
	PVLOG_HEAVYDEBUG("PVInspector::PVLayerStackModel::%s: at row %d and column %d\n", __FUNCTION__, index.row(), index.column());

	switch (index.column()) {
		case 0:
			//return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
			return Qt::ItemIsEditable | Qt::ItemIsEnabled;
			break;

		default:
			return (Qt::ItemIsEditable | Qt::ItemIsEnabled);
	}
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVLayerStackModel::headerData(int /*section*/, Qt::Orientation /*orientation*/, int role) const
{
	PVLOG_DEBUG("PVInspector::PVLayerStackModel::%s\n", __FUNCTION__);
	
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
 * PVInspector::PVLayerStackModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVLayerStackModel::rowCount(const QModelIndex &/*index*/) const
{
	PVLOG_HEAVYDEBUG("PVInspector::PVLayerStackModel::%s\n", __FUNCTION__);

	return lib_layer_stack->get_layer_count();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::setData
 *
 *****************************************************************************/
bool PVInspector::PVLayerStackModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

	/* We prepare a direct acces to the total number of layers */
	int layer_count = lib_layer_stack->get_layer_count();
	/* We create and store the true index of the layer in the lib */
	int lib_index = layer_count -1 - index.row();

	/* We prepare access to the layer_stack_view to resize columns */
	PVLayerStackView *layer_stack_view = parent_widget->get_layer_stack_widget()->get_layer_stack_view();

	switch (role) {
		case (Qt::EditRole):
			switch (index.column()) {
				case 0:
					/* this might be unnecessary */
					return true;
					break;

				case 2:
					lib_view->set_layer_stack_layer_n_name(lib_index, value.toByteArray().data());
					emit dataChanged(index, index);
					layer_stack_view->resizeColumnToContents(2);
					return true;
					break;

				default:
					return QAbstractTableModel::setData(index, value, role);
					break;
			}
			break;

		default:
			return QAbstractTableModel::setData(index, value, role);
	}

	return false;
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackModel::update_layer_stack
 *
 *****************************************************************************/
void PVInspector::PVLayerStackModel::update_layer_stack()
{
	beginResetModel();
	lib_view = parent_widget->get_lib_view();
	lib_layer_stack = &lib_view->layer_stack;
	endResetModel();
}
