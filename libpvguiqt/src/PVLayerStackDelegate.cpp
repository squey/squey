/**
 * \file PVLayerStackDelegate.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <cmath>

#include <QEvent>
#include <QMetaType>
#include <QVariant>

#include <picviz/PVSelection.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <pvguiqt/PVLayerStackDelegate.h>

#include <QPainter>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::PVLayerStackDelegate
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackDelegate::PVLayerStackDelegate(Picviz::PVView const& view, QObject* parent):
	QStyledItemDelegate(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::createEditor
 *
 *****************************************************************************/
QWidget *PVGuiQt::PVLayerStackDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
	return QStyledItemDelegate::createEditor(parent, option, index);
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::editorEvent
 *
 *****************************************************************************/
bool PVGuiQt::PVLayerStackDelegate::editorEvent(QEvent *event, QAbstractItemModel* model, const QStyleOptionViewItem &/*option*/, const QModelIndex &index)
{
	/*if (event->type() == QEvent::MouseMove) {
		layer_stack_view->mouse_hover_layer_index = index.row();
	}*/

	switch (index.column()) {
		case 0:
			if (event->type() == QEvent::MouseButtonPress) {
				model->setData(index, QVariant());
				/* We start by reprocessing only the layer_stack */
#if 0
				lib_view->process_layer_stack();
				/* We might need to reprocess the volatile_selection */
				//if (picviz_state_machine_get_square_area_mode(state_machine) != PICVIZ_SM_SQUARE_AREA_MODE_OFF) {
				/* square_area_selection is ACTIVE so we reprocess it */
				/* We do the selection on the layer_stack_output_layer's selection */
				lib_view->selection_A2B_select_with_square_area(lib_view->layer_stack_output_layer.get_selection(), lib_view->volatile_selection);
			}
			/* now we reprocess from the selection */
			lib_view->process_from_selection();
#endif
				return true;
			}
			break;

		/*case 1:
			if (event->type() == QEvent::MouseButtonPress) {
				model->setData(index, QVariant());
				//lib_view->toggle_layer_stack_layer_n_locked_state(lib_index);
				return true;
			}
			break;*/
	}
	return false;
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::paint
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index ) const
{
	switch (index.column()) {
		case 0:
			painter->save();
			painter->fillRect(option.rect, option.palette.base());

			painter->translate(option.rect.x() + 2, option.rect.y() + 2);
			if (index.data().toInt()) {
				painter->drawPixmap(0, 0, QPixmap(":/layer-active.png"));
			} else {
				painter->drawPixmap(0, 0, QPixmap(":/layer-inactive.png"));
			}
			painter->restore();
			break;

		/*case 1:
			painter->save();
			painter->fillRect(option.rect, option.palette.base());

			painter->translate(option.rect.x() + 2, option.rect.y() + 2);
			if (index.data().toInt()) {
				painter->drawPixmap(0, 0, QPixmap(":/pv-linked-20"));
			} else {
				painter->drawPixmap(0, 0, QPixmap(":/pv-white-20"));
			}
			painter->restore();
			break;*/

		default:
			QStyledItemDelegate::paint(painter, option, index);
	}
}



/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::setEditorData
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
	QStyledItemDelegate::setEditorData(editor, index);
}



/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::setModelData
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
	QStyledItemDelegate::setModelData(editor, model, index);
}



/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::sizeHint
 *
 *****************************************************************************/
QSize PVGuiQt::PVLayerStackDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index ) const
{
	switch (index.column()) {
		case 0:
			return QSize(24, 24);
			break;

		default:
			return QStyledItemDelegate::sizeHint(option, index);
			break;
	}
}

