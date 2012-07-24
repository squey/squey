/**
 * \file PVLayerStackDelegate.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <cmath>

#include <QtGui>
#include <QEvent>
#include <QMetaType>
#include <QVariant>

#include <pvkernel/core/general.h>
//#include <pvkernel/core/PVAxisIndexEditor.h>
//#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVSelection.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <PVLayerStackView.h>
#include <PVMainWindow.h>

#include <PVLayerStackDelegate.h>


/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::PVLayerStackDelegate
 *
 *****************************************************************************/
PVInspector::PVLayerStackDelegate::PVLayerStackDelegate(PVMainWindow *mw, PVLayerStackView *parent) : QStyledItemDelegate(parent)
{
//	PVLOG_DEBUG("PVInspector::PVLayerStackDelegate::%s\n", __FUNCTION__);

	main_window = mw;
	layer_stack_view = parent;
	
//	// FIXME PhS : for testing purposes ! Should be removed
//	// The next three lines are crazy !! Without these lines QMetaType::type("PVCore::PVAxisIndexType")=0 !!
//	// AG: I think that this is because this check is done at runtime, abd so the type needs to be registered with qRegisterMetaType.
//	// Instead, qMetaTypeId<PVCore::PVAxisIndexEnditor>() must be used.
//	QVariant value = QVariant();;
//	PVCore::PVAxisIndexType test = PVCore::PVAxisIndexType();
//	value.setValue(test);
//	PVLOG_INFO(" WWWWW!!!! ici PVCore::PVAxisIndexType vaut : %d  \n", QMetaType::type("PVCore::PVAxisIndexType"));
//	// We set a dedicated QItemEditorFactory on this delegate
//	QItemEditorFactory *factory = new QItemEditorFactory;
////	QItemEditorCreatorBase *pv_axis_index_creator = new QStandardItemEditorCreator<PVCore::PVAxisIndexEditor>();//
//	factory->registerEditor((QVariant::Type)(QMetaType::type("PVCore::PVAxisIndexType")), pv_axis_index_creator);
//	setItemEditorFactory(factory);

	
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::createEditor
 *
 *****************************************************************************/
QWidget *PVInspector::PVLayerStackDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
	return QStyledItemDelegate::createEditor(parent, option, index);
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::editorEvent
 *
 *****************************************************************************/
bool PVInspector::PVLayerStackDelegate::editorEvent(QEvent *event, QAbstractItemModel * /*model*/, const QStyleOptionViewItem &/*option*/, const QModelIndex &index)
{
	Picviz::PVView_p      lib_view;
	Picviz::PVStateMachine *state_machine;
	int                   layer_count;
	int                   lib_index;

	if (event->type() == QEvent::MouseMove) {
		layer_stack_view->mouse_hover_layer_index = index.row();
	}

	/* We need an access to the lib_view */
	lib_view = main_window->current_tab->get_lib_view();
	/* We (might) need an access to the state machine */
	state_machine = lib_view->state_machine;

	/* We set a direct acces to the total number of layers */
	layer_count = lib_view->layer_stack.get_layer_count();
	/* We create and store the true index of the layer in the lib */
	lib_index = layer_count -1 - index.row();

	switch (index.column()) {
		case 0:
			if (event->type() == QEvent::MouseButtonPress) {
				lib_view->toggle_layer_stack_layer_n_visible_state(lib_index);
				/* We start by reprocessing only the layer_stack */
				lib_view->process_layer_stack();
				/* We might need to reprocess the volatile_selection */
				//if (picviz_state_machine_get_square_area_mode(state_machine) != PICVIZ_SM_SQUARE_AREA_MODE_OFF) {
				if (true) {
					/* square_area_selection is ACTIVE so we reprocess it */
					/* We do the selection on the layer_stack_output_layer's selection */
					lib_view->selection_A2B_select_with_square_area(lib_view->layer_stack_output_layer.get_selection(), lib_view->volatile_selection);
				}
				/* now we reprocess from the selection */
				lib_view->process_from_selection();
				// We refresh the PVGLView
				main_window->update_pvglview(lib_view, PVSDK_MESSENGER_REFRESH_Z|PVSDK_MESSENGER_REFRESH_COLOR|PVSDK_MESSENGER_REFRESH_ZOMBIES|PVSDK_MESSENGER_REFRESH_SELECTION);
				/* We must update all dynamic listing model... */
				//temp->current_pv_view->update_row_count_in_all_dynamic_listing_model_Slot();
				// FIXME: we should send ... well we should do something, probably!
				/* We refresh the listing */
				// FIXME !!! DDX We don't have a listing window anymore !temp->pv_ListingWindow->refresh_listing_Slot();
				return true;
			}
			return true;
			break;

		case 1:
			if (event->type() == QEvent::MouseButtonPress) {
				lib_view->toggle_layer_stack_layer_n_locked_state(lib_index);
				return true;
			}
			return true;
			break;

		case 2:
			if (event->type() == QEvent::MouseButtonPress) {
				lib_view->set_layer_stack_selected_layer_index(lib_index);
				main_window->update_pvglview(layer_stack_view->get_parent()->get_parent_tab()->get_lib_view(), PVSDK_MESSENGER_REFRESH_SELECTED_LAYER);
				/* We force a refresh of the layer_stack because of weird hover artifacts */
				main_window->current_tab->get_layer_stack_model()->emit_layoutChanged();
				return false;
			}
			return false;
			break;
	}
	return false;
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::paint
 *
 *****************************************************************************/
void PVInspector::PVLayerStackDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index ) const
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

		case 1:
			painter->save();
			painter->fillRect(option.rect, option.palette.base());

			painter->translate(option.rect.x() + 2, option.rect.y() + 2);
			if (index.data().toInt()) {
				painter->drawPixmap(0, 0, QPixmap(":/pv-linked-20"));
			} else {
				painter->drawPixmap(0, 0, QPixmap(":/pv-white-20"));
			}
			painter->restore();
			break;

		default:
		QStyledItemDelegate::paint(painter, option, index);
	}
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::setEditorData
 *
 *****************************************************************************/
void PVInspector::PVLayerStackDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
	QStyledItemDelegate::setEditorData(editor, index);
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::setModelData
 *
 *****************************************************************************/
void PVInspector::PVLayerStackDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
	QStyledItemDelegate::setModelData(editor, model, index);
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackDelegate::sizeHint
 *
 *****************************************************************************/
QSize PVInspector::PVLayerStackDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index ) const
{
	switch (index.column()) {
		case 0:
			return QSize(24,24);
			break;

		default:
			return QStyledItemDelegate::sizeHint(option, index);
			break;
	}
}


