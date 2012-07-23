/**
 * \file PVLayerStackWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtGui>

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVLayerStackWidget.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::PVLayerStackWidget
 *
 *****************************************************************************/
PVInspector::PVLayerStackWidget::PVLayerStackWidget(PVMainWindow *mw, PVLayerStackModel * model, PVTabSplitter *parent) : QWidget(parent)
{
	QVBoxLayout *main_layout;
	QToolBar    *layer_stack_toolbar;

	PVLOG_DEBUG("PVInspector::PVLayerStackWidget::%s\n", __FUNCTION__);

	main_window = mw;
	parent_tab = parent;

	// SIZE STUFF
	// WARNING: nothing should be set here.

	// OBJECTNAME STUFF
	setObjectName("PVLayerStackWidget");
	
	
	// LAYOUT STUFF
	// We need a Layout for that Widget
	main_layout = new QVBoxLayout(this);
	// We fix the margins for that Layout
	main_layout->setContentsMargins(0,0,0,0);
	
	// PVLAYERSTACKVIEW
	pv_layer_stack_view = NULL; // Note that this value can be requested during the creation of the PVLayerStackView
	pv_layer_stack_view = new PVLayerStackView(main_window, model, this);
	pv_layer_stack_view->setVisible(true);

	// TOOLBAR
	// We create the ToolBar of the PVLayerStackWidget
	layer_stack_toolbar = new QToolBar("Layer Stack ToolBar");
	layer_stack_toolbar->setObjectName("QToolBar_of_PVLayerStackWidget");
	// SIZE STUFF for the ToolBar
	layer_stack_toolbar->setMinimumWidth(185);
	layer_stack_toolbar->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Fixed);
	// And we fill the ToolBar
	create_actions(layer_stack_toolbar);

	// Now we can add our Widgets to the Layout
	main_layout->addWidget(pv_layer_stack_view);
	main_layout->addWidget(layer_stack_toolbar);

	setLayout(main_layout);
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::create_actions()
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::create_actions(QToolBar *toolbar)
{
	QAction *delete_layer_Action;
	QAction *duplicate_layer_Action;
	QAction *move_down_Action;
	QAction *move_up_Action;
	QAction *new_layer_Action;
	PVLOG_DEBUG("PVInspector::PVLayerStackWidget::%s\n", __FUNCTION__);

	
	// The new_layer Action
	new_layer_Action = new QAction(tr("New Layer"), this);
	new_layer_Action->setIcon(QIcon(":/new_layer_icon"));
	new_layer_Action->setStatusTip(tr("Create a new layer."));
	new_layer_Action->setWhatsThis(tr("Use this to create a new layer."));
	toolbar->addAction(new_layer_Action);
	connect(new_layer_Action, SIGNAL(triggered()), this, SLOT(new_layer_Slot()));

	// The move_up Action
	move_up_Action = new QAction(tr("Move up"), this);
	move_up_Action->setIcon(QIcon(":/move_layer_up_icon"));
	move_up_Action->setStatusTip(tr("Move selected layer up."));
	move_up_Action->setToolTip(tr("Move selected layer up."));
	move_up_Action->setWhatsThis(tr("Use this to move the selected layer up."));
	toolbar->addAction(move_up_Action);
	connect(move_up_Action, SIGNAL(triggered()), this, SLOT(move_up_Slot()));

	// The move_down Action
	move_down_Action = new QAction(tr("Move down"), this);
	move_down_Action->setIcon(QIcon(":/move_layer_down_icon"));
	move_down_Action->setStatusTip(tr("Move selected layer down."));
	move_down_Action->setToolTip(tr("Move selected layer down."));
	move_down_Action->setWhatsThis(tr("Use this to move the selected layer down."));
	toolbar->addAction(move_down_Action);
	connect(move_down_Action, SIGNAL(triggered()), this, SLOT(move_down_Slot()));

	// The duplicate_layer Action
	duplicate_layer_Action = new QAction(tr("Duplicate layer"), this);
	duplicate_layer_Action->setIcon(QIcon(":/duplicate_layer_icon"));
	duplicate_layer_Action->setStatusTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setToolTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setWhatsThis(tr("Use this to duplicate the selected layer."));
	toolbar->addAction(duplicate_layer_Action);
	connect(duplicate_layer_Action, SIGNAL(triggered()), this, SLOT(duplicate_layer_Slot()));

	// The delete_layer Action
	delete_layer_Action = new QAction(tr("Delete layer"), this);
	delete_layer_Action->setIcon(QIcon(":/delete_layer_icon"));
	delete_layer_Action->setStatusTip(tr("Delete layer."));
	delete_layer_Action->setToolTip(tr("Delete layer."));
	delete_layer_Action->setWhatsThis(tr("Use this to delete the selected."));
	toolbar->addAction(delete_layer_Action);
	connect(delete_layer_Action, SIGNAL(triggered()), this, SLOT(delete_layer_Slot()));
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::delete_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::delete_layer_Slot()
{
	PVLayerStackModel *layer_stack_model = parent_tab->get_layer_stack_model();

	layer_stack_model->get_layer_stack_lib().delete_selected_layer();
	refresh();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::duplicate_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::duplicate_layer_Slot()
{

}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::move_down_Slot
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::move_down_Slot()
{
	PVLayerStackModel *layer_stack_model = parent_tab->get_layer_stack_model();

	layer_stack_model->get_layer_stack_lib().move_selected_layer_down();
	refresh();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::move_up_Slot
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::move_up_Slot()
{
	PVLayerStackModel *layer_stack_model = parent_tab->get_layer_stack_model();

	layer_stack_model->get_layer_stack_lib().move_selected_layer_up();
	refresh();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::new_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::new_layer_Slot()
{
	PVLayerStackModel *layer_stack_model = parent_tab->get_layer_stack_model();

	layer_stack_model->get_layer_stack_lib().append_new_layer();
	refresh();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackWidget::refresh
 *
 *****************************************************************************/
void PVInspector::PVLayerStackWidget::refresh()
{
	parent_tab->get_lib_view()->process_from_layer_stack();
	parent_tab->refresh_layer_stack_view_Slot();
	main_window->update_pvglview(parent_tab->get_lib_view(), PVSDK_MESSENGER_REFRESH_Z|PVSDK_MESSENGER_REFRESH_COLOR|PVSDK_MESSENGER_REFRESH_ZOMBIES|PVSDK_MESSENGER_REFRESH_SELECTION|PVSDK_MESSENGER_REFRESH_SELECTED_LAYER);
}
