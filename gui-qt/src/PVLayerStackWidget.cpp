//! \file PVLayerStackWidget.cpp
//! $Id: PVLayerStackWidget.cpp 3196 2011-06-23 16:24:50Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

	main_layout = new QVBoxLayout(this);

	layer_stack_toolbar = new QToolBar("Layer Stack ToolBar");
	create_actions(layer_stack_toolbar);

	pv_layer_stack_view = NULL; // Note that this value can be requested during the creation of the PVLayerStackView
	pv_layer_stack_view = new PVLayerStackView(main_window, model, this);
	pv_layer_stack_view->setVisible(true);

	main_layout->addWidget(pv_layer_stack_view);
	main_layout->addWidget(layer_stack_toolbar);

	setFixedWidth(200);

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
	new_layer_Action->setIcon(QIcon(":/document-new.png"));
	new_layer_Action->setStatusTip(tr("Create a new layer."));
	new_layer_Action->setWhatsThis(tr("Use this to create a new layer."));
	toolbar->addAction(new_layer_Action);
	connect(new_layer_Action, SIGNAL(triggered()), this, SLOT(new_layer_Slot()));

	// The move_up Action
	move_up_Action = new QAction(tr("Move up"), this);
	move_up_Action->setIcon(QIcon(":/go-up.png"));
	move_up_Action->setStatusTip(tr("Move selected layer up."));
	move_up_Action->setToolTip(tr("Move selected layer up."));
	move_up_Action->setWhatsThis(tr("Use this to move the selected layer up."));
	toolbar->addAction(move_up_Action);
	connect(move_up_Action, SIGNAL(triggered()), this, SLOT(move_up_Slot()));

	// The move_down Action
	move_down_Action = new QAction(tr("Move down"), this);
	move_down_Action->setIcon(QIcon(":/go-down.png"));
	move_down_Action->setStatusTip(tr("Move selected layer down."));
	move_down_Action->setToolTip(tr("Move selected layer down."));
	move_down_Action->setWhatsThis(tr("Use this to move the selected layer down."));
	toolbar->addAction(move_down_Action);
	connect(move_down_Action, SIGNAL(triggered()), this, SLOT(move_down_Slot()));

	// The duplicate_layer Action
	duplicate_layer_Action = new QAction(tr("Duplicate layer"), this);
	duplicate_layer_Action->setIcon(QIcon(":/preferences-system-windows.png"));
	duplicate_layer_Action->setStatusTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setToolTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setWhatsThis(tr("Use this to duplicate the selected layer."));
	toolbar->addAction(duplicate_layer_Action);
	connect(duplicate_layer_Action, SIGNAL(triggered()), this, SLOT(duplicate_layer_Slot()));

	// The delete_layer Action
	delete_layer_Action = new QAction(tr("Delete layer"), this);
	delete_layer_Action->setIcon(QIcon(":/user-trash.png"));
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
	parent_tab->get_lib_view()->process_from_layer_stack();
	parent_tab->refresh_layer_stack_view_Slot();
	main_window->update_pvglview(parent_tab->get_lib_view(), PVGL_COM_REFRESH_Z|PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_ZOMBIES|PVGL_COM_REFRESH_SELECTION);
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
	parent_tab->get_lib_view()->process_from_layer_stack();
	parent_tab->refresh_layer_stack_view_Slot();
	main_window->update_pvglview(parent_tab->get_lib_view(), PVGL_COM_REFRESH_Z|PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_ZOMBIES|PVGL_COM_REFRESH_SELECTION);
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
	parent_tab->get_lib_view()->process_from_layer_stack();
	parent_tab->refresh_layer_stack_view_Slot();
	main_window->update_pvglview(parent_tab->get_lib_view(), PVGL_COM_REFRESH_Z|PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_ZOMBIES|PVGL_COM_REFRESH_SELECTION);
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
	parent_tab->get_lib_view()->process_from_layer_stack();
	parent_tab->refresh_layer_stack_view_Slot();
	main_window->update_pvglview(parent_tab->get_lib_view(), PVGL_COM_REFRESH_Z|PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_ZOMBIES|PVGL_COM_REFRESH_SELECTION);
}

