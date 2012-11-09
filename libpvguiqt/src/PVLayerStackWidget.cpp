/**
 * \file PVLayerStackWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QAction>
#include <QToolBar>
#include <QVBoxLayout>

#include <pvguiqt/PVLayerStackDelegate.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>
#include <pvguiqt/PVLayerStackWidget.h>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::PVLayerStackWidget
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackWidget::PVLayerStackWidget(Picviz::PVView_sp& lib_view, QWidget* parent):
	QWidget(parent)
{
	QVBoxLayout *main_layout;
	QToolBar    *layer_stack_toolbar;

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
	PVLayerStackModel* model = new PVLayerStackModel(lib_view);
	PVLayerStackDelegate* delegate = new PVLayerStackDelegate(*lib_view);
	_layer_stack_view = new PVLayerStackView();
	_layer_stack_view->setItemDelegate(delegate);
	_layer_stack_view->setModel(model);
	_layer_stack_view->resizeColumnsToContents();

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
	main_layout->addWidget(_layer_stack_view);
	main_layout->addWidget(layer_stack_toolbar);

	setLayout(main_layout);

	PVHive::PVObserverSignal<Picviz::PVView*>* obs = new PVHive::PVObserverSignal<Picviz::PVView*>(this);
	PVHive::get().register_observer(lib_view, *obs);
	obs->connect_about_to_be_deleted(this, SLOT(deleteLater()));
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::create_actions()
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::create_actions(QToolBar *toolbar)
{
	QAction *delete_layer_Action;
	QAction *duplicate_layer_Action;
	QAction *move_down_Action;
	QAction *move_up_Action;
	QAction *new_layer_Action;
	PVLOG_DEBUG("PVGuiQt::PVLayerStackWidget::%s\n", __FUNCTION__);
	
	// The new_layer Action
	new_layer_Action = new QAction(tr("New Layer"), this);
	new_layer_Action->setIcon(QIcon(":/new_layer_icon"));
	new_layer_Action->setStatusTip(tr("Create a new layer."));
	new_layer_Action->setWhatsThis(tr("Use this to create a new layer."));
	toolbar->addAction(new_layer_Action);
	connect(new_layer_Action, SIGNAL(triggered()), this, SLOT(new_layer()));

	// The move_up Action
	move_up_Action = new QAction(tr("Move up"), this);
	move_up_Action->setIcon(QIcon(":/move_layer_up_icon"));
	move_up_Action->setStatusTip(tr("Move selected layer up."));
	move_up_Action->setToolTip(tr("Move selected layer up."));
	move_up_Action->setWhatsThis(tr("Use this to move the selected layer up."));
	toolbar->addAction(move_up_Action);
	connect(move_up_Action, SIGNAL(triggered()), this, SLOT(move_up()));

	// The move_down Action
	move_down_Action = new QAction(tr("Move down"), this);
	move_down_Action->setIcon(QIcon(":/move_layer_down_icon"));
	move_down_Action->setStatusTip(tr("Move selected layer down."));
	move_down_Action->setToolTip(tr("Move selected layer down."));
	move_down_Action->setWhatsThis(tr("Use this to move the selected layer down."));
	toolbar->addAction(move_down_Action);
	connect(move_down_Action, SIGNAL(triggered()), this, SLOT(move_down()));

	// The duplicate_layer Action
	duplicate_layer_Action = new QAction(tr("Duplicate layer"), this);
	duplicate_layer_Action->setIcon(QIcon(":/duplicate_layer_icon"));
	duplicate_layer_Action->setStatusTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setToolTip(tr("Duplicate selected layer."));
	duplicate_layer_Action->setWhatsThis(tr("Use this to duplicate the selected layer."));
	toolbar->addAction(duplicate_layer_Action);
	connect(duplicate_layer_Action, SIGNAL(triggered()), this, SLOT(duplicate_layer()));

	// The delete_layer Action
	delete_layer_Action = new QAction(tr("Delete layer"), this);
	delete_layer_Action->setIcon(QIcon(":/delete_layer_icon"));
	delete_layer_Action->setStatusTip(tr("Delete layer."));
	delete_layer_Action->setToolTip(tr("Delete layer."));
	delete_layer_Action->setWhatsThis(tr("Use this to delete the selected."));
	toolbar->addAction(delete_layer_Action);
	connect(delete_layer_Action, SIGNAL(triggered()), this, SLOT(delete_layer()));
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::delete_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::delete_layer()
{
	ls_model()->delete_selected_layer();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::duplicate_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::duplicate_layer()
{

}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::move_down
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::move_down()
{
#if 0
	PVLayerStackModel *layer_stack_model = parent_tab->get_layer_stack_model();

	layer_stack_model->get_layer_stack_lib().move_selected_layer_down();
	refresh();
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::move_up
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::move_up()
{
#if 0
	PVLayerStackModel *layer_stack_model = parent_tab->get_layer_stack_model();

	layer_stack_model->get_layer_stack_lib().move_selected_layer_up();
	refresh();
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackWidget::new_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackWidget::new_layer()
{
	ls_model()->add_new_layer();
}

PVGuiQt::PVLayerStackModel* PVGuiQt::PVLayerStackWidget::ls_model()
{
	return _layer_stack_view->ls_model();
}
