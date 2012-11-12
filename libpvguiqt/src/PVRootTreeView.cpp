/**
 * \file PVViewsListingView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>
#include <picviz/widgets/PVMappingPlottingEditDialog.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

#include <pvguiqt/PVQMapped.h>
#include <pvguiqt/PVQPlotted.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <QMenu>
#include <QMouseEvent>
#include <QAbstractItemModel>

PVGuiQt::PVRootTreeView::PVRootTreeView(QAbstractItemModel* model, QWidget* parent):
	QTreeView(parent)
{
	// Sizing
	setMinimumSize(100,0);
	setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);

	setHeaderHidden(true);
	setModel(model);
	setAllColumnsShowFocus(true);

	// Context menu
	setContextMenuPolicy(Qt::DefaultContextMenu);

	// Actions
	_act_new_view = new QAction(tr("Create new view..."), this);
	_act_new_plotted = new QAction(tr("Create new plotted..."), this);
	_act_new_mapped = new QAction(tr("Create new mapped..."), this);
	_act_edit_mapping = new QAction(tr("Edit mapping..."), this);
	_act_edit_plotting = new QAction(tr("Edit plotting..."), this);

	connect(_act_new_view, SIGNAL(triggered()), this, SLOT(create_new_view()));
	connect(_act_edit_plotting, SIGNAL(triggered()), this, SLOT(edit_plotting()));
	connect(_act_edit_mapping, SIGNAL(triggered()), this, SLOT(edit_mapping()));
}


void PVGuiQt::PVRootTreeView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);

	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		return;
	}

	PVCore::PVDataTreeObjectBase* obj = (PVCore::PVDataTreeObjectBase*) idx_click.internalPointer();
	Picviz::PVView* view = dynamic_cast<Picviz::PVView*>(obj);
	if (!view) {
		return;
	}

	// Double click on a view set this view as the current view of the parent source
	Picviz::PVRoot_sp root_sp(view->get_parent<Picviz::PVRoot>()->shared_from_this());
	
	// Call select_view throught the Hive :)
	PVHive::call<FUNC(Picviz::PVRoot::select_view)>(root_sp, *view);

	event->accept();
}

void PVGuiQt::PVRootTreeView::contextMenuEvent(QContextMenuEvent* event)
{
	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		return;
	}

	PVCore::PVDataTreeObjectBase* obj = (PVCore::PVDataTreeObjectBase*) idx_click.internalPointer();

	Picviz::PVPlotted* plotted = dynamic_cast<Picviz::PVPlotted*>(obj);
	if (plotted) {
		QMenu* ctxt_menu = new QMenu(this);
		ctxt_menu->addAction(_act_new_view);
		ctxt_menu->addAction(_act_edit_plotting);
		ctxt_menu->popup(QCursor::pos());
		return;
	}

	Picviz::PVMapped* mapped = dynamic_cast<Picviz::PVMapped*>(obj);
	if (mapped) {
		QMenu* ctxt_menu = new QMenu(this);
		ctxt_menu->addAction(_act_edit_mapping);
		ctxt_menu->popup(QCursor::pos());
		return;
	}
}

void PVGuiQt::PVRootTreeView::create_new_view()
{
	Picviz::PVPlotted* plotted = get_selected_obj_as<Picviz::PVPlotted>();
	if (plotted) {
		Picviz::PVPlotted_sp plotted_sp = plotted->shared_from_this();
		PVHive::call<FUNC(Picviz::PVPlotted::new_child_default)>(plotted_sp);
	}
}

void PVGuiQt::PVRootTreeView::edit_mapping()
{
	Picviz::PVMapped* mapped = get_selected_obj_as<Picviz::PVMapped>();
	if (mapped) {
		PVQMapped::edit_mapped(*mapped, this);
	}
}

void PVGuiQt::PVRootTreeView::edit_plotting()
{
	Picviz::PVPlotted* plotted = get_selected_obj_as<Picviz::PVPlotted>();
	if (plotted) {
		PVQPlotted::edit_plotted(*plotted, this);
	}
}

PVCore::PVDataTreeObjectBase* PVGuiQt::PVRootTreeView::get_selected_obj()
{
	QModelIndexList sel = selectedIndexes();
	if (sel.size() == 0) {
		return NULL;
	}
	return (PVCore::PVDataTreeObjectBase*) sel.at(0).internalPointer();
}

PVGuiQt::PVRootTreeModel* PVGuiQt::PVRootTreeView::tree_model()
{
	return qobject_cast<PVRootTreeModel*>(model());
}

PVGuiQt::PVRootTreeModel const* PVGuiQt::PVRootTreeView::tree_model() const
{
	return qobject_cast<PVRootTreeModel const*>(model());
}
