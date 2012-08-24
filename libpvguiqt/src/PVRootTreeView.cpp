/**
 * \file PVViewsListingView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>
#include <pvhive/waxes/waxes.h>

#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <QMouseEvent>

PVGuiQt::PVRootTreeView::PVRootTreeView(PVRootTreeModel* model, QWidget* parent):
	QTreeView(parent)
{
	// Sizing
	setMinimumSize(100,0);
	setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);

	setHeaderHidden(true);
	setModel(model);
	setAllColumnsShowFocus(true);
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
	Picviz::PVSource_sp src(view->get_parent<Picviz::PVSource>()->shared_from_this());
	
	// Call select_view throught the Hive :)
	PVHive::call<FUNC(Picviz::PVSource::select_view)>(src, *view);
}
