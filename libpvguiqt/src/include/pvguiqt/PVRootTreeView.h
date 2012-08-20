/**
 * \file PVViewsListingView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVVIEWSLISTINGVIEW_H
#define PVVIEWSLISTINGVIEW_H

#include <pvkernel/core/general.h>
#include <QTreeView>

namespace PVGuiQt {

class PVRootTreeModel;
class PVTabSplitter;

class PVRootTreeView: public QTreeView
{
public:
	PVRootTreeView(PVRootTreeModel* model, QWidget* parent = 0);

protected:
	void mouseDoubleClickEvent(QMouseEvent* event);

protected:
	PVRootTreeModel* tree_model() { return static_cast<PVRootTreeModel*>(model()); }
	PVRootTreeModel const* tree_model() const { return static_cast<PVRootTreeModel const*>(model()); }

};

}

#endif
