/**
 * \file PVViewsListingView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVVIEWSLISTINGVIEW_H
#define PVVIEWSLISTINGVIEW_H

#include <pvkernel/core/general.h>
#include <QTreeView>

namespace PVInspector {

class PVViewsModel;
class PVTabSplitter;

class PVViewsListingView: public QTreeView
{
public:
	PVViewsListingView(PVViewsModel* model, PVTabSplitter* tab, QWidget* parent = 0);

protected:
	void mouseDoubleClickEvent(QMouseEvent* event);

protected:
	PVTabSplitter* _tab;
	PVViewsModel* _model;
};

}

#endif
