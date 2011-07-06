//! \file PVArgumentListWidget.h
//! $Id: PVArgumentListWidget.h 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVARGUMENTLISTWIDGET_H
#define PVARGUMENTLISTWIDGET_H

#include <QtCore>
#include <QDialog>
#include <QEvent>


#include <QTableView>
#include <QListWidget>
#include <QVariant>

#include <pvfilter/PVArgument.h>
#include <picviz/general.h>
#include <picviz/PVView.h>

#include <PVArgumentListModel.h>
#include <PVArgumentListDelegate.h>

namespace PVInspector {
class PVMainWindow;

class LibExport PVArgumentListWidget: public QDialog
{
	Q_OBJECT

public:
	PVArgumentListWidget(Picviz::PVView& view, PVFilter::PVArgumentList &args, QString const& filter_desc, QWidget* parent);
	virtual ~PVArgumentListWidget();
	bool eventFilter(QObject *obj, QEvent *event);

/* public slots: */
/* 	void widget_clicked_Slot(); */

private:
	QTableView*               _args_view;
	PVArgumentListModel*      _args_model;
	PVArgumentListDelegate*   _args_del;
	PVFilter::PVArgumentList& _args;
	Picviz::PVView&           _view;
};


}

#endif



