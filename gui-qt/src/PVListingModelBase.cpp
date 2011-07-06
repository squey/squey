//! \file PVListingModelBase.cpp
//! $Id: PVListingModelBase.cpp 3244 2011-07-05 07:24:20Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <pvcore/general.h>
#include <picviz/PVView.h>
#include <picviz/state-machine.h>
#include <picviz/PVColor.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVListingModelBase.h>

/******************************************************************************
 *
 * PVInspector::PVListingModelBase::PVListingModelBase
 *
 *****************************************************************************/
PVInspector::PVListingModelBase::PVListingModelBase(PVMainWindow *mw, PVTabSplitter *parent) : QAbstractTableModel(parent)
{
	PVLOG_INFO("%s : Creating object\n", __FUNCTION__);
	main_window = mw;
	parent_widget = parent;

	select_brush = QBrush(QColor(255,240,200));
	unselect_brush = QBrush(QColor(180,180,180));

	select_font = QFont();
	select_font.setBold(true);

	unselect_font = QFont();
}

/******************************************************************************
 *
 * PVInspector::PVListingModelBase::columnCount
 *
 *****************************************************************************/
int PVInspector::PVListingModelBase::columnCount(const QModelIndex &index) const
{
	//PVLOG_DEBUG("PVInspector::PVListingModelBase::%s : at row %d and column %d\n", __FUNCTION__, index.row(), index.column());
	Picviz::PVView_p lib_view = parent_widget->get_lib_view();

	return lib_view->get_axes_count();
}

/******************************************************************************
 *
 * PVInspector::PVListingModelBase::emitLayoutChanged
 *
 *****************************************************************************/
void PVInspector::PVListingModelBase::emitLayoutChanged() {
	emit layoutChanged();
	PVLOG_DEBUG("PVInspector::PVListingModelBase::emitLayoutChanged\n");
}

/******************************************************************************
 *
 * PVInspector::PVListingModelBase::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVInspector::PVListingModelBase::flags(const QModelIndex &/*index*/) const
{
	//PVLOG_DEBUG("PVInspector::PVListingModelBase::%s\n", __FUNCTION__);
	return (Qt::ItemIsEnabled | Qt::ItemIsSelectable);
}

/******************************************************************************
 *
 * PVInspector::PVListingModelBase::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingModelBase::reset_model(bool initMatchTable)
{
  reset();
}
