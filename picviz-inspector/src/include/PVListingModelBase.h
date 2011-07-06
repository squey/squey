//! \file PVListingModelBase.h
//! $Id: PVListingModelBase.h 3244 2011-07-05 07:24:20Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGMODELBASE_H
#define PVLISTINGMODELBASE_H

#include <QAbstractTableModel>

#include <QtGui>
#include <QtCore>

#include <pvcore/general.h>
#include <picviz/PVSortQVectorQStringListThread.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingModelBase
 */
class PVListingModelBase : public QAbstractTableModel
{
Q_OBJECT
protected:
	PVMainWindow  *main_window;     //!<
	PVTabSplitter *parent_widget;   //!<

	QBrush select_brush;            //!<
	QFont  select_font;             //!<
	QBrush unselect_brush;          //!<
	QFont  unselect_font;           //!<

public:
	enum TypeOfSort{
		NoOrder,AscendingOrder,DescendingOrder
	};
	/**
	* Constructor.
	*
	* @param mw
	* @param parent
	*/
	PVListingModelBase(PVMainWindow *mw, PVTabSplitter *parent);

	/**
	*
	* @param index
	*
	* @return
	*/
	int columnCount(const QModelIndex &index) const;

	/**
	*
	*/
	virtual void reset_model(bool initCorrespondTable=true);

	/**
	*
	* @param index
	* @param role
	*
	* @return
	*/
	virtual QVariant data(const QModelIndex &index, int role) const = 0; // Argh!

	/**
	*
	* @param section
	* @param orientation
	* @param role
	*
	* @return
	*/
	virtual QVariant headerData(int section, Qt::Orientation orientation, int role) const = 0; // Argh!!!

	/**
	*
	* @param index
	*
	* @return
	*/
	Qt::ItemFlags flags(const QModelIndex &index) const;

	/**
	*
	* @param index
	*
	* @return
	*/
	virtual int rowCount(const QModelIndex &index) const = 0;
	
	/**
	* sort the table.
	* @param idOfTheColumn
	*/
	virtual void sortByColumn(int ) {PVLOG_WARN("void sortByColumn(int ) from PVListingModelBase is not implemented.");};
	
	/**
	* 
	*/
	virtual void initCorrespondance( ) {PVLOG_WARN("void sortByColumn(int ) from PVListingModelBase is not implemented.");};
	
	/**
	* call update for data
	*/
	void emitLayoutChanged(); 
};
}

#endif
