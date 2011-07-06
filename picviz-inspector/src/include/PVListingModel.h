//! \file PVListingModel.h
//! $Id: PVListingModel.h 3240 2011-07-05 05:11:55Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGMODEL_H
#define PVLISTINGMODEL_H

#include <QtGui>
#include <QtCore>

#include <PVListingModelBase.h>
#include <PVProgressBox.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingModel
 */
class PVListingModel : public PVListingModelBase
{
	
Q_OBJECT

	QBrush not_zombie_font_brush;   //!<
	QBrush zombie_font_brush;       //!<
	//QVector<QStringList> widgetCpyOfData;
	//corresponding table between widgetCpyOfData and nrow
	
	

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
	PVListingModel(PVMainWindow *mw, PVTabSplitter *parent);

	/**
	*
	* @param index
	* @param role
	*
	* @return
	*/
	QVariant data(const QModelIndex &index, int role) const;

	/**
	*
	* @param section
	* @param orientation
	* @param role
	*
	* @return
	*/
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;

	/**
	*
	* @param index
	*
	* @return
	*/
	int rowCount(const QModelIndex &index) const;
	
	
	/**
	* Order to PVView to sort table
	* @param idColumn the id of the column to sort the table.
	*/
	void sortByColumn(int idColumn);
	
	
	int getCorrespondance(int line);
	void initCorrespondance();
	virtual void reset_model(bool initCorrespondTable=true);
private:
	//sorting data
	QVector<int> correspondTable;
	TypeOfSort sortOrder;
	int colSorted;
	
};
}

#endif
