//! \file PVListingNoZombieModel.h
//! $Id: PVListingNoZombieModel.h 3248 2011-07-05 10:15:19Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGNOZOMBIEMODEL_H
#define PVLISTINGNOZOMBIEMODEL_H

#include <QtCore>
#include <QtGui>

#include <PVListingModelBase.h>
#include <PVProgressBox.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 *  \class PVListingNoZombieModel
 */
class PVListingNoZombieModel : public PVListingModelBase
{
Q_OBJECT

public:
	enum TypeOfSort{
		NoOrder,AscendingOrder,DescendingOrder
	};
	PVListingNoZombieModel(PVMainWindow *mw, PVTabSplitter *parent);

	virtual QVariant data(const QModelIndex &index, int role) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	int rowCount(const QModelIndex &index) const;
	
	/**
	* Order to PVView to sort table
	* @param idColumn the id of the column to sort the table.
	*/
	void sortByColumn(int idColumn);
	
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
