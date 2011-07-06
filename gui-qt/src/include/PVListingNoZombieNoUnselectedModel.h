//! \file PVListingNoZombieNoUnselectedModel.h
//! $Id: PVListingNoZombieNoUnselectedModel.h 3240 2011-07-05 05:11:55Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGNOZOMBIENOUNSELECTEDMODEL_H
#define PVLISTINGNOZOMBIENOUNSELECTEDMODEL_H

#include <QtCore>
#include <QtGui>

#include <PVListingModelBase.h>
#include <PVProgressBox.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingNoZombieNoUnselectedModel
 */
class PVListingNoZombieNoUnselectedModel : public PVListingModelBase
{
Q_OBJECT

public:
	enum TypeOfSort{
		NoOrder,AscendingOrder,DescendingOrder
	};
	PVListingNoZombieNoUnselectedModel(PVMainWindow *mw, PVTabSplitter *parent);

	QVariant data(const QModelIndex &index, int role) const;
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

#endif // PVLISTINGNOZOMBIENOUNSELECTEDMODEL_H
