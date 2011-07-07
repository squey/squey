//! \file PVListingNoUnselectedModel.h
//! $Id: PVListingNoUnselectedModel.h 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGNOUNSELECTEDMODEL_H
#define PVLISTINGNOUNSELECTEDMODEL_H

#include <QtGui>
#include <QtCore>

#include <PVListingModelBase.h>
#include <PVProgressBox.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingNoUnselectedModel
 *
 */
class PVListingNoUnselectedModel : public PVListingModelBase
{
Q_OBJECT

	QBrush not_zombie_font_brush;
	QBrush zombie_font_brush;

public:
	enum TypeOfSort{
		NoOrder,AscendingOrder,DescendingOrder
	};
	PVListingNoUnselectedModel(PVMainWindow *mw, PVTabSplitter *parent);

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
	QVector<int> matchingTable;
	TypeOfSort sortOrder;
	int colSorted;
};
}

#endif // PVLISTINGNOUNSELECTEDMODEL_H
