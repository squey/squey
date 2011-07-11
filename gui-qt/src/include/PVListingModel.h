//! \file PVListingModel.h
//! $Id: PVListingModel.h 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGMODEL_H
#define PVLISTINGMODEL_H

#include <QtGui>
#include <QtCore>

//#include <PVListingModelBase.h>
#include <picviz/PVStateMachine.h>
#include <PVProgressBox.h>
#include <QAbstractTableModel>

#include <pvcore/general.h>
#include <picviz/PVSortQVectorQStringListThread.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingModel
 */
class PVListingModel : public QAbstractTableModel {
    Q_OBJECT

    QBrush not_zombie_font_brush; //!<
    QBrush zombie_font_brush; //!<
    //QVector<QStringList> widgetCpyOfData;
    //corresponding table between widgetCpyOfData and nrow
public:

    enum TypeOfSort {
        NoOrder, AscendingOrder, DescendingOrder
    };
protected:
	PVMainWindow  *main_window;     //!<
	PVTabSplitter *parent_widget;   //!<

	QBrush select_brush;            //!<
	QFont  select_font;             //!<
	QBrush unselect_brush;          //!<
	QFont  unselect_font;           //!<
private:
    //sorting data
    QVector<int> matchingTable; //!<the table sort, modify this array to order the values
    TypeOfSort sortOrder; //!<save the current sorting state (NoOrder, AscendingOrder, DescendingOrder)
    int colSorted; //!<save the last column whiche was used to sort
    Picviz::PVStateMachineListingMode_t state_listing; //!<this state indicate the mode of listing


public:


    /**
     * Constructor.
     *
     * @param mw
     * @param parent
     */
    PVListingModel(PVMainWindow *mw, PVTabSplitter *parent, Picviz::PVStateMachineListingMode_t state = Picviz::LISTING_ALL);

    /**
     * return data requested by the View
     * @param index
     * @param role
     * @return
     */
    QVariant data(const QModelIndex &index, int role) const;

    /**
     * return header requested by the View
     * @param section
     * @param orientation
     * @param role
     * @return 
     */
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    /**
     * 
     * @param index
     * @return the number of log line.
     */
    int rowCount(const QModelIndex &index) const;


    /**
     * Order to PVView to sort table
     * @param idColumn the id of the column to sort the table.
     */
    void sortByColumn(int idColumn);

    /**
     * not implemented
     * @param line
     * @return 
     */
    int getCorrespondance(int line);

    /**
     * initialize the matching table for sort.
     */
    void initMatchingTable();

    /**
     * reset the model
     * @param initMatchTable
     */
    virtual void reset_model(bool initMatchTable = true);

    /**
     * @brief set listing mode
     * @param mode
     */
    void setState(Picviz::PVStateMachineListingMode_t mode);


    
    
    /**
	*
	* @param index
	*
	* @return
	*/
	int columnCount(const QModelIndex &index) const;

	/**
	*
	* @param index
	*
	* @return
	*/
	Qt::ItemFlags flags(const QModelIndex &index) const;


	/**
	* call update for data
	*/
	void emitLayoutChanged(); 
};
}

#endif
