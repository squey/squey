//! \file PVListingModel.h
//! $Id: PVListingModel.h 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011
#ifndef PVLISTINGMODEL_H
#define PVLISTINGMODEL_H

#include <vector>
typedef std::vector<int> MatchingTable_t;

#include <QtGui>
#include <QtCore>

#include <pvcore/general.h>
#include <picviz/PVSortQVectorQStringListThread.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <PVProgressBox.h>
#include <QAbstractTableModel>

#include <tbb/scalable_allocator.h>

namespace PVInspector {


class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingModel
 */

class PVListingModel : public QAbstractTableModel {
Q_OBJECT


public:
    enum TypeOfSort {
        NoOrder, AscendingOrder, DescendingOrder
    };

private:
	//sorting data
	std::vector<int, tbb::scalable_allocator<int> > localMatchingTable; //!<the table sort, modify this array to order the values
    QMutex localMatchingTable_locker;
	TypeOfSort sortOrder; //!<save the current sorting state (NoOrder, AscendingOrder, DescendingOrder)
	int colSorted; //!<save the last column whiche was used to sort
	
	QBrush not_zombie_font_brush; //!<
	QBrush zombie_font_brush; //!<

	Picviz::PVStateMachine *state_machine;
	Picviz::PVView_p lib_view;
    

protected:
	PVMainWindow  *main_window;     //!<
	PVTabSplitter *parent_widget;   //!<
    MatchingTable_t *sortMatchingTable;

	QBrush select_brush;            //!<
	QFont  select_font;             //!<
	QBrush unselect_brush;          //!<
	QFont  unselect_font;           //!<

public:
    /**
     * Constructor.
     *
     * @param mw
     * @param parent
     */
    PVListingModel(PVMainWindow *mw, PVTabSplitter *parent);

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

	PVRow getRealRowIndex(PVRow model_row) const;

    /**
     * Order to PVView to sort table
     * @param idColumn the id of the column to sort the table.
     */
    void sortByColumn(int idColumn);
    
    /**
     * @param line
     * @return 
     */
    unsigned int getInvertedMatch(unsigned int line);

    /**
     * @param line
     * @return 
     */
    unsigned int getLocalMatch(unsigned int line);

    /**
     * @param line
     * @return 
     */
    unsigned int getMatch(unsigned int line);

    /**
     * initialize the matching table for sort.
     */
    void initMatchingTable();
    
    /**
     * create a new matching table for nu, nz or nunz situation.
     */
    void initLocalMatchingTable();

    /**
     * reset the model
     * @param initMatchTable
     */
    virtual void reset_model(bool initMatchTable = true);

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

protected:
	mutable QReadWriteLock _local_table_mutex;
};
//MatchingTable_t PVInspector::PVListingModel::sortMatchingTable; //!<the table sort, modify this array to order the values
    

}

#endif
