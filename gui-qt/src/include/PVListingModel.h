/**
 * \file PVListingModel.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLISTINGMODEL_H
#define PVLISTINGMODEL_H

#include <vector>
#include <utility>

#include <QAbstractTableModel>
#include <QBrush>
#include <QFont>
#include <QFontDatabase>
#include <QModelIndex>
#include <QReadWriteLock>

#include <pvkernel/core/general.h>
#include <picviz/PVSortQVectorQStringListThread.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVProgressBox.h>
#include <QAbstractTableModel>

#include <tbb/tbb_allocator.h>
#include <tbb/cache_aligned_allocator.h>

namespace PVInspector {

typedef std::vector<int> MatchingTable_t;

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
	std::vector<int, tbb::tbb_allocator<int> > localMatchingTable; //!<the table sort, modify this array to order the values
    QMutex localMatchingTable_locker;
	TypeOfSort sortOrder; //!<save the current sorting state (NoOrder, AscendingOrder, DescendingOrder)
	int colSorted; //!<save the last column whiche was used to sort
	
	QBrush not_zombie_font_brush; //!<
	QBrush zombie_font_brush; //!<

	Picviz::PVStateMachine *state_machine;
	Picviz::PVView* lib_view;
    

protected:
	PVMainWindow  *main_window;     //!<
	PVTabSplitter *parent_widget;   //!<
    MatchingTable_t *sortMatchingTable;
    
	QFontDatabase test_fontdatabase;
	QFont  row_header_font;
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

	//PVRow getRealRowIndex(PVRow model_row) const;

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
	
	void reset_lib_view();

protected:
	mutable QReadWriteLock _local_table_mutex;
	typedef std::vector<std::pair<PVRow, PVRow>, tbb::cache_aligned_allocator<std::pair<PVRow,PVRow> > > map_sort_t;
	map_sort_t _map_sort;
	
};
//MatchingTable_t PVInspector::PVListingModel::sortMatchingTable; //!<the table sort, modify this array to order the values
    

}

#endif
