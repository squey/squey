//! \file PVListingModel.cpp
//! $Id: PVListingModel.cpp 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <pvcore/general.h>

#include <picviz/PVView.h>
#include <picviz/PVColor.h>
#include <picviz/PVStateMachine.h>

#include <PVListingModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>


/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVInspector::PVListingModel::PVListingModel(PVMainWindow *mw, PVTabSplitter *parent) : QAbstractTableModel(parent) {
        PVLOG_INFO("%s : Creating object\n", __FUNCTION__);

        main_window = mw;
        parent_widget = parent;
	assert(parent_widget);

        select_brush = QBrush(QColor(255, 240, 200));
        unselect_brush = QBrush(QColor(180, 180, 180));
        select_font = QFont();
        select_font.setBold(true);
        unselect_font = QFont();
        not_zombie_font_brush = QBrush(QColor(0, 0, 0));
        zombie_font_brush = QBrush(QColor(200, 200, 200));
        colSorted = -1;

        lib_view = parent_widget->get_lib_view();
	state_machine = lib_view->state_machine;

        initMatchingTable();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::columnCount(const QModelIndex &index) const 
{
        return lib_view->get_axes_count();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::data(const QModelIndex &index, int role) const {
        PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

        Picviz::PVColor color;
        int i;
        int correspondId;
        int tmp_count = 0;
        int real_row_index;

        unsigned char r;
        unsigned char g;
        unsigned char b;

	if (state_machine->are_listing_all_visible()) {
		real_row_index = matchingTable.at(index.row());
	}
	if (state_machine->are_listing_none_visible()) {
		real_row_index = lib_view->get_nznu_real_row_index(matchingTable.at(index.row()));
	}
	if (!state_machine->are_listing_unselected_visible()) {
		real_row_index = lib_view->get_nu_real_row_index(matchingTable.at(index.row()));
	}
	if (!state_machine->are_listing_zombie_visible()) {
		real_row_index = lib_view->get_nz_real_row_index(matchingTable.at(index.row()));
	}

        correspondId = matchingTable.at(index.row());
        //PVLOG_DEBUG("           correspondId %d\n", correspondId);


        switch (role) {
                case (Qt::DisplayRole)://***********************************************DISPLAY**********************************************
                        //PVLOG_DEBUG("       DisplayRole\n");
                        return lib_view->get_data(correspondId, index.column());
                        break;

                case (Qt::TextAlignmentRole)://***********************************************TextAlignmentRole**********************************************
                        return (Qt::AlignLeft + Qt::AlignVCenter);
                        break;

                case (Qt::BackgroundRole)://***********************************************BackgroundRole**********************************************
                        //PVLOG_DEBUG("       ForegroundRole\n");
                        /* We get the current selected axis index */
                        i = lib_view->active_axis;



                        if ((state_machine->is_axes_mode()) && (i == index.column())) {
                                /* We must provide an evidence of the active_axis ! */
                                return QBrush(QColor(130, 100, 25));
                        } else {
                                if (lib_view->get_line_state_in_output_layer(correspondId)) {
                                        color = lib_view->get_color_in_output_layer(correspondId);
                                        r = color.r();
                                        g = color.g();
                                        b = color.b();

                                        return QBrush(QColor(r, g, b));
                                } else {
                                        return unselect_brush;
                                }
                        }


                        break;

                case (Qt::ForegroundRole)://***********************************************ForegroundRole**********************************************
			if (lib_view->layer_stack_output_layer.get_selection().get_line(real_row_index)) {
				PVLOG_INFO("NOT A ZOMBIE\n");
				/* The line is NOT a ZOMBIE */
				return not_zombie_font_brush;
			} else {
				PVLOG_INFO("THIS IS A ZOMBIE\n");
				/* The line is a ZOMBIE */
				return zombie_font_brush;
			}
        }//**********************************END************************************
        return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::emitLayoutChanged
 *
 *****************************************************************************/
void PVInspector::PVListingModel::emitLayoutChanged() {
        emit layoutChanged();
        PVLOG_DEBUG("PVInspector::PVListingModelBase::emitLayoutChanged\n");
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVInspector::PVListingModel::flags(const QModelIndex &/*index*/) const {
        //PVLOG_DEBUG("PVInspector::PVListingModelBase::%s\n", __FUNCTION__);
        return (Qt::ItemIsEnabled | Qt::ItemIsSelectable);
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
void PVInspector::PVListingModel::initMatchingTable() 
{
        PVLOG_DEBUG("PVListingModel::initCorrespondance()\n");

        //init the table of corresponding table.
        matchingTable.resize(0);

        if (state_machine->are_listing_all_visible() && matchingTable.size()!=lib_view->get_qtnraw_parent().size()) {
                for (unsigned int i = 0; i < lib_view->get_qtnraw_parent().size(); i++) {
                        matchingTable.insert(i, i);
                }
	} else {
		if (state_machine->are_listing_none_visible()) { // No zombies and no unselected
			for (int i = 0; i < rowCount(QModelIndex()); i++) {//for each line...
				matchingTable.insert(i, lib_view->get_nznu_real_row_index(i));
			}
		} else {
			if (!state_machine->are_listing_unselected_visible()) {
				for (int i = 0; i < rowCount(QModelIndex()); i++) {//for each line...
					matchingTable.insert(i, lib_view->get_nu_real_row_index(i));
				}
			}
			
			if (!state_machine->are_listing_zombie_visible()) {
				if(matchingTable.size()!=lib_view->get_qtnraw_parent().size()){
					for (int i = 0; i < rowCount(QModelIndex()); i++) {//for each line...
						matchingTable.insert(i, lib_view->get_nz_real_row_index(i));
					}
				}
			}
		} // if (state_machine->are_listing_none_visible())
        }

        sortOrder = NoOrder;
        emitLayoutChanged();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const {
        PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

        int real_row_index;

	// We need to get the data index to display vertical values
	if (!state_machine->are_listing_unselected_visible()) {
		real_row_index = lib_view->get_nu_real_row_index(section);
	}
	if (!state_machine->are_listing_zombie_visible()) {
		real_row_index = lib_view->get_nz_real_row_index(section);
	}
	if (state_machine->are_listing_none_visible()) {
		real_row_index = lib_view->get_nznu_real_row_index(section);
	}

        switch (role) {
                case (Qt::DisplayRole)://**********************************DisplayRole************************************
                        if (orientation == Qt::Horizontal) {
                                return QVariant(lib_view->get_axis_name(section));
                        } else {
                                return matchingTable.at(section) + 1;
                        }
                        break;
                case (Qt::FontRole)://**********************************FontRole************************************
			if (state_machine->are_listing_all_visible()) {
				if ((lib_view->real_output_selection.get_line(section)) && (orientation == Qt::Vertical)) {
					return select_font;
				} else {
					return unselect_font;
				}
			}
			if (state_machine->are_listing_none_visible()) {
				if (orientation == Qt::Vertical) {
					return select_font;
				} else {
					return unselect_font;
				}
			}
			if (!state_machine->are_listing_unselected_visible()) {
				if ((lib_view->real_output_selection.get_line(real_row_index)) && (orientation == Qt::Vertical)) {
					return select_font;
				} else {
					return unselect_font;
				}
			}
			if (!state_machine->are_listing_zombie_visible()) {
				if ((lib_view->real_output_selection.get_line(real_row_index)) && (orientation == Qt::Vertical)) {
					return select_font;
				} else {
					return unselect_font;
				}
			}

                        break;
                case (Qt::TextAlignmentRole)://**********************************TextAlignmentRole************************************
                        if (orientation == Qt::Horizontal) {
                                return (Qt::AlignLeft + Qt::AlignVCenter);
                        } else {
                                return (Qt::AlignRight + Qt::AlignVCenter);
                        }
                        break;

                default:
                        return QVariant();
        }//**********************************END************************************

        return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingModel::sortByColumn(int idColumn) 
{
        if ((idColumn < 0) || (idColumn >= columnCount(QModelIndex()))) {
                PVLOG_DEBUG("Cannot sort the column %d", idColumn);
                return;
        }
        //variables
        QVector<int> matchTableNew;
        Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0); //class whiche can sort.
        PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting...")); //dialog showing the progress box.
        connect(sortThread, SIGNAL(finished()), dialogBox, SLOT(accept()), Qt::QueuedConnection); //connection to close the progress box after thread finish.

	if (lib_view) {//if lib_view is valid...
		PVRush::PVNraw::nraw_table &data = lib_view->get_qtnraw_parent();
		if (state_machine->are_listing_all_visible()) {
			sortThread->setList(&data, &matchingTable);
		} else if (state_machine->are_listing_none_visible()) {
			for (int i = 0; i < rowCount(QModelIndex()); i++) {
				int real_row_index = lib_view->get_nznu_real_row_index(i);
				matchTableNew.insert(i, real_row_index);
			}
			sortThread->setList(&data, &matchTableNew);
		} else if (!state_machine->are_listing_unselected_visible()) {
			for (int i = 0; i < rowCount(QModelIndex()); i++) {
				matchTableNew.insert(i, lib_view->get_nu_real_row_index(i));
			}
			sortThread->setList(&data, &matchTableNew);
		} else if (!state_machine->are_listing_zombie_visible()) {
			for (int i = 0; i < rowCount(QModelIndex()); i++) {
				matchTableNew.insert(i, lib_view->get_nz_real_row_index(i));
			}
			sortThread->setList(&data, &matchTableNew);
		} 

		
		//find the good order to sort
		if ((colSorted == idColumn) && (sortOrder == AscendingOrder)) {
			sortOrder = DescendingOrder;
			sortThread->init(idColumn, Qt::DescendingOrder);
		} else {
			colSorted = idColumn;
			sortOrder = AscendingOrder;
			sortThread->init(idColumn, Qt::AscendingOrder);
		}

		sortThread->start(QThread::LowPriority);
		PVLOG_DEBUG("Waiting : sort processing... \n");

		//management of the progress box closing condition
		if (dialogBox->exec()) {//show dialog and wait for event
			//... update table
			sortThread->update();
		} else {//if we cancel during the sort...
			//... no update.
			//... stop the the thread.
			sortThread->exit(0);
		}
		PVLOG_DEBUG("   the sort is finished.\n");

		//*********update matching table*********
		PVLOG_DEBUG("   start update match...\n");
		if (!state_machine->are_listing_all_visible()) {
			matchingTable.resize(0);
			for (int i = 0; i < rowCount(QModelIndex()); i++) {
				matchingTable.insert(i, matchTableNew.at(i));
			}
		}
		
		PVLOG_DEBUG("   ...end update match\n");
		
		emit layoutChanged();
        }
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::rowCount(const QModelIndex &/*index*/) const 
{
	if (state_machine->are_listing_all_visible()) {
		return int(lib_view->get_row_count());
	}
	if (state_machine->are_listing_none_visible()) {
		return int(lib_view->get_nznu_index_count());
	}
	if (!state_machine->are_listing_unselected_visible()) {
		return int(lib_view->get_nu_index_count());
	}
	if (!state_machine->are_listing_zombie_visible()) {
		return int(lib_view->get_nz_index_count());
	}

	PVLOG_ERROR("Unknown listing visibility state!\n");
        return 0;
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::getCorrespondance
 *
 *****************************************************************************/
int PVInspector::PVListingModel::getMatch(int l) {
        PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);
        return matchingTable.at(l);
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingModel::reset_model(bool initMatchTable) {
        PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);
            reset();
        if (initMatchTable) {
                initMatchingTable();
        }
        emitLayoutChanged();
        //PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}

