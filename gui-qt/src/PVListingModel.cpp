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
PVInspector::PVListingModel::PVListingModel(PVMainWindow *mw, PVTabSplitter *parent, Picviz::PVStateMachineListingMode_t state) : QAbstractTableModel(parent) {
        PVLOG_INFO("%s : Creating object\n", __FUNCTION__);

        main_window = mw;
        parent_widget = parent;
        
        select_brush = QBrush(QColor(255, 240, 200));
        unselect_brush = QBrush(QColor(180, 180, 180));
        select_font = QFont();
        select_font.setBold(true);
        unselect_font = QFont();
        Picviz::PVView_p lib_view;
        not_zombie_font_brush = QBrush(QColor(0, 0, 0));
        zombie_font_brush = QBrush(QColor(200, 200, 200));
        lib_view = parent_widget->get_lib_view();
        colSorted = -1;
        
        state_listing = state;
        initMatchingTable();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::columnCount(const QModelIndex &) const {
        //PVLOG_DEBUG("PVInspector::PVListingModelBase::%s : at row %d and column %d\n", __FUNCTION__, index.row(), index.column());
        Picviz::PVView_p lib_view = parent_widget->get_lib_view();
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
        Picviz::PVView_p lib_view;
        int real_row_index;
        Picviz::PVStateMachine *state_machine;

        unsigned char r;
        unsigned char g;
        unsigned char b;
        lib_view = parent_widget->get_lib_view();
        state_machine = lib_view->state_machine;

        //initializing
        switch (state_listing) {
                case Picviz::LISTING_ALL:// we list all the lines***********************************************************
                        real_row_index = parent_widget->sortMatchingTable.at(index.row());
                        break;
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.***************************************
                        //real_row_index = sortMatchingTable.at(lib_view->get_nu_real_row_index(index.row()));
                        real_row_index = (lib_view->get_nu_real_row_index(index.row()));
                        break;
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.****************************************
                        //real_row_index = sortMatchingTable.at(lib_view->get_nz_real_row_index(index.row()));
                        real_row_index = lib_view->get_nz_real_row_index(index.row());
                        break;
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.****
                        //real_row_index = sortMatchingTable.at(lib_view->get_nznu_real_row_index(index.row()));
                        real_row_index = (lib_view->get_nznu_real_row_index(index.row()));
                        break;
                default:
                        PVLOG_ERROR("PVInspector::PVListingModel::data :  bad stat_listing.");
                        break;
        }

        PVLOG_HEAVYDEBUG("           correspondId %d\n", real_row_index);


        switch (role) {
                case (Qt::DisplayRole)://***********************************************DISPLAY**********************************************
                        //PVLOG_DEBUG("       DisplayRole\n");
                        return lib_view->get_data(real_row_index, index.column());
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
                                if (lib_view->get_line_state_in_output_layer(real_row_index)) {
                                        color = lib_view->get_color_in_output_layer(real_row_index);
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
                            //PVLOG_DEBUG("       ForegroundRole\n");
                        if (lib_view->layer_stack_output_layer.get_selection().get_line(real_row_index)) {
                                /* The line is NOT a ZOMBIE */
                                return not_zombie_font_brush;
                        } else {
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
void PVInspector::PVListingModel::initMatchingTable() {
        PVLOG_DEBUG("PVListingModel::initCorrespondance()\n");
        Picviz::PVView_p lib_view = parent_widget->get_lib_view();
        //init the table of corresponding table.
        if(lib_view) {
                //if the size of nraw is not the same as the matching table...
                if (lib_view->get_qtnraw_parent().size() != parent_widget->sortMatchingTable.size()) {
                        PVLOG_DEBUG("         init LISTING_ALL\n");
                        //...reinit the matching table.
                        parent_widget->sortMatchingTable.resize(0);
                        for (unsigned int i = 0; i < lib_view->get_qtnraw_parent().size(); i++) {
                                parent_widget->sortMatchingTable.push_back(i);
                        }
                        parent_widget->sortMatchingTable_invert.resize(parent_widget->sortMatchingTable.size());
                        for(int i=0;i<parent_widget->sortMatchingTable.size();i++){
                                int j=parent_widget->sortMatchingTable.at(i);
                                parent_widget->sortMatchingTable_invert.at(j)=i;
                        }

                        sortOrder = NoOrder;//... reset the last order remember
                        emitLayoutChanged();//... notify to the view that data has changed
                }
        }
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const {
        PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

        unsigned int real_row_index;
        Picviz::PVView_p lib_view = parent_widget->get_lib_view();

        //
        switch (state_listing) {
                case Picviz::LISTING_ALL:// we list all the lines
                        real_row_index = section;
                        break;
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                        real_row_index = parent_widget->sortMatchingTable_invert.at(lib_view->get_nu_real_row_index(section));
                        break;
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                        real_row_index = parent_widget->sortMatchingTable_invert.at(lib_view->get_nz_real_row_index(section));
                        break;
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                        real_row_index = parent_widget->sortMatchingTable_invert.at(lib_view->get_nznu_real_row_index(section));
                        break;
                default:
                        PVLOG_ERROR("   bad state_listing");
                        break;
        }
        real_row_index = parent_widget->sortMatchingTable.at(real_row_index);

        switch (role) {
                case (Qt::DisplayRole)://**********************************DisplayRole************************************
                        if (orientation == Qt::Horizontal) {
                                return QVariant(lib_view->get_axis_name(section));
                        } else {
                                //return real_row_index + 1;
                                return real_row_index ;///TODO replace by prev after debug
                        }
                        break;
                case (Qt::FontRole)://**********************************FontRole************************************
                        if ((lib_view->real_output_selection.get_line(real_row_index)) && (orientation == Qt::Vertical)) {
                                return select_font;
                        } else {
                                return unselect_font;
                        }
                        break;
                  


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
 * PVInspector::PVListingModel::setState
 *
 *****************************************************************************/
void PVInspector::PVListingModel::setState(Picviz::PVStateMachineListingMode_t mode) {
        PVLOG_DEBUG("PVInspector::PVListingModel::setState(%d)\n", (int) mode);
        state_listing = mode;
        initMatchingTable();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingModel::sortByColumn(int idColumn) {
        PVLOG_INFO("PVInspector::PVListingModel::sortByColumn(%d)\n", idColumn);

        if ((idColumn < 0) || (idColumn >= columnCount(QModelIndex()))) {
                PVLOG_DEBUG("     can't sort the column %d\n",idColumn);
                return;
        }
        
        //variables init
        Picviz::PVView_p lib_view;
        Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0); //class whiche can sort.
        PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting...")); //dialog showing the progress box.
        connect(sortThread, SIGNAL(finished()), dialogBox, SLOT(accept()), Qt::QueuedConnection); //connection to close the progress box after thread finish.
        PVLOG_DEBUG("   declaration ok\n");


        if (parent_widget != 0) {//if parent widget is valid...
                //get the view
                lib_view = parent_widget->get_lib_view();
                if (lib_view) {//if lib_view is valid...
                        PVRush::PVNraw::nraw_table &data = lib_view->get_qtnraw_parent();

                        //*********init sort**********
                        PVLOG_DEBUG("   init sort\n");
                        sortThread->setList(&data, &parent_widget->sortMatchingTable);
                        PVLOG_DEBUG("   init sort finished\n");
                        //find the good order to sort
                        if ((colSorted == idColumn) && (sortOrder == AscendingOrder)) {
                                sortOrder = DescendingOrder;
                                sortThread->init(idColumn, Qt::DescendingOrder);
                        } else {
                                colSorted = idColumn;
                                sortOrder = AscendingOrder;
                                sortThread->init(idColumn, Qt::AscendingOrder);
                        }
                        

                        //thread sorter start here
                        PVLOG_DEBUG("   the sort will start in a thread.\n");
                        sortThread->start(QThread::LowPriority);
                        PVLOG_INFO("    waitting : sort processing... \n");

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
                        for(int i=0;i<parent_widget->sortMatchingTable.size();i++){
                                int j=parent_widget->sortMatchingTable.at(i);
                                parent_widget->sortMatchingTable_invert.at(j)=i;
                        }

                        emit layoutChanged();
                } else {//if lib_view isn't valid...
                        PVLOG_ERROR("   no lib_view : %s : %d\n", __FILE__, __LINE__);
                }
        } else {//if parent widget isn't valid...
                PVLOG_ERROR("   no parent widget : %s : %d\n", __FILE__, __LINE__);
        }
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::rowCount(const QModelIndex &/*index*/) const {
        //PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);
        Picviz::PVView_p lib_view = parent_widget->get_lib_view();
        switch (state_listing) {
                case Picviz::LISTING_ALL:// we list all the lines
                        return int(lib_view->get_row_count());
                        break;
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                        return int(lib_view->get_nu_index_count());
                        break;
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                        return int(lib_view->get_nz_index_count());
                        break;
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                        return int(lib_view->get_nznu_index_count());
                        break;
	         case Picviz::LISTING_BAD_LISTING_MODE:
			 break;
        }
        PVLOG_ERROR("PVInspector::PVListingModel::rowCount :  bad stat_listing.");


        return 0;
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::getCorrespondance
 *
 *****************************************************************************/
unsigned int PVInspector::PVListingModel::getMatch(unsigned int l) {
        PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);
        return int(parent_widget->sortMatchingTable.at(l));
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

