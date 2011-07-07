//! \file PVListingModel.cpp
//! $Id: PVListingModel.cpp 3252 2011-07-07 03:41:16Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <pvcore/general.h>

#include <picviz/PVView.h>
#include <picviz/PVColor.h>


#include <PVListingModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>


/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVInspector::PVListingModel::PVListingModel(PVMainWindow *mw, PVTabSplitter *parent, Picviz::StateMachine_ListingMode_t state) : PVListingModelBase(mw, parent) {
    Picviz::PVView_p lib_view;

    PVLOG_INFO("%s : Creating object\n", __FUNCTION__);

    not_zombie_font_brush = QBrush(QColor(0, 0, 0));
    zombie_font_brush = QBrush(QColor(200, 200, 200));

    lib_view = parent_widget->get_lib_view();
    //widgetCpyOfData = (const QVector<QStringList>&) lib_view->get_qtnraw_parent();

    colSorted = -1;
    state_listing = state;

    initCorrespondance();

}


/******************************************************************************
 *
 * PVInspector::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::data(const QModelIndex &index, int role) const {
    PVLOG_DEBUG("PVInspector::PVListingModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

    Picviz::PVColor color;
    int i;
    int correspondId;
    int tmp_count=0;
    Picviz::PVView_p lib_view;
    int real_row_index;
    Picviz::StateMachine *state_machine;

    unsigned char r;
    unsigned char g;
    unsigned char b;
    lib_view = parent_widget->get_lib_view();
    state_machine = lib_view->state_machine;

    //initializing
    switch (state_listing) {
        case Picviz::LISTING_ALL :// we list all the lines***********************************************************
            //nop
            break;
        case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.***************************************
            real_row_index = lib_view->get_nu_real_row_index(matchingTable.at(index.row()));
            break;
        case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.****************************************
            real_row_index = lib_view->get_nz_real_row_index(matchingTable.at(index.row()));
            break;
        case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.****
            real_row_index = lib_view->get_nznu_real_row_index(matchingTable.at(index.row()));
            break;
        default:
            PVLOG_ERROR("PVInspector::PVListingModel::data :  bad stat_listing.");
            break;
    }
    correspondId = matchingTable.at(index.row());
    PVLOG_DEBUG("           correspondId %d\n", correspondId);


    switch (role) {
        case (Qt::DisplayRole)://***********************************************DISPLAY**********************************************
            PVLOG_DEBUG("       DisplayRole\n");
            return lib_view->get_data(correspondId, index.column());
            break;

        case (Qt::TextAlignmentRole)://***********************************************TextAlignmentRole**********************************************
            return (Qt::AlignLeft + Qt::AlignVCenter);
            break;

        case (Qt::BackgroundRole)://***********************************************BackgroundRole**********************************************
            PVLOG_DEBUG("       ForegroundRole\n");
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
            PVLOG_DEBUG("       ForegroundRole\n");
            switch (state_listing) {
                case Picviz::LISTING_ALL:// we list all the lines
                    /* We test if the line is a ZOMBIE one */
                    //correspondId = correspondTable.at(index.row());
                    if (lib_view->layer_stack_output_layer.get_selection().get_line(index.row())) {
                        /* The line is NOT a ZOMBIE */
                        return not_zombie_font_brush;
                    } else {
                        /* The line is a ZOMBIE */
                        return zombie_font_brush;
                    }
                    break;
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                    /* We test if the line is a ZOMBIE one */
                    if (lib_view->layer_stack_output_layer.get_selection().get_line(real_row_index)) {
                        /* The line is NOT a ZOMBIE */
                        return not_zombie_font_brush;
                    } else {
                        /* The line is a ZOMBIE */
                        return zombie_font_brush;
                    }
                    break;
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                    break;
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                    break;
            }

    }//**********************************END************************************
    return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
void PVInspector::PVListingModel::initCorrespondance() {
    PVLOG_INFO("PVListingModel::initCorrespondance()\n");
    Picviz::PVView_p lib_view = parent_widget->get_lib_view();
    //init the table of corresponding table.
    matchingTable.resize(0);
    if (state_listing == Picviz::LISTING_ALL) {
        for (unsigned int i = 0; i < lib_view->get_qtnraw_parent().size(); i++) {
            matchingTable.insert(i, i);
        }
    } else if(state_listing == Picviz::LISTING_NO_UNSEL||state_listing == Picviz::LISTING_NO_ZOMBIES||state_listing == Picviz::LISTING_NO_UNSEL_NO_ZOMBIES) {
        for (int i = 0; i < rowCount(QModelIndex()); i++) {//for each line...
            switch (state_listing) {
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                    matchingTable.insert(i, lib_view->get_nu_real_row_index(i));
                    break;
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                    matchingTable.insert(i, lib_view->get_nz_real_row_index(i));
                    break;
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                    matchingTable.insert(i, lib_view->get_nznu_real_row_index(i));
                    break;
            }
        }
    }else{
        PVLOG_ERROR("PVInspector::PVListingModel::initCorrespondance : initializing with a bad stat_listing.");
    }
    sortOrder = NoOrder;
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const {
    PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

    int real_row_index;
    Picviz::PVView_p lib_view = parent_widget->get_lib_view();

    switch (state_listing) {
        case Picviz::LISTING_ALL:// we list all the lines
            break;
        case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
            // We compute the real row index 
            real_row_index = lib_view->get_nu_real_row_index(section);
            break;
        case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
            real_row_index = lib_view->get_nz_real_row_index(section);
            break;
        case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
            real_row_index = lib_view->get_nznu_real_row_index(section);
            break;
        default:
            PVLOG_ERROR("   bad state_listing");
            break;
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
            switch (state_listing) {
                case Picviz::LISTING_ALL:// we list all the lines
                    if ((lib_view->real_output_selection.get_line(section)) && (orientation == Qt::Vertical)) {
                        return select_font;
                    } else {
                        return unselect_font;
                    }
                    break;
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                    if ((lib_view->real_output_selection.get_line(real_row_index)) && (orientation == Qt::Vertical)) {
                        return select_font;
                    } else {
                        return unselect_font;
                    }
                    break;
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                    if ((lib_view->real_output_selection.get_line(real_row_index)) && (orientation == Qt::Vertical)) {
                        return select_font;
                    } else {
                        return unselect_font;
                    }
                    break;
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                    if (orientation == Qt::Vertical) {
                        return select_font;
                    } else {
                        return unselect_font;
                    }
                    break;
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
 * PVInspector::PVListingModel::setState
 *
 *****************************************************************************/
void PVInspector::PVListingModel::setState(Picviz::StateMachine_ListingMode_t mode) {
    PVLOG_INFO("PVInspector::PVListingModel::setState(%d)\n", (int)mode);
    state_listing = mode;
    initCorrespondance();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingModel::sortByColumn(int idColumn) {
    PVLOG_INFO("PVInspector::PVListingModel::sortByColumn(%d)\n", idColumn);
    //variables
    Picviz::PVView_p lib_view;
    QVector<int> matchTableNew;
    Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0);//class whiche can sort.
    PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting..."));//dialog showing the progress box.
    connect(sortThread, SIGNAL(finished()), dialogBox, SLOT(accept()), Qt::QueuedConnection);//connection to close the progress box after thread finish.
    PVLOG_DEBUG("   declaration ok\n");
    
    
    if (parent_widget != 0) {//if parent widget is valid...
        lib_view = parent_widget->get_lib_view();
        if (lib_view) {//if lib_view is valid...
            PVRush::PVNraw::nraw_table &data = lib_view->get_qtnraw_parent();

            //*********init sort**********
            {
                PVLOG_DEBUG("   init sort\n");
                switch (state_listing) {
                    case Picviz::LISTING_ALL:// we list all the lines
                        sortThread->setList(&data, &matchingTable);
                        break;
                    case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                        for (int i = 0; i < rowCount(QModelIndex()); i++) {
                             matchTableNew.insert(i, lib_view->get_nu_real_row_index(i));
                        }
                        sortThread->setList(&data, &matchTableNew);
                        break;
                    case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                        for (int i = 0; i < rowCount(QModelIndex()); i++) {
                            matchTableNew.insert(i, lib_view->get_nz_real_row_index(i));
                        }
                        sortThread->setList(&data, &matchTableNew);
                        break;
                    case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                        for (int i = 0; i < rowCount(QModelIndex()); i++) {
                            int real_row_index = lib_view->get_nznu_real_row_index(i);
                            matchTableNew.insert(i, real_row_index);
                        }
                        sortThread->setList(&data, &matchTableNew);
                        break;
                    default:
                        PVLOG_ERROR("PVInspector::PVListingModel::sortByColumn : bad stat_listing.");
                        break;
                }
                PVLOG_DEBUG("   init sort finished\n");
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
            
            //*********update matching table*********
            PVLOG_DEBUG("   start update match...\n");
            switch (state_listing) {
                case Picviz::LISTING_ALL:// we list all the lines
                    break;
                case Picviz::LISTING_NO_UNSEL:// we don't list the unselected lines.
                case Picviz::LISTING_NO_ZOMBIES:// we don't list the zombies lines.
                case Picviz::LISTING_NO_UNSEL_NO_ZOMBIES:// we don't list the zombies lines and the unselected lines.
                    matchingTable.resize(0);
                    for (int i = 0; i < rowCount(QModelIndex()); i++) {
                        matchingTable.insert(i, matchTableNew.at(i));
                    }
                    break;
            }
            PVLOG_DEBUG("   ...end update match\n");
            
            emit layoutChanged();
        } else {//if lib_view isn't valid...
            PVLOG_ERROR("   no lib_view : %s : %d", __FILE__, __LINE__);
        }
    } else {//if parent widget isn't valid...
        PVLOG_ERROR("   no parent widget : %s : %d", __FILE__, __LINE__);
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
    }
    PVLOG_ERROR("PVInspector::PVListingModel::rowCount :  bad stat_listing.");
    return 0;
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::getCorrespondance
 *
 *****************************************************************************/
int getCorrespondance(int ) {
    return 0;
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingModel::reset_model(bool initMatchTable) {
    PVListingModelBase::reset_model();
    if (initMatchTable) {
        initCorrespondance();
    }
    //PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}

