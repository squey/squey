//! \file PVListingNoUnselectedModel.cpp
//! $Id: PVListingNoUnselectedModel.cpp 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>


#include <picviz/PVColor.h>
#include <pvcore/general.h>
#include <picviz/PVView.h>

#include <PVListingNoUnselectedModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>

using Picviz::PVColor;


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::PVListingNoUnselectedModel
 *
 *****************************************************************************/
PVInspector::PVListingNoUnselectedModel::PVListingNoUnselectedModel(PVMainWindow *mw, PVTabSplitter *parent) : PVListingModelBase(mw, parent) {
    PVLOG_INFO("%s : Creating object\n", __FUNCTION__);

    not_zombie_font_brush = QBrush(QColor(0, 0, 0));
    zombie_font_brush = QBrush(QColor(200, 200, 200));

    initCorrespondance();
}


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingNoUnselectedModel::data(const QModelIndex &index, int role) const {
    PVLOG_DEBUG("PVInspector::PVListingNoUnselectedModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

    /* VARIABLES */
    PVColor color;
    int i;
    Picviz::PVView_p lib_view;
    int real_row_index;
    Picviz::StateMachine *state_machine;

    unsigned char r;
    unsigned char g;
    unsigned char b;

    /* CODE */
    lib_view = parent_widget->get_lib_view();
    state_machine = lib_view->state_machine;

    PVLOG_DEBUG("       index.row()%d\n", index.row());
    int tmp_count = lib_view->get_nu_index_count();
    PVLOG_DEBUG("       real count %d, correspondTable.size %d\n", tmp_count, matchingTable.size());
    real_row_index = lib_view->get_nu_real_row_index(matchingTable.at(index.row()));
    PVLOG_DEBUG("       real_row_index %d\n", real_row_index);

    switch (role) {
        case (Qt::BackgroundRole):
            PVLOG_DEBUG("       BackgroundRole\n");
            /* We get the current selected axis index */
            i = lib_view->active_axis;
            /* We test  whether AXES_MODE is active or not */
            /* AND if our current colomn corresponds to the active axis */
            if ((state_machine->is_axes_mode()) && (i == index.column())) {
                /* We must provide an evidence of the active_axis ! */
                return QBrush(QColor(130, 100, 25));
            } else {
                PVLOG_DEBUG("           real color\n");
                color = lib_view->get_color_in_output_layer(real_row_index);
                r = color.r();
                g = color.g();
                b = color.b();

                return QBrush(QColor(r, g, b));
            }
            break;

        case (Qt::DisplayRole):
        {
            PVLOG_DEBUG("       DisplayRole\n");
            //int correspondId = correspondTable.at(real_row_index);
            int correspondId = matchingTable.at(real_row_index);
            PVLOG_DEBUG("           correspondId %d\n", correspondId);
            //return lib_view->get_data(correspondId, index.column());
            return lib_view->get_data(real_row_index, index.column()); //old
        }
            break;

        case (Qt::ForegroundRole):
            PVLOG_DEBUG("       ForegroundRole\n");
            /* We test if the line is a ZOMBIE one */
            if (lib_view->layer_stack_output_layer.get_selection().get_line(real_row_index)) {
                /* The line is NOT a ZOMBIE */
                return not_zombie_font_brush;
            } else {
                /* The line is a ZOMBIE */
                return zombie_font_brush;
            }

        case (Qt::TextAlignmentRole):
            return (Qt::AlignLeft + Qt::AlignVCenter);
            break;

        default:
            return QVariant();
    }

}


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::initCorrespondance
 *
 *****************************************************************************/
void PVInspector::PVListingNoUnselectedModel::initCorrespondance() {
    Picviz::PVView_p lib_view = parent_widget->get_lib_view();
    //init the table of corresponding table.
    matchingTable.resize(0);
    for (int i = 0; i < rowCount(QModelIndex()); i++) {
        matchingTable.insert(i, lib_view->get_nu_real_row_index(i));
    }
    sortOrder = NoOrder;
}


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingNoUnselectedModel::headerData(int section, Qt::Orientation orientation, int role) const {
    PVLOG_DEBUG("PVInspector::PVListingNoUnselectedModel::%s\n", __FUNCTION__);

    /* VARIABLES */
    Picviz::PVView_p lib_view;
    int real_row_index;

    /* CODE */
    lib_view = parent_widget->get_lib_view();

    /* We compute the real row index */
    real_row_index = lib_view->get_nu_real_row_index(section);

    switch (role) {

        case (Qt::DisplayRole):
            if (orientation == Qt::Horizontal) {
                return QVariant(lib_view->get_axis_name(section));
            } else {
                //return correspondTable.at(real_row_index) + 1;
                return matchingTable.at(section) + 1;
            }
            break;

        case (Qt::FontRole):
            if ((lib_view->real_output_selection.get_line(real_row_index)) && (orientation == Qt::Vertical)) {
                return select_font;
            } else {
                return unselect_font;
            }
            break;

        case (Qt::TextAlignmentRole):
            if (orientation == Qt::Horizontal) {
                return (Qt::AlignLeft + Qt::AlignVCenter);
            } else {
                return (Qt::AlignRight + Qt::AlignVCenter);
            }
            break;

        default:
            return QVariant();
    }
}


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::rowCount
 *
 * WARNING !
 * Be aware that nearly everything (scrolling but also mouse hover
 *  launched a rowCount!
 * Great!
 *
 *****************************************************************************/
int PVInspector::PVListingNoUnselectedModel::rowCount(const QModelIndex &/*index*/) const {
    Picviz::PVView_p lib_view;

    //PVLOG_DEBUG("PVInspector::PVListingNoUnselectedModel::%s\n", __FUNCTION__);

    lib_view = parent_widget->get_lib_view();
    return int(lib_view->get_nu_index_count());
}


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingNoUnselectedModel::sortByColumn(int idColumn) {
    PVLOG_INFO("PVInspector::PVListingNoUnselectedModel::sortByColumn(%d) size%d\n", idColumn, rowCount(QModelIndex()));

    Picviz::PVView_p lib_view = parent_widget->get_lib_view();
    //initCorrespondance();

    //init correspondence table
    QVector<int> matchTableNew;
    for (int i = 0; i < rowCount(QModelIndex()); i++) {
        int real_row_index = lib_view->get_nu_real_row_index(i);
        matchTableNew.insert(i, real_row_index);
        //PVLOG_INFO("i%d real_row_index%d\n",i,correspondTableNew.at(i));
    }


    if (parent_widget != 0) {//if parent widget is valid...

        if (lib_view) {//if lib_view is valid...
            //Picviz::PVSortQVectorQStringList sorter;
            Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0);
            PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting..."));
            connect(sortThread, SIGNAL(finished()), dialogBox, SLOT(accept()), Qt::QueuedConnection);

            //sorting
            //init sort
            PVRush::PVNraw::nraw_table &data = lib_view->get_qtnraw_parent(); //new PVRush::PVNraw::nraw_table();
            //sorter.setList(&data,&correspondTableNew);
            sortThread->setList(&data, &matchTableNew);
            if (colSorted == idColumn && sortOrder == AscendingOrder) {
                sortOrder = DescendingOrder;
                sortThread->init(idColumn, Qt::DescendingOrder);
            } else {
                colSorted = idColumn;
                sortOrder = AscendingOrder;
                sortThread->init(idColumn, Qt::AscendingOrder);
            }
            sortThread->start(QThread::LowPriority);
            PVLOG_INFO("waitting : sort processing... \n");
            if (dialogBox->exec()) {//show dialog and wait for event
                sortThread->update();
            } else {//if we cancel during the sort...
                //... no update.
                //... stop the the thread.
                sortThread->exit(0);
            }

            //PVLOG_INFO("%s \n       %s %d\n",__FILE__,__FUNCTION__,__LINE__);
            //update matching
            matchingTable.resize(0);
            for (int i = 0; i < rowCount(QModelIndex()); i++) {
                int real_row_index = lib_view->get_nu_real_row_index(i);
                matchingTable.insert(i, matchTableNew.at(i));
            }
            emit layoutChanged();
        } else {//if lib_view isn't valid...
            PVLOG_INFO("no lib_view : %s : %d", __FILE__, __LINE__);
        }
    } else {//if parent widget isn't valid...
        PVLOG_INFO("no parent widget : %s : %d", __FILE__, __LINE__);
    }
}


/******************************************************************************
 *
 * PVInspector::PVListingNoUnselectedModel::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingNoUnselectedModel::reset_model(bool initMatchTable) {
    PVListingModelBase::reset_model();
    if (initMatchTable) {
        initCorrespondance();
    }
    //PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}


