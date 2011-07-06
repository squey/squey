//! \file PVListingNoZombieModel.cpp
//! $Id: PVListingNoZombieModel.cpp 3248 2011-07-05 10:15:19Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>
#include <QVector>

#include <picviz/PVColor.h>
#include <pvcore/general.h>
#include <picviz/PVView.h>

#include <PVListingNoZombieModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>

using Picviz::PVColor;

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieModel::PVListingNoZombieModel
 *
 *****************************************************************************/
PVInspector::PVListingNoZombieModel::PVListingNoZombieModel(PVMainWindow *mw, PVTabSplitter *parent) : PVListingModelBase(mw, parent)
{
	PVLOG_INFO("%s : Creating object\n", __FUNCTION__);
	
	initCorrespondance();
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingNoZombieModel::data(const QModelIndex &index, int role) const
{
	PVLOG_DEBUG("PVInspector::PVListingNoZombieModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

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

    PVLOG_DEBUG("       index.row()%d, match %d\n", index.row(), correspondTable.at(index.row()));
    //    if (lib_view->get_nz_index_count() != correspondTable.size()) {
    int tmp_count = lib_view->get_nz_index_count();
    PVLOG_DEBUG("       real count %d, correspondTable.size %d\n", tmp_count, correspondTable.size());
    //        initCorrespondance();
    //    }
    
    real_row_index = lib_view->get_nz_real_row_index(correspondTable.at(index.row()));
    PVLOG_DEBUG("       real_row_index %d\n", real_row_index);

    switch (role) {
        case (Qt::DisplayRole):
        {
            PVLOG_DEBUG("       DisplayRole\n");
            int correspondId = correspondTable.at(index.row());
            //return lib_view->get_data(correspondId, index.column());
            return lib_view->get_data(correspondTable.at(index.row()), index.column()); //old
        }
            break;

        case (Qt::TextAlignmentRole):
            return (Qt::AlignLeft + Qt::AlignVCenter);
            break;

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
                color = lib_view->get_color_in_output_layer(real_row_index);
                r = color.r();
                g = color.g();
                b = color.b();

                return QBrush(QColor(r, g, b));
            }
            break;

		default:
			return QVariant();
	}
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieModel::initCorrespondance
 *
 *****************************************************************************/
void PVInspector::PVListingNoZombieModel::initCorrespondance(){
	Picviz::PVView_p   lib_view = parent_widget->get_lib_view();
	//init the table of corresponding table.
	correspondTable.resize(0);
	for(int i=0;i<rowCount(QModelIndex());i++){
		correspondTable.insert(i,lib_view->get_nz_real_row_index(i));
		//correspondTable.insert(i,i);//test
	}
	sortOrder = NoOrder;
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingNoZombieModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	PVLOG_DEBUG("PVInspector::PVListingNoZombieModel::%s\n", __FUNCTION__);

	/* VARIABLES */
	Picviz::PVView_p lib_view;
	int real_row_index;

	/* CODE */
	lib_view = parent_widget->get_lib_view();

	/* We compute the real row index */
	real_row_index = lib_view->get_nz_real_row_index(section);

	switch (role) {
		case (Qt::DisplayRole):
			if (orientation == Qt::Horizontal) {
				return QVariant(lib_view->get_axis_name(section));
			} else {
				return correspondTable.at(section) + 1;
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
 * PVInspector::PVListingNoZombieModel::rowCount
 *
 * WARNING !
 * Be aware that nearly everything (scrolling but also mouse hover
 *  launched a rowCount!
 * Great!
 *
 *****************************************************************************/
int PVInspector::PVListingNoZombieModel::rowCount(const QModelIndex &/*index*/) const
{
	Picviz::PVView_p lib_view;

	//PVLOG_DEBUG("PVInspector::PVListingNoZombieModel::%s\n", __FUNCTION__);

	lib_view = parent_widget->get_lib_view();
	return int(lib_view->get_nz_index_count());
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingNoZombieModel::sortByColumn(int idColumn){
	PVLOG_INFO("PVInspector::PVListingNoZombieModel::sortByColumn(%d)\n",idColumn);
	
	Picviz::PVView_p lib_view = parent_widget->get_lib_view();
	//initCorrespondance();
	
	//init correspondence table
	QVector<int> correspondTableNew;
	for(int i=0;i<rowCount(QModelIndex());i++){
		int real_row_index = lib_view->get_nz_real_row_index(i);
		correspondTableNew.insert(i,real_row_index);
	}
	

	if(parent_widget!=0){//if parent widget is valid...
		
		if(lib_view){//if lib_view is valid...
			//Picviz::PVSortQVectorQStringList sorter;
			Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0);
			PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting..."));
			connect(sortThread,SIGNAL(finished()),dialogBox,SLOT(accept()), Qt::QueuedConnection);
			
			//sorting
			PVRush::PVNraw::nraw_table *data=&lib_view->get_qtnraw_parent();///FIXME refaire data...
			//sorter.setList(data,&correspondTableNew);
			sortThread->setList(data,&correspondTableNew);
			if(colSorted==idColumn && sortOrder==AscendingOrder){
				sortOrder=DescendingOrder;
				sortThread->init(idColumn,Qt::DescendingOrder);
				//sorter.sort(idColumn,Qt::DescendingOrder);
			}else{
				colSorted = idColumn;
				sortOrder=AscendingOrder;
				sortThread->init(idColumn,Qt::AscendingOrder);
				//sorter.sort(idColumn,Qt::AscendingOrder);
			}
			sortThread->start(QThread::LowPriority);
			PVLOG_INFO("waitting : sort processing... \n");
			if(dialogBox->exec()){//show dialog and wait for event
				sortThread->update();
			}else{//if we cancel during the sort...
				//... no update.
				//... stop the the thread.
				sortThread->exit(0);
			}
			
			//update correspondence
			correspondTable.resize(0);
			for(int i=0;i<rowCount(QModelIndex());i++){
				correspondTable.insert(i,correspondTableNew.at(i));
			}
			emit layoutChanged();
		}else{//if lib_view isn't valid...
			PVLOG_INFO("no lib_view : %s : %d",__FILE__,__LINE__);
		}
	}else{//if parent widget isn't valid...
		PVLOG_INFO("no parent widget : %s : %d",__FILE__,__LINE__);
	}
}


/******************************************************************************
 *
 * PVInspector::PVListingNoZombieModel::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingNoZombieModel::reset_model(bool initCorrespondTable) {
    PVListingModelBase::reset_model();
    if (initCorrespondTable) {
        initCorrespondance();
    }
    //PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}
