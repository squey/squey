//! \file PVListingNoZombieNoUnselectedModel.cpp
//! $Id: PVListingNoZombieNoUnselectedModel.cpp 3248 2011-07-05 10:15:19Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <picviz/PVColor.h>
#include <pvcore/general.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVListingNoZombieNoUnselectedModel.h>

#include <picviz/PVSortQVectorQStringList.h>

using Picviz::PVColor;

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieNoUnselectedModel::PVListingNoZombieNoUnselectedModel
 *
 *****************************************************************************/
PVInspector::PVListingNoZombieNoUnselectedModel::PVListingNoZombieNoUnselectedModel(PVMainWindow *mw, PVTabSplitter *parent) :
	PVListingModelBase(mw, parent)
{
	PVLOG_INFO("%s : Creating object\n", __FUNCTION__);
	
	initCorrespondance();
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieNoUnselectedModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingNoZombieNoUnselectedModel::data(const QModelIndex &index, int role) const
{
	PVLOG_DEBUG("PVInspector::PVListingNoZombieNoUnselectedModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

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

	real_row_index = lib_view->get_nznu_real_row_index(index.row());

	switch (role) {
        case (Qt::DisplayRole):
        {
            PVLOG_DEBUG("       DisplayRole\n");
            int correspondId = correspondTable.at(index.row());
            return lib_view->get_data(correspondId, index.column());
            //return lib_view->get_data(real_row_index, index.column());//old
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
                color = lib_view->get_color_in_output_layer(correspondTable.at(index.row()));
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
 * PVInspector::PVListingNoZombieNoUnselectedModel::initCorrespondance
 *
 *****************************************************************************/
void PVInspector::PVListingNoZombieNoUnselectedModel::initCorrespondance() {
    Picviz::PVView_p lib_view = parent_widget->get_lib_view();
    //init the table of corresponding table.
    correspondTable.resize(0);
    for (int i = 0; i < rowCount(QModelIndex()); i++) {
        correspondTable.insert(i, lib_view->get_nznu_real_row_index(i));
    }
    sortOrder = NoOrder;
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieNoUnselectedModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingNoZombieNoUnselectedModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	PVLOG_DEBUG("PVInspector::PVListingNoZombieNoUnselectedModel::%s\n", __FUNCTION__);

	/* VARIABLES */
	Picviz::PVView_p lib_view;
	int real_row_index;

    /* CODE */
    lib_view = parent_widget->get_lib_view();
    /* We compute the real row index */
    real_row_index = lib_view->get_nznu_real_row_index(section);

    switch (role) {

        case (Qt::DisplayRole):
            if (orientation == Qt::Horizontal) {
                return QVariant(lib_view->get_axis_name(section));
            } else {
                return correspondTable.at(section) + 1;
            }
            break;

        case (Qt::FontRole):
            if (orientation == Qt::Vertical) {
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
 * PVInspector::PVListingNoZombieNoUnselectedModel::rowCount
 *
 * WARNING !
 * Be aware that nearly everything (scrolling but also mouse hover
 *  launched a rowCount!
 * Great!
 *
 *****************************************************************************/
int PVInspector::PVListingNoZombieNoUnselectedModel::rowCount(const QModelIndex &/*index*/) const
{
	Picviz::PVView_p lib_view;

	//PVLOG_DEBUG("PVInspector::PVListingNoZombieNoUnselectedModel::%s\n", __FUNCTION__);

	lib_view = parent_widget->get_lib_view();
	return lib_view->get_nznu_index_count();
}

/******************************************************************************
 *
 * PVInspector::PVListingNoZombieNoUnselectedModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingNoZombieNoUnselectedModel::sortByColumn(int idColumn){
	PVLOG_INFO("PVInspector::PVListingNoZombieNoUnselectedModel::sortByColumn(%d)\n",idColumn);
	
	Picviz::PVView_p lib_view = parent_widget->get_lib_view();
    	
	//init correspondence table
	QVector<int> correspondTableNew;
	for(int i=0;i<rowCount(QModelIndex());i++){
		int real_row_index = lib_view->get_nznu_real_row_index(i);
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
 * PVInspector::PVListingNoZombieNoUnselectedModel::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingNoZombieNoUnselectedModel::reset_model(bool initCorrespondTable){
	PVListingModelBase::reset_model();
	if(initCorrespondTable) {
            initCorrespondance();
        }
	//PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}
