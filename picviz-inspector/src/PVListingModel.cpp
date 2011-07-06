//! \file PVListingModel.cpp
//! $Id: PVListingModel.cpp 3240 2011-07-05 05:11:55Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <pvcore/general.h>

#include <picviz/PVView.h>
#include <picviz/PVColor.h>
#include <picviz/state-machine.h>

#include <PVListingModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>



/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVInspector::PVListingModel::PVListingModel(PVMainWindow *mw, PVTabSplitter *parent) : PVListingModelBase(mw, parent)
{
	Picviz::PVView_p      lib_view;

	PVLOG_INFO("%s : Creating object\n", __FUNCTION__);

	not_zombie_font_brush = QBrush(QColor(0,0,0));
	zombie_font_brush = QBrush(QColor(200,200,200));
	
	lib_view = parent_widget->get_lib_view();
	//widgetCpyOfData = (const QVector<QStringList>&) lib_view->get_qtnraw_parent();
	
	colSorted = -1;

	initCorrespondance();
	
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::data(const QModelIndex &index, int role) const
{
	//PVLOG_DEBUG("PVInspector::PVListingModel::%s : at row %d and column %d with role %d\n", __FUNCTION__, index.row(), index.column(), role);

	Picviz::PVColor       color;
	int                   i;
	int 		      correspondId;
	Picviz::PVView_p      lib_view;
	Picviz::StateMachine *state_machine;

	unsigned char r;
	unsigned char g;
	unsigned char b;

	lib_view = parent_widget->get_lib_view();
	state_machine = lib_view->state_machine;
	
	//if(correspondTable.size()!=lib_view->get_qtnraw_parent().size()){
	//	initCorrespondance();
	//}
	
	switch (role) {
		case (Qt::DisplayRole):
			//return lib_view->get_data(index.row(), index.column());//old
			{
				
				correspondId = correspondTable.at(index.row());
				return lib_view->get_data(correspondId, index.column());
			}
			break;

		case (Qt::TextAlignmentRole):
			return (Qt::AlignLeft + Qt::AlignVCenter);
			break;

		case (Qt::BackgroundRole):
			/* We get the current selected axis index */
			i = lib_view->active_axis;
			//correspondId = correspondTable.at(index.row());
			/* We test  whether AXES_MODE is active or not */
			/* AND if our current colomn corresponds to the active axis */
			if ( (state_machine->is_axes_mode()) && ( i == index.column() ) ) {
				/* We must provide an evidence of the active_axis ! */
					return QBrush(QColor(130, 100, 25));
			} else {
				if (lib_view->get_line_state_in_output_layer(index.row())) {
					color = lib_view->get_color_in_output_layer(index.row());
					r = color.r();
					g = color.g();
					b = color.b();

					return QBrush(QColor(r,g,b));
				} else {
					return unselect_brush;
				}
			}
			break;

		case (Qt::ForegroundRole):
			/* We test if the line is a ZOMBIE one */
			//correspondId = correspondTable.at(index.row());
			if (lib_view->layer_stack_output_layer.get_selection().get_line(index.row())) {
				/* The line is NOT a ZOMBIE */
				return not_zombie_font_brush;
			} else {
				/* The line is a ZOMBIE */
				return zombie_font_brush;
			}
	}
	return QVariant();
}
/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
void PVInspector::PVListingModel::initCorrespondance(){
	Picviz::PVView_p   lib_view = parent_widget->get_lib_view();
        PVLOG_INFO("PVListingModel::initCorrespondance()\n");
	//init the table of corresponding table.
	correspondTable.resize(0);
	for(int i=0;i<lib_view->get_qtnraw_parent().size();i++){
		correspondTable.insert(i,i);
	}
	sortOrder = NoOrder;
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);

	Picviz::PVView_p lib_view = parent_widget->get_lib_view();

	switch (role) {
/*		case (Qt::BackgroundRole):
			if (picviz_view_get_line_state_in_output_layer(lib_view, index.row())) {
				color = picviz_view_get_color_in_output_layer(lib_view, index.row());
				r = picviz_color_get_r(color);
				g = picviz_color_get_g(color);
				b = picviz_color_get_b(color);

				return QBrush(QColor(r,g,b));
			} else {
				return unselect_brush;
			}
			break;
*/
		case (Qt::DisplayRole):
			if (orientation == Qt::Horizontal) {
				return QVariant(lib_view->get_axis_name(section));
			} else {
				return correspondTable.at(section)+1;
				//return section+1;//old
			}
			break;

		case (Qt::FontRole):
			if ((lib_view->real_output_selection.get_line(section)) && (orientation == Qt::Vertical)) {
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

	return QVariant();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::sortByColumn
 *
 *****************************************************************************/
void PVInspector::PVListingModel::sortByColumn(int idColumn){
	PVLOG_INFO("PVInspector::PVListingModel::sortByColumn(%d)\n",idColumn);
	if(parent_widget!=0){//if parent widget is valid...
		Picviz::PVView_p lib_view = parent_widget->get_lib_view();
		if(lib_view){//if lib_view is valid...
			//Picviz::PVSortQVectorQStringList sorter(0);//old
			Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0);
			PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting..."));
			connect(sortThread,SIGNAL(finished()),dialogBox,SLOT(accept()), Qt::QueuedConnection);

			//sorting
			PVRush::PVNraw::nraw_table *data=&lib_view->get_qtnraw_parent();
			//sorter.setList(data,&correspondTable);//old
			sortThread->setList(data,&correspondTable);
			if ((colSorted == idColumn) && (sortOrder == AscendingOrder)) {
				sortOrder=DescendingOrder;
				sortThread->init(idColumn,Qt::DescendingOrder);
				//sorter.sort(idColumn,Qt::DescendingOrder);
			} else {
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
			
			emit layoutChanged();
		}else{//if lib_view isn't valid...
			PVLOG_WARN("no lib_view : %s : %d",__FILE__,__LINE__);
		}
	}else{//if parent widget isn't valid...
		PVLOG_WARN("no parent widget : %s : %d",__FILE__,__LINE__);
	}
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::rowCount(const QModelIndex &/*index*/) const
{
	//PVLOG_DEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);
	Picviz::PVView_p lib_view = parent_widget->get_lib_view();

	return lib_view->get_row_count();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::getCorrespondance
 *
 *****************************************************************************/
int getCorrespondance(int line){
	// AG: robin, do you use this ?
	// Need to return something under MSVC, 0 for now

	return 0;
}
/******************************************************************************
 *
 * PVInspector::PVListingModel::reset_model
 *
 *****************************************************************************/
void PVInspector::PVListingModel::reset_model(bool initCorrespondTable){
	PVListingModelBase::reset_model();
	if(initCorrespondTable) {
            initCorrespondance();
        }
	//PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}

