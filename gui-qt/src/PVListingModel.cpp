//! \file PVListingModel.cpp
//! $Id: PVListingModel.cpp 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <pvkernel/core/general.h>

#include <picviz/PVView.h>
#include <pvkernel/core/PVColor.h>
#include <picviz/PVStateMachine.h>

#include <PVListingModel.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <picviz/PVSortQVectorQStringList.h>

#include <map>


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
	assert(lib_view);
	state_machine = lib_view->state_machine;

	initMatchingTable();
	initLocalMatchingTable();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::columnCount(const QModelIndex &) const 
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

	PVCore::PVColor color;
	int i;
	int real_row_index;

	unsigned char r;
	unsigned char g;
	unsigned char b;

	real_row_index = getRealRowIndex(index.row());

	PVLOG_HEAVYDEBUG("           correspondId %d\n", real_row_index);

	switch (role) {
		case (Qt::DisplayRole):
			return lib_view->get_data(real_row_index, index.column());

		case (Qt::TextAlignmentRole):
			return (Qt::AlignLeft + Qt::AlignVCenter);

		case (Qt::BackgroundRole):
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

		case (Qt::ForegroundRole):
			if (lib_view->layer_stack_output_layer.get_selection().get_line(real_row_index)) {
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
 * PVInspector::PVListingModel::initLocalMatchingTable
 *
 *****************************************************************************/
void PVInspector::PVListingModel::initLocalMatchingTable(){
	PVLOG_DEBUG("PVListingModel::initLocalMatchingTable()\n");

	// Put a mutex for the local mathing table, because a thread can create it while being read by another one (like the model's thread) !
	QWriteLocker locker(&_local_table_mutex);

	std::map<int,int> myMap;
	std::map<int,int>::iterator myIterator;


	if (state_machine->are_listing_all()) {
		PVLOG_DEBUG("       ALL\n");
	}
	else
	if (state_machine->are_listing_no_nu_nz()) {
		PVLOG_DEBUG("       NZNU\n");
		localMatchingTable.resize(lib_view->get_nznu_index_count());
		//init map
		for(int i=0;i<lib_view->get_nznu_index_count();i++){
			int real=lib_view->get_nznu_real_row_index(i);
			myMap.insert(std::pair<int,int>(parent_widget->sortMatchingTable_invert.at(real),real));
		}
		//write in local matching table
		myIterator = myMap.begin();
		for(int i=0;i<lib_view->get_nznu_index_count();i++){
			localMatchingTable[i] = (*myIterator++).second;
			PVLOG_HEAVYDEBUG("       locale%d\n",localMatchingTable[i]);
		}
	}
	else
	if (state_machine->are_listing_no_nu()) {
		PVLOG_DEBUG("       NU\n");
		localMatchingTable.resize(lib_view->get_nu_index_count());
		//init map
		for(int i=0;i<lib_view->get_nu_index_count();i++){
			int real=lib_view->get_nu_real_row_index(i);
			myMap.insert(std::pair<int,int>(parent_widget->sortMatchingTable_invert.at(real),real));
		}
		//write in local matching table
		myIterator = myMap.begin();
		for(int i=0;i<lib_view->get_nu_index_count();i++){
			localMatchingTable[i] = (*myIterator++).second;
		}
	}
	else 
	if (state_machine->are_listing_no_nz()) {
		PVLOG_DEBUG("       NZ\n");
		localMatchingTable.resize(lib_view->get_nz_index_count());
		//init map
		for(int i=0;i<lib_view->get_nz_index_count();i++){
			int real=lib_view->get_nz_real_row_index(i);
			myMap.insert(std::pair<int,int>(parent_widget->sortMatchingTable_invert.at(real),real));
		}
		//write in local matching table
		myIterator = myMap.begin();
		for(int i=0;i<lib_view->get_nz_index_count();i++){
			localMatchingTable[i] = (*myIterator++).second;
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::initMatchingTable
 *
 *****************************************************************************/
void PVInspector::PVListingModel::initMatchingTable() {
	PVLOG_DEBUG("PVListingModel::initCorrespondance()\n");
	//init the table of corresponding table.
	//if the size of nraw is not the same as the matching table...
	if (lib_view->get_qtnraw_parent().size() != parent_widget->sortMatchingTable.size()) {
		PVLOG_DEBUG("         init LISTING_ALL\n");
		//...reinit the matching table.
		parent_widget->sortMatchingTable.resize(0);
		for (unsigned int i = 0; i < lib_view->get_qtnraw_parent().size(); i++) {
			parent_widget->sortMatchingTable.push_back(i);
		}
		parent_widget->sortMatchingTable_invert.resize(parent_widget->sortMatchingTable.size());
		for (unsigned int i = 0; i < parent_widget->sortMatchingTable.size(); i++) {
			int j = parent_widget->sortMatchingTable.at(i);
			parent_widget->sortMatchingTable_invert.at(j) = i;
		}

		sortOrder = NoOrder; //... reset the last order remember
		initLocalMatchingTable();
		emitLayoutChanged(); //... notify to the view that data has changed
	}

}


/******************************************************************************
 *
 * PVInspector::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVInspector::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const {
	PVLOG_HEAVYDEBUG("PVInspector::PVListingModel::%s\n", __FUNCTION__);
	if (!(section >= 0 && section < (int)lib_view->get_qtnraw_parent().size())) {
		return QVariant();
	}

	switch (role) {
		case (Qt::DisplayRole):
			if (orientation == Qt::Horizontal) {
				return QVariant(lib_view->get_axis_name(section));
			} else {
				return getRealRowIndex(section);
			}
			break;
		case (Qt::FontRole):
			if ((orientation == Qt::Vertical) && (lib_view->real_output_selection.get_line(getRealRowIndex(section)))) {
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
void PVInspector::PVListingModel::sortByColumn(int idColumn) 
{       
	if ((idColumn < 0) || (idColumn >= columnCount(QModelIndex()))) {
		PVLOG_DEBUG("     can't sort the column %d\n",idColumn);
		return;
	}


	if (state_machine->are_listing_all()) {//all
		PVLOG_DEBUG("PVInspector::PVListingModel::sortByColumn   all(%d)\n",idColumn);
	}
	else
	if (state_machine->are_listing_no_nu_nz()) {//NU NZ
		PVLOG_DEBUG("PVInspector::PVListingModel::sortByColumn   NU NZ(%d)\n",idColumn);
	}
	else
	if (!state_machine->are_listing_no_nu()) {//NU
		PVLOG_DEBUG("PVInspector::PVListingModel::sortByColumn   NU(%d)\n",idColumn);
	}
	else
	if (!state_machine->are_listing_no_nz()) {//NZ
		PVLOG_DEBUG("PVInspector::PVListingModel::sortByColumn   NZ(%d)\n",idColumn);
	}
	else {
		PVLOG_ERROR("PVInspector::PVListingModel::sortByColumn   :  listing mode unknow\n");
	}

	//variables init
	Picviz::PVSortQVectorQStringListThread *sortThread = new Picviz::PVSortQVectorQStringListThread(0); //class whiche can sort.
	PVProgressBox *dialogBox = new PVProgressBox(tr("Sorting...")); //dialog showing the progress box.
	connect(sortThread, SIGNAL(finished()), dialogBox, SLOT(accept()), Qt::QueuedConnection); //connection to close the progress box after thread finish.
	PVLOG_DEBUG("   declaration ok\n");

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
	PVLOG_DEBUG("    waitting : sort processing... \n");

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
	for (unsigned int i = 0; i < parent_widget->sortMatchingTable.size(); i++) {
		int j = parent_widget->sortMatchingTable.at(i);
		PVLOG_HEAVYDEBUG("   %d\n",j);
		parent_widget->sortMatchingTable_invert.at(j) = i;
	}

	initLocalMatchingTable();

	emit layoutChanged();
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVInspector::PVListingModel::rowCount(const QModelIndex &/*index*/) const 
{
	if (state_machine->are_listing_all()) {
		return int(lib_view->get_row_count());
	}
	else
	if (state_machine->are_listing_no_nu_nz()) {
		return int(lib_view->get_nznu_index_count());
	}
	else
	if (state_machine->are_listing_no_nu()) {
		return int(lib_view->get_nu_index_count());
	}
	else
	if (state_machine->are_listing_no_nz()) {
		return int(lib_view->get_nz_index_count());
	}

	PVLOG_ERROR("Unknown listing visibility state!\n");
	return 0;
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::getRealRowIndex
 *
 *****************************************************************************/
PVRow PVInspector::PVListingModel::getRealRowIndex(PVRow model_row) const
{
	// Be sure that no thread are actually creating this table
	QReadLocker locker(&_local_table_mutex);

	PVCol real_row_index = 0;
	if (state_machine->are_listing_all()) {
		real_row_index = parent_widget->sortMatchingTable.at(model_row);
	}
	else
	if (state_machine->are_listing_no_nu_nz()) {//NU NZ
		real_row_index = localMatchingTable[model_row];//(lib_view->get_nznu_real_row_index(index.row()));//
	}
	else
	if (state_machine->are_listing_no_nu()) {//NU
		real_row_index = localMatchingTable[model_row];//(lib_view->get_nu_real_row_index(index.row()));
	}
	else
	if (state_machine->are_listing_no_nz()) {//NZ
		real_row_index = localMatchingTable[model_row];//(lib_view->get_nz_real_row_index(index.row()));
	}
	else {
		PVLOG_ERROR("Unknown listing visibility state!\n");
	}

	return real_row_index;
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::getInvertedMatch
 *
 *****************************************************************************/
unsigned int PVInspector::PVListingModel::getInvertedMatch(unsigned int line){
	return int(parent_widget->sortMatchingTable_invert.at(line));
}

/******************************************************************************
 *
 * PVInspector::PVListingModel::getLocalMatch
 *
 *****************************************************************************/
unsigned int PVInspector::PVListingModel::getLocalMatch(unsigned int line){
	return int(localMatchingTable.at(line));
}


/******************************************************************************
 *
 * PVInspector::PVListingModel::getMatch
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
	initLocalMatchingTable();
	emitLayoutChanged();
	//PVLOG_INFO("reset_model() : rowCount=%d, corresp.size=%d\n",rowCount(QModelIndex()),correspondTable.size());
}

