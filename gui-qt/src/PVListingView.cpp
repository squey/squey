//! \file PVListingView.cpp
//! $Id: PVListingView.cpp 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>


#include <pvcore/general.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVListingView.h>

/******************************************************************************
 *
 * PVInspector::PVListingView::PVListingView
 *
 *****************************************************************************/
PVInspector::PVListingView::PVListingView(PVMainWindow *mw, Picviz::PVView_p pv_view, PVTabSplitter *parent) : QTableView(parent),main_window(mw)
{
	PVLOG_DEBUG("PVInspector::PVListingView::%s\n", __FUNCTION__);

	lib_view = pv_view;

// DDX: remove since it is now in PVTabSplitter! pv_layer_stack_model = new PVLayerStackModel(main_window, this); 

/*	pv_listing_model = new PVListingModel(main_window, this);
	pv_listing_no_unselected_model = new PVListingNoUnselectedModel(main_window, this);
	pv_listing_no_zombie_model = new PVListingNoZombieModel(main_window, this);
	pv_listing_no_zombie_no_unselected_model = new PVListingNoZombieNoUnselectedModel(main_window, this);

	setModel(pv_listing_model);

	screenshot_index = 0;*/

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::NoFocus);
	
	//ini the double click action
	connect(this->horizontalHeader(),SIGNAL(sectionDoubleClicked (int)),this,SLOT(slotDoubleClickOnVHead(int)));
}

/******************************************************************************
 *
 * PVInspector::PVListingView::mouseReleaseEvent
 *
 *****************************************************************************/
void PVInspector::PVListingView::mouseReleaseEvent(QMouseEvent *event)
{
	/* VARIABLES */
	Picviz::PVStateMachine *state_machine;
	int i;
	int number_of_items;
	int real_row_index;
	QModelIndexList selected_items_list;

	/* CODE */
	state_machine = lib_view->state_machine;

	/* We start by turning the square_area_mode OFF */

	/* We set square_area_mode */
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	/* We define the volatile_selection using selection in the lsiting */
	lib_view->volatile_selection.select_none();
	selected_items_list = selectedIndexes();
	number_of_items = selected_items_list.size();
	for (i=0; i<number_of_items; i++) {
		real_row_index = lib_view->get_real_row_index(selected_items_list[i].row());
		lib_view->volatile_selection.set_line(real_row_index, 1);
	}
	/* We reprocess the view from the selection */
	lib_view->process_from_selection();
	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
	/* We refresh the PVGLView */
	main_window->update_pvglview(lib_view, PVGL_COM_REFRESH_SELECTION);
	/* We refresh the listing */
	main_window->current_tab->refresh_listing_with_horizontal_header_Slot();
	main_window->current_tab->update_pv_listing_model_Slot();
	main_window->current_tab->refresh_layer_stack_view_Slot();

	/* we leave the ongoing job to the parent's method */
	QTableView::mouseReleaseEvent(event);
}

/******************************************************************************
 *
 * PVInspector::PVListingView::refresh_listing_Slot
 *
 *****************************************************************************/
/*void PVInspector::PVListingView::refresh_listing_Slot()
{
  viewport()->update();
	verticalHeader()->viewport()->update();
}*/

/******************************************************************************
 *
 * PVInspector::PVListingView::refresh_listing_with_horizontal_header_Slot
 *
 *****************************************************************************/
/*void PVInspector::PVListingView::refresh_listing_with_horizontal_header_Slot()
{
	horizontalHeader()->viewport()->update();
	viewport()->update();
}*/



/******************************************************************************
 *
 * PVInspector::PVListingView::selection_changed_Slot
 *
 *****************************************************************************/
/*void PVInspector::PVListingView::selection_changed_Slot()
{
	refresh_listing_Slot();
}*/

/******************************************************************************
 *
 * PVInspector::PVListingView::slotDoubleClickOnVHead
 *
 *****************************************************************************/
void PVInspector::PVListingView::slotDoubleClickOnVHead(int idHeader) 
{
	assert(model());
	static_cast<PVListingModel *>(model())->sortByColumn(idHeader);
}




