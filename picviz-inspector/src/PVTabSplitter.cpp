//! \file PVTabSplitter.cpp
//! $Id: PVTabSplitter.cpp 3248 2011-07-05 10:15:19Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include <pvcore/general.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVListingView.h>

#include <PVTabSplitter.h>

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVTabSplitter
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVTabSplitter(PVMainWindow *mw, Picviz::PVView_p pv_view, QString const& tab_name, QWidget *parent) :
	QSplitter(parent)
{
	PVLOG_DEBUG("PVInspector::PVTabSplitter::%s\n", __FUNCTION__);

	main_window = mw;
	lib_view = pv_view;
	pv_layer_stack_widget = NULL; // Note that this value can be requested during the creating of the PVLayerStackWidget!
	_tab_name = tab_name;

	pv_listing_model = new PVListingModel(main_window, this);
	pv_listing_no_unselected_model = new PVListingNoUnselectedModel(main_window, this);
	pv_listing_no_zombie_model = new PVListingNoZombieModel(main_window, this);
	pv_listing_no_zombie_no_unselected_model = new PVListingNoZombieNoUnselectedModel(main_window, this);
	pv_listing_view = new PVListingView(main_window, lib_view, this);
	pv_listing_view->setModel(pv_listing_model);

	addWidget(pv_listing_view);
	pv_layer_stack_model = new PVLayerStackModel(main_window, this);
	pv_layer_stack_widget = new PVLayerStackWidget(main_window, pv_layer_stack_model, this);
	addWidget(pv_layer_stack_widget);

	_pv_extractor = new PVExtractorWidget(this);

	screenshot_index = 0;

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::NoFocus);
//	moveSplitter(250, 1);
}

PVInspector::PVTabSplitter::~PVTabSplitter()
{
	PVLOG_INFO("In PVTabSplitter destructor\n");
	_pv_extractor->deleteLater();
//	pv_listing_view->deleteLater();
//	pv_listing_model->deleteLater();
//	pv_listing_no_unselected_model->deleteLater();
//	pv_listing_no_zombie_model->deleteLater();
//	pv_listing_no_zombie_no_unselected_model->deleteLater();
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::get_screenshot_index
 *
 *****************************************************************************/
int PVInspector::PVTabSplitter::get_screenshot_index()
{
	return screenshot_index;
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::increment_screenshot_index
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::increment_screenshot_index()
{
	screenshot_index++;
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::refresh_listing_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::refresh_listing_Slot()
{	PVLOG_DEBUG("%s \n       %s %d\n",__FILE__,__FUNCTION__,__LINE__);
	if (pv_listing_view) {
		pv_listing_view->viewport()->update();
		pv_listing_view->verticalHeader()->viewport()->update();
		//static_cast<PVListingModelBase*>(pv_listing_view->model())->reset_model();
		//update the size of the corresponding table.
		static_cast<PVListingModelBase*>(pv_listing_view->model())->initCorrespondance();
		static_cast<PVListingModelBase*>(pv_listing_view->model())->emitLayoutChanged();
	}
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::refresh_listing_with_horizontal_header_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::refresh_listing_with_horizontal_header_Slot()
{	//PVLOG_INFO("%s \n       %s %d\n",__FILE__,__FUNCTION__,__LINE__);
	if (pv_listing_view) {
		pv_listing_view->horizontalHeader()->viewport()->update();
		pv_listing_view->viewport()->update();
	}
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::selection_changed_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::selection_changed_Slot()
{	
	//PVLOG_INFO("%s \n       %s %d\n",__FILE__,__FUNCTION__,__LINE__);
	refresh_listing_Slot();
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::update_pv_listing_model_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::update_pv_listing_model_Slot()
{
	
	PVLOG_DEBUG("%s \n       %s \n",__FILE__,__FUNCTION__);
	
	updateFilterMenuEnabling();

	if (!pv_listing_view)
		return;
	/* We get an access to the current StateMachine */
	Picviz::StateMachine *state_machine = lib_view->state_machine;
	/* We prepare a pointer of type (QAbstractTableModel *) */
	QAbstractTableModel *next_model;

	/* We set the model according to the two listing modes */
	/* First of all, we check if the Unselected are visible */
	if (state_machine->are_listing_unselected_visible()) {
		/* The Unselected are visible */
		/* We then check if the Zombie are visible */
		if (state_machine->are_listing_zombie_visible()) {
			/* The Zombie are visible too */
			pv_listing_model->reset_model(false); // This is needed, but I don't know why. Maybe only needed once [DDX] XXX ???
			next_model = pv_listing_model;
		} else {
			/* The Zombie are NOT visible */
			pv_listing_no_zombie_model->reset_model(false); // Ditto XXX ???
			next_model = pv_listing_no_zombie_model;
		}
	} else {
		/* The UNSELECTED are NOT visible! */
		/* We then check if the Zombie are visible */
		if (state_machine->are_listing_zombie_visible()) {
			/* The Zombie are visible */
			pv_listing_no_unselected_model->reset_model(false); // Ditto XXX ???
			next_model = pv_listing_no_unselected_model;
		} else {
			/* The Zombie are NOT visible */
			pv_listing_no_zombie_no_unselected_model->reset_model(false); // Ditto XXX ???
			next_model = pv_listing_no_zombie_no_unselected_model;
		}
	}

	/* Now we can set the model ! */
	pv_listing_view->setModel(next_model);
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::refresh_layer_stack_view_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::refresh_layer_stack_view_Slot()
{
	/* this doesn't work !!! */
	//pv_layer_stack_widget->pv_layer_stack_view->viewport()->update();

	pv_layer_stack_model->emit_layoutChanged();
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::updateFilterMenuEnabling
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::updateFilterMenuEnabling(){
	int countSelLine = get_lib_view()->get_nu_index_count();
	if(countSelLine>0){
		emit selection_changed_signal(true);
	}else{
		emit selection_changed_signal(false);
	}
}

