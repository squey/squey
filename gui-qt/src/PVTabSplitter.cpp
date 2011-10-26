//! \file PVTabSplitter.cpp
//! $Id: PVTabSplitter.cpp 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) SÃ©bastien Tricaud 2009-2011
//! Copyright (C) Philippe SaadÃ© 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVListingView.h>
#include <PVExtractorWidget.h>
#include <PVAxesCombinationDialog.h>
#include <PVMappingPlottingEditDialog.h>
#include <PVViewsListingWidget.h>

#include <PVTabSplitter.h>

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVTabSplitter
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVTabSplitter(PVMainWindow *mw, Picviz::PVSource_p lib_src, QWidget *parent) :
	QSplitter(parent),
	_lib_src(lib_src)
{
	PVLOG_DEBUG("PVInspector::PVTabSplitter::%s\n", __FUNCTION__);
	assert(lib_src->get_views().size() > 0);
	// Select the first view

	main_window = mw;
	pv_layer_stack_widget = NULL; // Note that this value can be requested during the creating of the PVLayerStackWidget!

	pv_listing_model = new PVListingModel(main_window, this);
	pv_listing_view = new PVListingView(main_window, this);
	pv_listing_view->setModel(pv_listing_model);

	addWidget(pv_listing_view);

	QVBoxLayout* right_layout = new QVBoxLayout();
	pv_layer_stack_model = new PVLayerStackModel(main_window, this);
	pv_layer_stack_widget = new PVLayerStackWidget(main_window, pv_layer_stack_model, this);
	right_layout->addWidget(pv_layer_stack_widget);
	PVViewsListingWidget* views_widget = new PVViewsListingWidget(this);
	right_layout->addWidget(views_widget);

	QWidget* right_widget = new QWidget();
	right_widget->setLayout(right_layout);
	addWidget(right_widget);

	_pv_extractor = new PVExtractorWidget(this);
	pv_axes_combination_editor = new PVAxesCombinationDialog(this, mw);

	screenshot_index = 0;

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::NoFocus);
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
{
	PVLOG_DEBUG("%s \n       %s %d\n",__FILE__,__FUNCTION__,__LINE__);
	Picviz::PVView_p current_lib_view = get_lib_view();
	if (pv_listing_view) {
		current_lib_view->gl_call_locker.lock();
		pv_listing_view->viewport()->update();
		pv_listing_view->verticalHeader()->viewport()->update();
		//static_cast<PVListingModelBase*>(pv_listing_view->model())->reset_model();
		//update the size of the corresponding table.
		static_cast<PVListingModel*>(pv_listing_view->model())->initMatchingTable();
                static_cast<PVListingModel*>(pv_listing_view->model())->initLocalMatchingTable();
		static_cast<PVListingModel*>(pv_listing_view->model())->emitLayoutChanged();
		current_lib_view->gl_call_locker.unlock();
		main_window->update_statemachine_label(current_lib_view);
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
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::refresh_layer_stack_view_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::refresh_layer_stack_view_Slot()
{
	PVLOG_DEBUG("PVInspector::PVTabSplitter::refresh_layer_stack_view_Slot()\n");
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

void PVInspector::PVTabSplitter::refresh_axes_combination_Slot()
{
	if (pv_axes_combination_editor->isVisible()) {
		pv_axes_combination_editor->update_used_axes();
	}
}

void PVInspector::PVTabSplitter::select_view(Picviz::PVView_p view)
{
	assert(view->get_source_parent() == _lib_src.get());
	_lib_src->select_view(view);

	sortMatchingTable.clear();
	sortMatchingTable_invert.clear();

	// Update the layer stack
	pv_layer_stack_model->update_layer_stack();

	// And the listing
	pv_listing_model->reset_lib_view();
	pv_listing_view->update_view();
}

void PVInspector::PVTabSplitter::create_new_mapped()
{
	Picviz::PVMapping new_mapping(get_lib_src().get());
	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(&new_mapping, NULL, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	Picviz::PVMapped_p new_mapped(new Picviz::PVMapped(new_mapping));
	get_lib_src()->add_mapped(new_mapped);
	_views_widget->force_refresh();
}

void PVInspector::PVTabSplitter::select_plotted(Picviz::PVPlotted* plotted)
{
	if (!plotted->is_uptodate()) {
		plotted->process_from_parent_mapped(true);
	}
	select_view(plotted->get_view());
}

void PVInspector::PVTabSplitter::create_new_plotted(Picviz::PVMapped* mapped_parent)
{
	Picviz::PVPlotting new_plotting(mapped_parent);
	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(NULL, &new_plotting, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	Picviz::PVPlotted_p new_plotted(new Picviz::PVPlotted(new_plotting));
	mapped_parent->add_plotted(new_plotted);
}

void PVInspector::PVTabSplitter::edit_mapped(Picviz::PVMapped* mapped)
{
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(&mapped->get_mapping(), NULL, this);
	if (dlg->exec() == QDialog::Accepted) {
		// If a plotted was selected and that it is the current view...
		if (get_liv_view() == plotted->get_view() && !plotted->is_uptodate()) {
			// If something has changed, reprocess it
			plotted->process_from_parent_mapped(true);
		}
	}
}

void PVInspector::PVTabSplitter::edit_plotted(Picviz::PVPlotted* plotted)
{
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(NULL, &plotted->get_plotting(), this);
	dlg->exec();
}
