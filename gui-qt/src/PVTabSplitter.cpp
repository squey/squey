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
#include <PVViewsModel.h>

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
	_views_widget = new PVViewsListingWidget(this);
	right_layout->addWidget(_views_widget);

	QWidget* right_widget = new QWidget();
	right_widget->setFocusPolicy(Qt::StrongFocus);
	right_widget->setLayout(right_layout);

	QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
	right_widget->setSizePolicy(sizePolicy);
	right_widget->setMinimumSize(QSize(229, 0));
	addWidget(right_widget);

	_pv_extractor = new PVExtractorWidget(this);

	screenshot_index = 0;

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::StrongFocus);

	// Update notifications
	connect(pv_layer_stack_model, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(source_changed_Slot()));
	connect(pv_layer_stack_model, SIGNAL(layoutChanged()), this, SLOT(source_changed_Slot()));
	connect(_views_widget->get_model(), SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(source_changed_Slot()));
	connect(_views_widget->get_model(), SIGNAL(layoutChanged()), this, SLOT(source_changed_Slot()));

	// HACK: AG: this is ugly but I can't find another way to have
	// the right widget with 220px width at the beggining.
	QList<int> wsizes = sizes();
	wsizes[1] = 229;
	wsizes[0] = 100000;
	setSizes(wsizes);
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
	// TODO: do this only for the good view
	QHash<Picviz::PVView const*, PVViewWidgets>::const_iterator it;
	for (it = _view_widgets.begin(); it != _view_widgets.end(); it++) {
		PVAxesCombinationDialog* pv_axes_combination_editor = it.value().pv_axes_combination_editor; 
		if (pv_axes_combination_editor->isVisible()) {
			pv_axes_combination_editor->update_used_axes();
		}
	}
}

PVInspector::PVAxesCombinationDialog* PVInspector::PVTabSplitter::get_axes_combination_editor(Picviz::PVView_p view)
{
	PVViewWidgets const& widgets = get_view_widgets(view);
	return widgets.pv_axes_combination_editor;
}

PVInspector::PVTabSplitter::PVViewWidgets const& PVInspector::PVTabSplitter::get_view_widgets(Picviz::PVView_p view)
{
	assert(view->get_source_parent() == _lib_src.get());
	if (!_view_widgets.contains(view.get())) {
		PVViewWidgets widgets(view, this);
		return *(_view_widgets.insert(view.get(), widgets));
	}
	return _view_widgets[view.get()];
}

void PVInspector::PVTabSplitter::select_view(Picviz::PVView_p view)
{
	assert(view->get_source_parent() == _lib_src.get());
	_lib_src->select_view(view);

	// Create view widgets if necessary
	get_view_widgets(view);

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

	// Create new default name
	unsigned int nmapped = get_lib_src()->get_mappeds().size();
	QString new_name(tr("New mapped %1").arg(nmapped));
	new_mapping.set_name(new_name);

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
		main_window->update_pvglview(plotted->get_view(), PVSDK_MESSENGER_REFRESH_POSITIONS);
	}
	select_view(plotted->get_view());
	main_window->ensure_glview_exists(plotted->get_view());
}

void PVInspector::PVTabSplitter::create_new_plotted(Picviz::PVMapped* mapped_parent)
{
	Picviz::PVPlotting new_plotting(mapped_parent);

	// Create new default name
	unsigned int nplotted = mapped_parent->get_plotteds().size();
	QString new_name(tr("New plotted %1").arg(nplotted));
	new_plotting.set_name(new_name);

	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(NULL, &new_plotting, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	Picviz::PVPlotted_p new_plotted(new Picviz::PVPlotted(new_plotting));
	mapped_parent->add_plotted(new_plotted);
	_views_widget->force_refresh();
}

void PVInspector::PVTabSplitter::edit_mapped(Picviz::PVMapped* mapped)
{
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(&mapped->get_mapping(), NULL, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	Picviz::PVView_p cur_view = get_lib_view();
	if (cur_view->get_mapped_parent() == mapped) {
		mapped->process_parent_source();
		cur_view->get_plotted_parent()->process_from_parent_mapped(true);
		main_window->update_pvglview(cur_view, PVSDK_MESSENGER_REFRESH_POSITIONS);
	}
}

void PVInspector::PVTabSplitter::edit_plotted(Picviz::PVPlotted* plotted)
{
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(NULL, &plotted->get_plotting(), this);
	if (dlg->exec() == QDialog::Accepted) {
		// If a plotted was selected and that it is the current view...
		if (get_lib_view() == plotted->get_view() && !plotted->is_uptodate()) {
			// If something has changed, reprocess it
			plotted->process_from_parent_mapped(true);
			main_window->update_pvglview(plotted->get_view(), PVSDK_MESSENGER_REFRESH_POSITIONS);
		}
	}
}

QString PVInspector::PVTabSplitter::get_current_view_name(Picviz::PVSource_p src)
{
	Picviz::PVView_p view = src->current_view();
	if (view) {
		return view->get_window_name();
	}

	QString ret = get_tab_name(src) + " | ";
	ret += "mapped/plotted: default/default";
	return ret;
}

void PVInspector::PVTabSplitter::source_changed_Slot()
{
	emit source_changed();
}

// PVViewWidgets
PVInspector::PVTabSplitter::PVViewWidgets::PVViewWidgets(Picviz::PVView_p view, PVTabSplitter* tab)
{
	pv_axes_combination_editor = new PVAxesCombinationDialog(view, tab, tab->main_window);
}
