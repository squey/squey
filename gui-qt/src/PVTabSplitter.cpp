//! \file PVTabSplitter.cpp
//! $Id: PVTabSplitter.cpp 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) SÃ©bastien Tricaud 2009-2011
//! Copyright (C) Philippe SaadÃ© 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVProgressBox.h>
#include <picviz/PVView.h>

#include <PVAxisPropertiesWidget.h>
#include <PVMainWindow.h>
#include <PVListingView.h>
#include <PVExtractorWidget.h>
#include <PVAxesCombinationDialog.h>
#include <PVListingSortFilterProxyModel.h>
#include <PVListColNrawDlg.h>
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

	// SIZE STUFF
	// Nothing here !
	
	// OBJECTNAME STUFF
	setObjectName("PVTabSplitter");
	
	// FOCUS POLICY
	setFocusPolicy(Qt::StrongFocus);
	
	// We initialize our pointer to NULL
	pv_layer_stack_widget = NULL; // Note that this value can be requested during the creating of the PVLayerStackWidget!

	// PVLISTINGVIEW
	// We prepare the listing part and add it to the PVTabSplitter
	pv_listing_model = new PVListingModel(main_window, this);
	pv_listing_proxy_model = new PVListingSortFilterProxyModel(this, this);
	pv_listing_view = new PVListingView(main_window, this);
	pv_listing_proxy_model->setSourceModel(pv_listing_model);
	pv_listing_view->setModel(pv_listing_proxy_model);
	pv_listing_view->sortByColumn(-1, Qt::AscendingOrder);
	pv_listing_view->setSortingEnabled(true);
	addWidget(pv_listing_view);
	
	// Layout of the RIGHT_WIDGET
	// We prepare the right part of the view (with the listing and the Format editor)
	// We need a Layout
	QVBoxLayout* right_layout = new QVBoxLayout();
	// We set the margins in that Layout
	right_layout->setContentsMargins(10,10,10,10);
	
	// We prepare the PVLayerStackWidget and add it to the layout
	pv_layer_stack_model = new PVLayerStackModel(main_window, this);
	pv_layer_stack_widget = new PVLayerStackWidget(main_window, pv_layer_stack_model, this);
	right_layout->addWidget(pv_layer_stack_widget);
	
	// We prepare the PVViewsListingWidget and add it to the layout
	_views_widget = new PVViewsListingWidget(this);
	right_layout->addWidget(_views_widget);

	// RIGHT_WIDGET
	// Now we really create the right part QWidget and stuff it.
	QWidget* right_widget = new QWidget();
	// SIZE STUFF
	right_widget->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
	right_widget->setMinimumSize(QSize(100, 0));
	// OBJECTNAME STUFF
	right_widget->setObjectName("right_widget_of_PVTabSplitter");
	// FOCUS POLICY
	right_widget->setFocusPolicy(Qt::StrongFocus);
	
	// We put the right_layout in the RIGHT_WIDGET
	right_widget->setLayout(right_layout);
	
	// Now we can add the RIGHT_WIDGET to our PVTabSplitter
	addWidget(right_widget);
	
	// INITIAL SIZES
	// We now set the initial size of the components of that PVTabSplitter (mostly the right_widget...)
	QList<int> list_of_initial_sizes;
	list_of_initial_sizes << 1 << 220;
	setSizes(list_of_initial_sizes);

	// We also need a PVExtractorWidget
	_pv_extractor = new PVExtractorWidget(this);

	screenshot_index = 0;

	// Update notifications
	connect(pv_layer_stack_model, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(source_changed_Slot()));
	connect(pv_layer_stack_model, SIGNAL(layoutChanged()), this, SLOT(source_changed_Slot()));
	connect(_views_widget->get_model(), SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(source_changed_Slot()));
	connect(_views_widget->get_model(), SIGNAL(layoutChanged()), this, SLOT(source_changed_Slot()));

}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::~PVTabSplitter
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::~PVTabSplitter()
{
	PVLOG_INFO("In PVTabSplitter destructor\n");
	_pv_extractor->deleteLater();
//	pv_listing_view->deleteLater();
//	pv_listing_model->deleteLater();
//	pv_listing_no_unselected_model->deleteLater();
//	pv_listing_no_zombie_model->deleteLater();
//	pv_listing_no_zombie_no_unselected_model->deleteLater();

	QHash<Picviz::PVView const*, PVViewWidgets>::iterator it;
	for (it = _view_widgets.begin(); it != _view_widgets.end(); it++) {
		it.value().delete_widgets();
	}
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::create_new_mapped
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::create_new_mapped()
{
	Picviz::PVMapping new_mapping(get_lib_src().get());

	// Create new default name
	unsigned int nmapped = get_lib_src()->get_mappeds().size();
	QString new_name(tr("New mapping %1").arg(nmapped));
	new_mapping.set_name(new_name);

	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(&new_mapping, NULL, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	Picviz::PVMapped_p new_mapped(new Picviz::PVMapped(new_mapping));
	get_lib_src()->add_mapped(new_mapped);
	_views_widget->force_refresh();
}

void PVInspector::PVTabSplitter::toggle_listing_sort()
{
	if (pv_listing_view->isSortingEnabled()) {
		pv_listing_proxy_model->reset_to_default_ordering();
		pv_listing_view->setSortingEnabled(false);
	}
	else {
		pv_listing_view->sortByColumn(-1, Qt::AscendingOrder);
		pv_listing_view->setSortingEnabled(true);
	}
}


/******************************************************************************
 *
 * PVInspector::PVTabSplitter::create_new_plotted
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::create_new_plotted(Picviz::PVMapped* mapped_parent)
{
	Picviz::PVPlotting new_plotting(mapped_parent);

	// Create new default name
	unsigned int nplotted = mapped_parent->get_plotteds().size();
	QString new_name(tr("New plotting %1").arg(nplotted));
	new_plotting.set_name(new_name);

	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(NULL, &new_plotting, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	Picviz::PVPlotted_p new_plotted(new Picviz::PVPlotted(new_plotting));
	mapped_parent->add_plotted(new_plotted);
	_views_widget->force_refresh();
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::edit_mapped
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::edit_mapped(Picviz::PVMapped* mapped)
{
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(&mapped->get_mapping(), NULL, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	process_mapped_if_current(mapped);
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::edit_plotted
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::edit_plotted(Picviz::PVPlotted* plotted)
{
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(NULL, &plotted->get_plotting(), this);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	process_plotted_if_current(plotted);
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::ensure_column_visible
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::ensure_column_visible(PVCol col)
{
	// That's a hack to force our column to be at the left
	pv_listing_view->horizontalScrollBar()->setValue(pv_listing_view->horizontalScrollBar()->maximum());

	QModelIndex first_visible_idx = pv_listing_view->indexAt(QPoint(0,0));
	QModelIndex col_idx = pv_listing_model->index(first_visible_idx.row(), col);
	pv_listing_view->scrollTo(col_idx, QAbstractItemView::PositionAtTop);
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::get_axes_combination_editor
 *
 *****************************************************************************/
PVInspector::PVAxesCombinationDialog* PVInspector::PVTabSplitter::get_axes_combination_editor(Picviz::PVView_p view)
{
	PVViewWidgets const& widgets = get_view_widgets(view);
	return widgets.pv_axes_combination_editor;
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::get_axes_properties_widget
 *
 *****************************************************************************/
PVInspector::PVAxisPropertiesWidget* PVInspector::PVTabSplitter::get_axes_properties_widget(Picviz::PVView_p view)
{
	PVViewWidgets const& widgets = get_view_widgets(view);
	return widgets.pv_axes_properties;
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::get_current_view_name
 *
 *****************************************************************************/
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
 * PVInspector::PVTabSplitter::get_view_widgets
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVViewWidgets const& PVInspector::PVTabSplitter::get_view_widgets(Picviz::PVView_p view)
{
	assert(view->get_source_parent() == _lib_src.get());
	if (!_view_widgets.contains(view.get())) {
		PVViewWidgets widgets(view, this);
		return *(_view_widgets.insert(view.get(), widgets));
	}
	return _view_widgets[view.get()];
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
 * PVInspector::PVTabSplitter::process_extraction_job
 *
 *****************************************************************************/
bool PVInspector::PVTabSplitter::process_extraction_job(PVRush::PVControllerJob_p job)
{
	bool ret = true;
	PVRush::PVExtractor& ext = get_lib_src()->get_extractor();
	// Show a progress box that will finish with "accept" when the job is done
	if (!PVExtractorWidget::show_job_progress_bar(job, ext.get_format().get_format_name(), job->nb_elts_max(), this)) {
		ext.restore_nraw();
		ret = false;
	}
	else {
		get_lib_src()->wait_extract_end(job);
		if (ext.get_nraw().get_number_rows() == 0) {
			// Empty extraction, cancel it.
			QMessageBox::warning(this, tr("Empty extraction"), tr("The extraction just performed is empty. Returning to the previous state..."));
			ext.restore_nraw();
			ret = false;
		}
		else {
			ext.clear_saved_nraw();
		}
	}

	// Update libpicviz's views according to opened GL views (should be in the future PVSDK !!)
	QList<Picviz::PVView_p> views = main_window->list_displayed_picviz_views();
	for (int i = 0; i < views.size(); i++) {
		Picviz::PVView_p cur_view = views.at(i);
		if (cur_view->get_source_parent() != get_lib_src().get()) {
			continue;
		}
		if (ret) {
			// We can't cancel this process, because we could have already lost the previous mapping/plotting.
			// The user, by pressing cancel, expect to come back in 
			PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(tr("Processing..."), this);
			pbox->set_enable_cancel(false);
			PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, cur_view->get_plotted_parent(), false), pbox);
		}
		cur_view->set_consistent(true);

		// Send a message to PVGL
		PVSDK::PVMessage message;
		message.function = PVSDK_MESSENGER_FUNCTION_REINIT_PVVIEW;
		message.pv_view = cur_view;
		main_window->get_pvmessenger()->post_message_to_gl(message);
	}

	if (ret) {
		PVLOG_INFO("extractor: the normalization job took %0.4f seconds.\n", job->duration().seconds());
		emit_source_changed();
		refresh_layer_stack_view_Slot();
		refresh_listing_Slot();
	}

	return ret;
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::process_mapped_if_current
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::process_mapped_if_current(Picviz::PVMapped* mapped)
{
	Picviz::PVView_p cur_view = get_lib_view();
	if (cur_view->get_mapped_parent() == mapped) {
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVMapped::process_parent_source, mapped), tr("Processing..."), (QWidget*) this)) {
			return;
		}
		Picviz::PVPlotted* plotted_parent = cur_view->get_plotted_parent();
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, plotted_parent, true), tr("Processing..."), (QWidget*) this)) {
			return;
		}
	}

	main_window->update_pvglview(cur_view, PVSDK_MESSENGER_REFRESH_POSITIONS);
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::process_plotted_if_current
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::process_plotted_if_current(Picviz::PVPlotted* plotted)
{
	// If a plotted was selected and that it is the current view...
	if (get_lib_view() == plotted->get_view() && !plotted->is_uptodate()) {
		// If something has changed, reprocess it
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, plotted, true), tr("Processing..."), (QWidget*) this)) {
			return;
		}
		main_window->update_pvglview(plotted->get_view(), PVSDK_MESSENGER_REFRESH_POSITIONS);
	}
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::refresh_axes_combination_Slot
 *
 *****************************************************************************/
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
		//pv_listing_view->get_listing_model()->reset_model();
		pv_listing_view->refresh_listing_filter();
		//update the size of the corresponding table.
//		static_cast<PVListingModel*>(pv_listing_view->model())->initMatchingTable();
//                static_cast<PVListingModel*>(pv_listing_view->model())->initLocalMatchingTable();
//		static_cast<PVListingModel*>(pv_listing_view->model())->emitLayoutChanged();
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
 * PVInspector::PVTabSplitter::select_plotted
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::select_plotted(Picviz::PVPlotted* plotted)
{
	if (!plotted->is_uptodate()) {
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, plotted, true), tr("Processing..."), (QWidget*) this)) {
			return;
		}
		main_window->update_pvglview(plotted->get_view(), PVSDK_MESSENGER_REFRESH_POSITIONS);
	}
	select_view(plotted->get_view());
	main_window->ensure_glview_exists(plotted->get_view());
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::select_view
 *
 *****************************************************************************/
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
	pv_listing_proxy_model->reset_lib_view();
	pv_listing_view->update_view();
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
 * PVInspector::PVTabSplitter::source_changed_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::source_changed_Slot()
{
	emit source_changed();
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::update_pv_listing_model_Slot
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::update_pv_listing_model_Slot()
{
	PVLOG_DEBUG("%s \n       %s \n",__FILE__,__FUNCTION__);
	refresh_listing_Slot();
	updateFilterMenuEnabling();
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::updateFilterMenuEnabling
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::updateFilterMenuEnabling(){
	bool enable_menu = !get_lib_view()->is_real_output_selection_empty();
	emit selection_changed_signal(enable_menu);
}

size_t PVInspector::PVTabSplitter::get_unique_indexes_for_current_listing(PVCol column, std::vector<int>& idxes)
{
	// TODO: optimise to use current sorting if relevant
	Picviz::PVView_p current_lib_view = get_lib_view();
	size_t ret = 0;
	if (current_lib_view) {
		QVector<int> const& pidxes = pv_listing_proxy_model->get_proxy_indexes();
		idxes.resize(pidxes.size());
		std::copy(pidxes.begin(), pidxes.end(), idxes.begin());
		ret = current_lib_view->sort_unique_indexes_with_axes_combination(column, idxes);
	}
	return ret;
}

void PVInspector::PVTabSplitter::show_unique_values(PVCol col)
{
	std::vector<int> rows;
	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(tr("Computing values..."), this);
	pbox->set_enable_cancel(false);
	size_t nvalues;
	PVCore::PVProgressBox::progress(boost::bind(&PVTabSplitter::get_unique_indexes_for_current_listing, this, col, boost::ref(rows)), pbox, nvalues);
	if (nvalues == 0) {
		return;
	}

	PVListColNrawDlg* dlg = new PVListColNrawDlg(*get_lib_view(), rows, nvalues, col, this);
	dlg->exec();
}


// PVViewWidgets
/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVViewWidgets::PVViewWidgets
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVViewWidgets::PVViewWidgets(Picviz::PVView_p view, PVTabSplitter* tab)
{
	pv_axes_combination_editor = new PVAxesCombinationDialog(view, tab, tab->main_window);
	pv_axes_properties = new PVAxisPropertiesWidget(view, tab, tab->main_window);
}

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVViewWidgets::~PVViewWidgets
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVViewWidgets::~PVViewWidgets()
{
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVViewWidgets::delete_widgets
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::PVViewWidgets::delete_widgets()
{
	pv_axes_combination_editor->deleteLater();
	pv_axes_properties->deleteLater();
}
