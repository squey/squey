/**
 * \file PVTabSplitter.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVProgressBox.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/waxes/waxes.h>

#include <PVAxisPropertiesWidget.h>
#include <PVMainWindow.h>
#include <PVSimpleStringListModel.h>
#include <PVExtractorWidget.h>
#include <PVMappingPlottingEditDialog.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVLayerStackWidget.h>
#include <pvguiqt/PVListDisplayDlg.h>
#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <PVTabSplitter.h>

#include <QScrollBar>
#include <QMessageBox>
#include <QVBoxLayout>

/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVTabSplitter
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVTabSplitter(Picviz::PVSource& lib_src, QWidget *parent) :
	QSplitter(parent)
{
	// Observer on this PVSource
	_obs_src.connect_about_to_be_deleted(this, SLOT(source_about_to_be_deleted()));
	{
		Picviz::PVSource_sp src = lib_src.shared_from_this();
		PVHive::get().register_observer(src, _obs_src);
	}

	assert(get_lib_src()->get_children<Picviz::PVView>().size() > 0);

	// SIZE STUFF
	// Nothing here !
	
	// OBJECTNAME STUFF
	setObjectName("PVTabSplitter");
	
	// FOCUS POLICY
	setFocusPolicy(Qt::StrongFocus);
	
	// PVLISTINGVIEW
	Picviz::PVView_sp cur_view = get_lib_src()->current_view()->shared_from_this();
	pv_listing_model = new PVGuiQt::PVListingModel(cur_view, this);
	pv_listing_proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(cur_view, this);
	pv_listing_view = new PVGuiQt::PVListingView(cur_view, this);
	pv_listing_proxy_model->setSourceModel(pv_listing_model);
	pv_listing_view->setModel(pv_listing_proxy_model);
	pv_listing_view->sortByColumn(-1, Qt::AscendingOrder);
	pv_listing_view->setSortingEnabled(true);
	addWidget(pv_listing_view);

	// Invalid elements widget
	PVSimpleStringListModel<QStringList>* inv_elts_model = new PVSimpleStringListModel<QStringList>(get_lib_src()->get_invalid_elts());
	PVGuiQt::PVListDisplayDlg* inv_dlg = new PVGuiQt::PVListDisplayDlg(inv_elts_model, this);
	inv_dlg->setWindowTitle(tr("Invalid elements"));
	inv_dlg->set_description(tr("There were invalid elements during the extraction:"));
	_inv_elts_dlg = inv_dlg;
	
	// Layout of the RIGHT_WIDGET
	// We prepare the right part of the view (with the LayerStack and the Format editor)
	// We need a Layout
	QVBoxLayout* right_layout = new QVBoxLayout();
	// We set the margins in that Layout
	right_layout->setContentsMargins(8,8,8,8);
	
	// We prepare the PVLayerStackWidget and add it to the layout
	pv_layer_stack_widget = new PVGuiQt::PVLayerStackWidget(cur_view);
	right_layout->addWidget(pv_layer_stack_widget);
	
	_data_tree_model = new PVGuiQt::PVRootTreeModel(lib_src);
	_data_tree_view = new PVGuiQt::PVRootTreeView(_data_tree_model);
	
	right_layout->addWidget(_data_tree_view);

	// RIGHT_WIDGET
	// Now we really create the right part QWidget and stuff it.
	QWidget* right_widget = new QWidget();
	right_widget->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
	right_widget->setMinimumSize(QSize(100, 0));
	right_widget->setObjectName("right_widget_of_PVTabSplitter");
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
	//connect(pv_layer_stack_model, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(source_changed_Slot()));
	//connect(pv_layer_stack_model, SIGNAL(layoutChanged()), this, SLOT(source_changed_Slot()));
	//connect(_views_widget->get_model(), SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(source_changed_Slot()));
	//connect(_views_widget->get_model(), SIGNAL(layoutChanged()), this, SLOT(source_changed_Slot()));

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
	pv_listing_view->deleteLater();
	pv_listing_model->deleteLater();
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
#if 0
	Picviz::PVMapped_p mapped(get_lib_src()->shared_from_this());

	Picviz::PVMapping* new_mapping = new Picviz::PVMapping(mapped.get());

	// Create new default name
	unsigned int nmapped = get_lib_src()->get_children<Picviz::PVMapped>().size();
	QString new_name(tr("New mapping %1").arg(nmapped));
	new_mapping->set_name(new_name);
	mapped->set_mapping(new_mapping);

	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(new_mapping, NULL, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	//_views_widget->force_refresh();
#endif
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
#if 0
	Picviz::PVPlotted_p plotted(mapped_parent->shared_from_this());
	Picviz::PVPlotting_p new_plotting(new Picviz::PVPlotting(plotted.get()));

	// Create new default name
	unsigned int nplotted = mapped_parent->get_children<Picviz::PVPlotted>().size();
	QString new_name(tr("New plotting %1").arg(nplotted));
	new_plotting->set_name(new_name);
	plotted->set_plotting(new_plotting);

	PVMappingPlottingEditDialog* dlg = new PVMappingPlottingEditDialog(NULL, new_plotting.get(), this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	//_views_widget->force_refresh();
#endif
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::edit_mapped
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::edit_mapped(Picviz::PVMapped* mapped)
{
#if 0
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(mapped->get_mapping(), NULL, this);
	if (dlg->exec() == QDialog::Rejected) {
		return;
	}

	process_mapped_if_current(mapped);
#endif
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::edit_plotted
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::edit_plotted(Picviz::PVPlotted* plotted)
{
#if 0
	PVMappingPlottingEditDialog* dlg;
	dlg = new PVMappingPlottingEditDialog(NULL, &plotted->get_plotting(), this);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	process_plotted_if_current(plotted);
#endif
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
PVGuiQt::PVAxesCombinationDialog* PVInspector::PVTabSplitter::get_axes_combination_editor(Picviz::PVView* view)
{
	PVViewWidgets const& widgets = get_view_widgets(view);
	return widgets.pv_axes_combination_editor;
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::get_axes_properties_widget
 *
 *****************************************************************************/
PVInspector::PVAxisPropertiesWidget* PVInspector::PVTabSplitter::get_axes_properties_widget(Picviz::PVView* view)
{
	PVViewWidgets const& widgets = get_view_widgets(view);
	return widgets.pv_axes_properties;
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::get_current_view_name
 *
 *****************************************************************************/
QString PVInspector::PVTabSplitter::get_current_view_name(Picviz::PVSource* src)
{
	Picviz::PVView* view = src->current_view();
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
PVInspector::PVTabSplitter::PVViewWidgets const& PVInspector::PVTabSplitter::get_view_widgets(Picviz::PVView* view)
{
	assert(view->get_parent<Picviz::PVSource>() == get_lib_src());
	if (!_view_widgets.contains(view)) {
		PVViewWidgets widgets(view, this);
		return *(_view_widgets.insert(view, widgets));
	}
	return _view_widgets[view];
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

#if 0
	// Update libpicviz's views according to opened GL views (should be in the future PVSDK !!)
	QList<Picviz::PVView_sp> views = main_window->list_displayed_picviz_views();
	for (int i = 0; i < views.size(); i++) {
		Picviz::PVView_sp cur_view = views.at(i);
		if (cur_view->get_parent<Picviz::PVSource>() != get_lib_src().get()) {
			continue;
		}
		if (ret) {
			// We can't cancel this process, because we could have already lost the previous mapping/plotting.
			// The user, by pressing cancel, expect to come back in 
			PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(tr("Processing..."), this);
			pbox->set_enable_cancel(false);
			PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, cur_view->get_parent<Picviz::PVPlotted>()), pbox);
		}
		cur_view->set_consistent(true);

		// Send a message to PVGL
		/*PVSDK::PVMessage message;
		message.function = PVSDK_MESSENGER_FUNCTION_REINIT_PVVIEW;
		message.pv_view = cur_view;
		main_window->get_pvmessenger()->post_message_to_gl(message);*/
	}

	if (ret) {
		PVLOG_INFO("extractor: the normalization job took %0.4f seconds.\n", job->duration().seconds());
		/*
		emit_source_changed();
		refresh_layer_stack_view_Slot();
		refresh_listing_Slot();
		pv_listing_model->reset_lib_view();
		pv_listing_proxy_model->reset_lib_view();*/
	}
#endif

	return ret;
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::process_mapped_if_current
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::process_mapped_if_current(Picviz::PVMapped* mapped)
{
	Picviz::PVView* cur_view = get_lib_view();
	if (cur_view->get_parent<Picviz::PVMapped>() == mapped) {
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVMapped::process_parent_source, mapped), tr("Processing..."), (QWidget*) this)) {
			return;
		}
		Picviz::PVPlotted* plotted_parent = cur_view->get_parent<Picviz::PVPlotted>();
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, plotted_parent), tr("Processing..."), (QWidget*) this)) {
			return;
		}
	}

	//main_window->update_pvglview(cur_view, PVSDK_MESSENGER_REFRESH_POSITIONS);
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::process_plotted_if_current
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::process_plotted_if_current(Picviz::PVPlotted* plotted)
{
	// If a plotted was selected and that it is the current view...
	if (get_lib_view() == plotted->current_view() && !plotted->is_uptodate()) {
		// If something has changed, reprocess it
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, plotted), tr("Processing..."), (QWidget*) this)) {
			return;
		}
		//main_window->update_pvglview(plotted->current_view(), PVSDK_MESSENGER_REFRESH_POSITIONS);
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
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVPlotted::process_from_parent_mapped, plotted), tr("Processing..."), (QWidget*) this)) {
			return;
		}
		//main_window->update_pvglview(plotted->current_view(), PVSDK_MESSENGER_REFRESH_POSITIONS);
	}
	select_view(plotted->current_view());
	//main_window->ensure_glview_exists(plotted->current_view());
}



/******************************************************************************
 *
 * PVInspector::PVTabSplitter::select_view
 *
 *****************************************************************************/
void PVInspector::PVTabSplitter::select_view(Picviz::PVView* view)
{
	// TODO: hive !
	assert(view->get_parent<Picviz::PVSource>() == get_lib_src());
	get_lib_src()->select_view(*view);

	// Create view widgets if necessary
	get_view_widgets(view);
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
	Picviz::PVView* current_lib_view = get_lib_view();
	size_t ret = 0;
	if (current_lib_view) {
		QVector<int> const& pidxes = pv_listing_proxy_model->get_proxy_indexes();
		idxes.resize(pidxes.size());
		std::copy(pidxes.begin(), pidxes.end(), idxes.begin());
		ret = current_lib_view->sort_unique_indexes_with_axes_combination(column, idxes);
	}
	return ret;
}

// PVViewWidgets
/******************************************************************************
 *
 * PVInspector::PVTabSplitter::PVViewWidgets::PVViewWidgets
 *
 *****************************************************************************/
PVInspector::PVTabSplitter::PVViewWidgets::PVViewWidgets(Picviz::PVView* view, PVTabSplitter* tab)
{
	Picviz::PVView_sp view_sp = view->shared_from_this();
	pv_axes_combination_editor = new PVGuiQt::PVAxesCombinationDialog(view_sp, tab);
	//pv_axes_properties = new PVAxisPropertiesWidget(view_sp, tab, tab->main_window);
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
	//pv_axes_properties->deleteLater();
}

void PVInspector::PVTabSplitter::source_about_to_be_deleted()
{
	hide();
	deleteLater();
}
