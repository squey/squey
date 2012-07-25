/**
 * \file PVLayerStackView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtGui>
#include <QEvent>

#include <pvkernel/core/general.h>
//#include <picviz/PVView.h>

#include <PVMainWindow.h>

#include <PVLayerStackView.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::PVLayerStackView
 *
 *****************************************************************************/
PVInspector::PVLayerStackView::PVLayerStackView(PVMainWindow *mw, PVLayerStackModel *model, PVLayerStackWidget *parent):
	QTableView(parent),
	_parent(parent)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackView::%s\n", __FUNCTION__);

	main_window = mw;

	// SIZE STUFF
	setMinimumWidth(190);
	setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);

	// OBJECTNAME STUFF
	setObjectName("PVLayerStackView");
	// We need to name the headers if we want to style them by CSS (without interfering with other headers...
	horizontalHeader()->setObjectName("horizontalHeader_of_PVLayerStackView");
	verticalHeader()->setObjectName("verticalHeader_of_PVLayerStackView");
	
	// FOCUS POLICY
	setFocusPolicy(Qt::NoFocus);
	
	// HEADERS : we hide them
	verticalHeader()->hide();
	horizontalHeader()->hide();

	//viewport()->setMouseTracking(true);
	//viewport()->setAttribute(Qt::WA_Hover, true);

	// We use a delegate to render the Icons 
	layer_stack_delegate = new PVLayerStackDelegate(mw, this);
	setItemDelegate(layer_stack_delegate);

	layer_stack_event_filter = new PVLayerStackEventFilter(mw, this);
	viewport()->installEventFilter(layer_stack_event_filter);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;

	// Context menu
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	_ctxt_menu = new QMenu(this);
#ifdef CUSTOMER_CAPABILITY_SAVE
	_ctxt_menu_save_act = new QAction(tr("Export this layer..."), NULL);
	_ctxt_menu_load_act = new QAction(tr("Import a layer..."), NULL);
	_ctxt_menu->addAction(_ctxt_menu_save_act);
	_ctxt_menu->addAction(_ctxt_menu_load_act);

	_ctxt_menu->addSeparator();
	_ctxt_menu_save_ls_act = new QAction(tr("Save the layer stack..."), NULL);
	_ctxt_menu_load_ls_act = new QAction(tr("Load a layer stack..."), NULL);
	_ctxt_menu->addAction(_ctxt_menu_save_ls_act);
	_ctxt_menu->addAction(_ctxt_menu_load_ls_act);
	_ctxt_menu->addSeparator();
#endif
	_ctxt_menu_set_sel_layer = new QAction(tr("Set selection from this layer..."), NULL);
	_ctxt_menu->addAction(_ctxt_menu_set_sel_layer);

	setModel(model);
	resizeColumnsToContents();
	//resizeRowsToContents();
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackView::import_layer
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::import_layer()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QFileDialog* dlg = new QFileDialog(this, tr("Import a layer..."), QString(), PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	dlg->setFileMode(QFileDialog::ExistingFile);
	dlg->setAcceptMode(QFileDialog::AcceptOpen);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	QString file = dlg->selectedFiles().at(0);

	// Create a new layer
	PVLayerStackModel* model_ = (PVLayerStackModel*) model();
	Picviz::PVLayer* layer = model_->get_layer_stack_lib().append_new_layer();

	// And load it
	layer->load_from_file(file);
	layer->compute_min_max(*_parent->get_parent_tab()->get_lib_view()->get_parent<Picviz::PVPlotted>());

	_parent->refresh();
#endif
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackView::leaveEvent
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::leaveEvent(QEvent * /*event*/)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackView::%s\n", __FUNCTION__);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;
	viewport()->update();
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackView::load_layer_stack
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::load_layer_stack()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QFileDialog* dlg = new QFileDialog(this, tr("Import a layer stack..."), QString(), PICVIZ_LAYERSTACK_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	dlg->setFileMode(QFileDialog::ExistingFile);
	dlg->setAcceptMode(QFileDialog::AcceptOpen);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	QString file = dlg->selectedFiles().at(0);

	PVLayerStackModel* model_ = (PVLayerStackModel*) model();
	Picviz::PVLayerStack& stack = model_->get_layer_stack_lib();
	stack.load_from_file(file);
	stack.compute_min_maxs(*_parent->get_parent_tab()->get_lib_view()->get_parent<Picviz::PVPlotted>());

	_parent->refresh();
#endif
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackView::save_layer
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::save_layer(int /*idx*/)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	// Get layer with index 'idx'
	Picviz::PVLayer& layer = ((PVLayerStackModel*) model())->get_layer_stack_lib().get_selected_layer();

	// Ask for a filename
	QFileDialog* dlg = new QFileDialog(this, tr("Choose a file..."), QString(), PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	dlg->setAcceptMode(QFileDialog::AcceptSave);
	dlg->setDefaultSuffix(PICVIZ_LAYER_ARCHIVE_EXT);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	QString file = dlg->selectedFiles().at(0);

	layer.save_to_file(file);
#endif
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackView::save_layer_stack
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::save_layer_stack()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	// Ask for a filename
	QFileDialog* dlg = new QFileDialog(this, tr("Choose a file..."), QString(), PICVIZ_LAYERSTACK_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	dlg->setAcceptMode(QFileDialog::AcceptSave);
	dlg->setDefaultSuffix(PICVIZ_LAYERSTACK_ARCHIVE_EXT);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	QString file = dlg->selectedFiles().at(0);

	((PVLayerStackModel*) model())->get_layer_stack_lib().save_to_file(file);
#endif
}



/******************************************************************************
 *
 * PVInspector::PVLayerStackView::show_ctxt_menu
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::show_ctxt_menu(const QPoint& pt)
{
	QModelIndex idx_click = indexAt(pt);

	QAction* act = _ctxt_menu->exec(QCursor::pos());
	if (act == _ctxt_menu_set_sel_layer) {
		main_window->selection_set_from_current_layer_Slot();
		return;
	}
#ifdef CUSTOMER_CAPABILITY_SAVE
	_ctxt_menu_save_act->setEnabled(idx_click.isValid());
	if (act == _ctxt_menu_save_act) {
		save_layer(idx_click.row());
	}
	else
	if (act == _ctxt_menu_load_act) {
		import_layer();
	}
	else
	if (act == _ctxt_menu_save_ls_act) {
		save_layer_stack();
	}
	else
	if (act == _ctxt_menu_load_ls_act) {
		load_layer_stack();
	}
#endif
}



