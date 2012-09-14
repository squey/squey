/**
 * \file PVLayerStackView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QAction>
#include <QEvent>
#include <QFileDialog>
#include <QHeaderView>
#include <QMenu>

#include <pvguiqt/PVCustomQtRoles.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::PVLayerStackView
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackView::PVLayerStackView(QWidget* parent):
	QTableView(parent)
{
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

#if 0
	// We use a delegate to render the Icons 
	layer_stack_delegate = new PVLayerStackDelegate(mw, this);
	setItemDelegate(layer_stack_delegate);

	layer_stack_event_filter = new PVLayerStackEventFilter(mw, this);
	viewport()->installEventFilter(layer_stack_event_filter);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;
#endif

	connect(this, SIGNAL(clicked(QModelIndex const&)), this, SLOT(layer_clicked(QModelIndex const&)));

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
}


/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::import_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::import_layer()
{
#if 0
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
	Picviz::PVLayer* layer = ls_model()->lib_layer_stack().append_new_layer();

	// And load it
	layer->load_from_file(file);
	layer->compute_min_max(*_parent->get_parent_tab()->get_lib_view()->get_parent<Picviz::PVPlotted>());

	_parent->refresh();
#endif
#endif
}

PVGuiQt::PVLayerStackModel* PVGuiQt::PVLayerStackView::ls_model()
{
#ifdef NDEBUG
	return static_cast<PVLayerStackModel*>(model());
#else
	PVLayerStackModel* const ret = dynamic_cast<PVLayerStackModel*>(model());
	assert(ret);
	return ret;
#endif
}


/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::leaveEvent
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::leaveEvent(QEvent * /*event*/)
{
	PVLOG_DEBUG("PVGuiQt::PVLayerStackView::%s\n", __FUNCTION__);

	//mouse_hover_layer_index = -1;
	//last_mouse_hover_layer_index = -1;
	viewport()->update();
}



/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::load_layer_stack
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::load_layer_stack()
{
#if 0
#ifdef CUSTOMER_CAPABILITY_SAVE
	QFileDialog* dlg = new QFileDialog(this, tr("Import a layer stack..."), QString(), PICVIZ_LAYERSTACK_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	dlg->setFileMode(QFileDialog::ExistingFile);
	dlg->setAcceptMode(QFileDialog::AcceptOpen);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	QString file = dlg->selectedFiles().at(0);

	PVLayerStackModel* model_ = (PVLayerStackModel*) model();
	Picviz::PVLayerStack& stack = model_->lib_layer_stack();
	stack.load_from_file(file);
	stack.compute_min_maxs(*_parent->get_parent_tab()->get_lib_view()->get_parent<Picviz::PVPlotted>());

	_parent->refresh();
#endif
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::save_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::save_layer(int /*idx*/)
{
#if 0
#ifdef CUSTOMER_CAPABILITY_SAVE
	// Get layer with index 'idx'
	Picviz::PVLayer& layer = ((PVLayerStackModel*) model())->lib_layer_stack().get_selected_layer();

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
#endif
}



/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::save_layer_stack
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::save_layer_stack()
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

	//ls_model()->lib_layer_stack().save_to_file(file);
#endif
}



/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::show_ctxt_menu
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::show_ctxt_menu(const QPoint& pt)
{
	QModelIndex idx_click = indexAt(pt);

	QAction* act = _ctxt_menu->exec(QCursor::pos());
	if (act == _ctxt_menu_set_sel_layer) {
		//main_window->selection_set_from_current_layer_Slot();
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

void PVGuiQt::PVLayerStackView::layer_clicked(QModelIndex const& idx)
{
	if (!idx.isValid()) {
		// Qt says it's only called when idx is valid, but still..
		return;
	}

	ls_model()->setData(idx, QVariant(true), PVCustomQtRoles::RoleSetSelectedItem);
}
