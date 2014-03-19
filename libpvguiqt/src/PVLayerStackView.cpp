/**
 * \file PVLayerStackView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QAction>
#include <QEvent>
#include <QHeaderView>
#include <QMouseEvent>
#include <QMenu>
#include <QInputDialog>

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
	connect(this, SIGNAL(doubleClicked(QModelIndex const&)), this, SLOT(layer_double_clicked(QModelIndex const&)));

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
#endif

	_ctxt_menu_set_sel_layer = new QAction(tr("Set selection from this layer's content"), NULL);
	_ctxt_menu->addAction(_ctxt_menu_set_sel_layer);
	_ctxt_menu_reset_colors = new QAction(tr("Reset this layer's colors to white"), NULL);
	_ctxt_menu->addAction(_ctxt_menu_reset_colors);

#ifdef CUSTOMER_CAPABILITY_SAVE
	_ctxt_menu->addSeparator();
	_ctxt_menu_save_ls_act = new QAction(tr("Save the layer stack..."), NULL);
	_ctxt_menu_load_ls_act = new QAction(tr("Load a layer stack..."), NULL);
	_ctxt_menu->addAction(_ctxt_menu_save_ls_act);
	_ctxt_menu->addAction(_ctxt_menu_load_ls_act);
	_ctxt_menu->addSeparator();
#endif

	_ctxt_menu_copy_to_clipboard_act = new QAction(tr("Copy the layer stack's details to clipboard"), nullptr);
	_ctxt_menu->addAction(_ctxt_menu_copy_to_clipboard_act);
}


/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::import_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::import_layer()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QString file = _layer_dialog.getOpenFileName(this, tr("Import a layer..."), _layer_dialog.directory().absolutePath(), PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);

	if (!file.isEmpty()) {
		// Create a new layer
		ls_model()->add_new_layer_from_file(file);
	}
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
 * PVGuiQt::PVLayerStackView::enterEvent
 *****************************************************************************/

void PVGuiQt::PVLayerStackView::enterEvent(QEvent*)
{
	setFocus(Qt::MouseFocusReason);
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
	clearFocus();
}

void PVGuiQt::PVLayerStackView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QModelIndex idx = indexAt(event->pos());
	if (!idx.isValid()) {
		return;
	}

	if (idx.column() == 1) {
		edit(idx);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::keyPressEvent
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::keyPressEvent(QKeyEvent* event)
{
	switch (event->key()) {
	case Qt::Key_F2:
		int model_index = ls_model()->lib_layer_stack().get_selected_layer_index();
		Picviz::PVLayer &layer = get_layer_from_idx(model_index);
		QString current_name = layer.get_name();
		QString name = QInputDialog::getText(this,
		                                     "Rename current layer",
		                                     "New layer name:",
		                                     QLineEdit::Normal, current_name);
		if (!name.isEmpty()) {
			layer.set_name(name);
		}

		event->accept();
	}

	if (event->isAccepted() == false) {
		QTableView::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::load_layer_stack
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::load_layer_stack()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QString file = _layerstack_dialog.getOpenFileName(this, tr("Import a layer stack..."), _layerstack_dialog.directory().absolutePath(), PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if(!file.isEmpty()) {
		ls_model()->load_from_file(file);
	}
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::save_layer
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::save_layer(int idx)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QString file = _layer_dialog.getSaveFileName(this, tr("Export this layer..."), _layer_dialog.directory().absolutePath(), PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if (!file.isEmpty()) {
		get_layer_from_idx(idx).save_to_file(file);
	}
#endif
}

Picviz::PVLayer& PVGuiQt::PVLayerStackView::get_layer_from_idx(int model_idx)
{
	QVariant var = ls_model()->data(ls_model()->index(model_idx, 0), PVCustomQtRoles::UnderlyingObject);
	return *reinterpret_cast<Picviz::PVLayer*>(var.value<void*>());
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::save_layer_stack
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::save_layer_stack()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	QString file = _layerstack_dialog.getSaveFileName(this, tr("Save layer stack..."), _layerstack_dialog.directory().absolutePath(), PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if(!file.isEmpty()) {
		Picviz::PVLayerStack& layer_stack = ls_model()->lib_layer_stack();
		layer_stack.save_to_file(file);
	}
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackView::copy_to_clipboard
 *
 *****************************************************************************/
void PVGuiQt::PVLayerStackView::copy_to_clipboard()
{
	ls_model()->lib_layer_stack().copy_details_to_clipboard();
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
		set_current_selection_from_layer(idx_click.row());
		return;
	}
	if (act == _ctxt_menu_reset_colors) {
		reset_layer_colors(idx_click.row());
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
	if(act == _ctxt_menu_copy_to_clipboard_act) {
		copy_to_clipboard();
	}
}

void PVGuiQt::PVLayerStackView::set_current_selection_from_layer(int model_idx)
{
	Picviz::PVLayer const& layer = get_layer_from_idx(model_idx);
	ls_model()->view_actor().call<FUNC(Picviz::PVView::set_selection_from_layer)>(layer);
	ls_model()->view_actor().call<FUNC(Picviz::PVView::process_real_output_selection)>();
}

void PVGuiQt::PVLayerStackView::reset_layer_colors(int layer_idx)
{
	ls_model()->reset_layer_colors(layer_idx);
}

void PVGuiQt::PVLayerStackView::layer_clicked(QModelIndex const& idx)
{
	if (!idx.isValid()) {
		// Qt says it's only called when idx is valid, but still..
		return;
	}
	
	ls_model()->setData(idx, QVariant(true), PVCustomQtRoles::RoleSetSelectedItem);
}

void PVGuiQt::PVLayerStackView::layer_double_clicked(QModelIndex const&)
{
}
