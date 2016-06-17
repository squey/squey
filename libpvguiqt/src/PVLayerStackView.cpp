/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
#include <pvguiqt/PVExportSelectionDlg.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::PVLayerStackView
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackView::PVLayerStackView(QWidget* parent) : QTableView(parent)
{
	// SIZE STUFF
	setMinimumWidth(190);
	setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

	// OBJECTNAME STUFF
	setObjectName("PVLayerStackView");
	// We need to name the headers if we want to style them by CSS (without
	// interfering with other headers...
	horizontalHeader()->setObjectName("horizontalHeader_of_PVLayerStackView");
	verticalHeader()->setObjectName("verticalHeader_of_PVLayerStackView");

	// FOCUS POLICY
	setFocusPolicy(Qt::NoFocus);

	// HEADERS : we hide them
	verticalHeader()->hide();
	horizontalHeader()->hide();

	connect(this, &QTableView::clicked, [this](QModelIndex const& idx) {
		int layer_count = ls_model()->lib_view().get_layer_stack().get_layer_count();
		/* We create and store the true index of the layer in the lib */
		int lib_index = layer_count - 1 - idx.row();
		ls_model()->lib_view().set_layer_stack_selected_layer_index(lib_index);
	});

	// Context menu
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this,
	        SLOT(show_ctxt_menu(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	_ctxt_menu = new QMenu(this);
	_ctxt_menu->addSeparator();

	_ctxt_menu_set_sel_layer = new QAction(tr("Set selection from this layer content"), NULL);
	_ctxt_menu->addAction(_ctxt_menu_set_sel_layer);
	_ctxt_menu_export_layer_sel = new QAction(tr("Export this layer selection"), NULL);
	_ctxt_menu->addAction(_ctxt_menu_export_layer_sel);
	_ctxt_menu_reset_colors = new QAction(tr("Reset this layer colors to white"), NULL);
	_ctxt_menu->addAction(_ctxt_menu_reset_colors);

	_ctxt_menu->addSeparator();
	_ctxt_menu->addSeparator();

	_ctxt_menu_copy_to_clipboard_act =
	    new QAction(tr("Copy layerstack details to clipboard"), nullptr);
	_ctxt_menu->addAction(_ctxt_menu_copy_to_clipboard_act);
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
void PVGuiQt::PVLayerStackView::leaveEvent(QEvent* /*event*/)
{
	PVLOG_DEBUG("PVGuiQt::PVLayerStackView::%s\n", __FUNCTION__);

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
		Inendi::PVLayer& layer = get_layer_from_idx(model_index);
		QString current_name = layer.get_name();
		QString name = QInputDialog::getText(this, "Rename current layer", "New layer name:",
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

Inendi::PVLayer& PVGuiQt::PVLayerStackView::get_layer_from_idx(int model_idx)
{
	QVariant var =
	    ls_model()->data(ls_model()->index(model_idx, 0), PVCustomQtRoles::UnderlyingObject);
	return *reinterpret_cast<Inendi::PVLayer*>(var.value<void*>());
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
	if (act == _ctxt_menu_export_layer_sel) {
		export_layer_selection(idx_click.row());
		return;
	}
	if (act == _ctxt_menu_reset_colors) {
		reset_layer_colors(idx_click.row());
	}
	if (act == _ctxt_menu_copy_to_clipboard_act) {
		copy_to_clipboard();
	}
}

void PVGuiQt::PVLayerStackView::set_current_selection_from_layer(int model_idx)
{
	Inendi::PVLayer const& layer = get_layer_from_idx(model_idx);
	ls_model()->lib_view().set_selection_from_layer(layer);
	ls_model()->lib_view().process_real_output_selection();
}

void PVGuiQt::PVLayerStackView::export_layer_selection(int model_idx)
{
	Inendi::PVLayer const& layer = get_layer_from_idx(model_idx);

	const Inendi::PVSelection& sel = layer.get_selection();
	Inendi::PVView& view = ls_model()->lib_view();

	PVGuiQt::PVExportSelectionDlg::export_selection(view, sel);
}

void PVGuiQt::PVLayerStackView::reset_layer_colors(int layer_idx)
{
	ls_model()->reset_layer_colors(layer_idx);
}
