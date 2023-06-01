//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <QAction>
#include <QEvent>
#include <QHeaderView>
#include <QMouseEvent>
#include <QMenu>
#include <QInputDialog>

#include <squey/PVSelection.h>

#include <pvguiqt/PVCustomQtRoles.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>
#include <pvguiqt/PVExportSelectionDlg.h>

/******************************************************************************
 *
 * App::PVLayerStackView::PVLayerStackView
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
	connect(this, &QWidget::customContextMenuRequested, this, &PVLayerStackView::show_ctxt_menu);
	setContextMenuPolicy(Qt::CustomContextMenu);

	_ctxt_menu = new QMenu(this);
	_ctxt_menu->setToolTipsVisible(true);
	_ctxt_menu->addSeparator();

	_ctxt_menu_set_sel_layer = new QAction(tr("Set selection from this layer content"), nullptr);
	_ctxt_menu->addAction(_ctxt_menu_set_sel_layer);
	_ctxt_menu_export_layer_sel = new QAction(tr("Export this layer selection"), nullptr);
	_ctxt_menu->addAction(_ctxt_menu_export_layer_sel);
	_ctxt_menu_reset_colors = new QAction(tr("Reset this layer colors to white"), nullptr);
	_ctxt_menu->addAction(_ctxt_menu_reset_colors);

	_ctxt_menu->addSeparator();

	_ctxt_menu_show_this_layer_only = new QAction(tr("Show this layer only"), nullptr);
	_ctxt_menu->addAction(_ctxt_menu_show_this_layer_only);

	_ctxt_menu->addSeparator();

	_ctxt_menu_union = new QAction(QIcon(":/union"), tr("Union"), nullptr);
	_ctxt_menu_union->setToolTip(
	    "Union of the current selection and this layer content (restricted to visible layer)");
	_ctxt_menu->addAction(_ctxt_menu_union);

	_ctxt_menu_intersection = new QAction(QIcon(":/intersection"), tr("Intersection"), nullptr);
	_ctxt_menu_intersection->setToolTip("Intersection of the current selection and this layer "
	                                    "content (restricted to visible layer)");
	_ctxt_menu->addAction(_ctxt_menu_intersection);

	_ctxt_menu_difference = new QAction(QIcon(":/difference"), tr("Difference"), nullptr);
	_ctxt_menu_difference->setToolTip(
	    "Difference of the current selection and this layer content (restricted to visible layer)");
	_ctxt_menu->addAction(_ctxt_menu_difference);

	_ctxt_menu_symmetric_differrence =
	    new QAction(QIcon(":/symmetric"), tr("Symmetric difference"), nullptr);
	_ctxt_menu_symmetric_differrence->setToolTip("Symmetric difference of the current selection "
	                                             "and this layer content (restricted to visible "
	                                             "layer)");
	_ctxt_menu->addAction(_ctxt_menu_symmetric_differrence);

	_ctxt_menu_activate_union =
	    new QAction(QIcon(":/union_activate"), tr("Activate and Union"), nullptr);
	_ctxt_menu_activate_union->setToolTip(
	    "Activate this layer and Union of the current selection and its content");
	_ctxt_menu->addAction(_ctxt_menu_activate_union);

	_ctxt_menu_activate_intersection =
	    new QAction(QIcon(":/intersection_activate"), tr("Activate and Intersection"), nullptr);
	_ctxt_menu_activate_intersection->setToolTip(
	    "Activate this layer and Intersection of the current selection and its content");
	_ctxt_menu->addAction(_ctxt_menu_activate_intersection);

	_ctxt_menu_activate_difference =
	    new QAction(QIcon(":/difference_activate"), tr("Activate and Difference"), nullptr);
	_ctxt_menu_activate_difference->setToolTip(
	    "Activate this layer and Difference of the current selection and its content");
	_ctxt_menu->addAction(_ctxt_menu_activate_difference);

	_ctxt_menu_activate_symmetric_differrence = new QAction(
	    QIcon(":/symmetric_activate"), tr("Activate and Symmetric difference"), nullptr);
	_ctxt_menu_activate_symmetric_differrence->setToolTip(
	    "Activate this layer and Symmetric difference of the current selection and its content");
	_ctxt_menu->addAction(_ctxt_menu_activate_symmetric_differrence);

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

void PVGuiQt::PVLayerStackView::enterEvent(QEnterEvent*)
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
		Squey::PVLayer& layer = ls_model()->lib_layer_stack().get_layer_n(model_index);
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

Squey::PVLayer& PVGuiQt::PVLayerStackView::get_layer_from_idx(int model_idx)
{
	QVariant var =
	    ls_model()->data(ls_model()->index(model_idx, 0), PVCustomQtRoles::UnderlyingObject);
	return *reinterpret_cast<Squey::PVLayer*>(var.value<void*>());
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
	if (not idx_click.isValid()) {
		return; // don't create the menu if we are not over a layer
	}

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
	if (act == _ctxt_menu_show_this_layer_only) {
		show_this_layer_only(idx_click.row());
	}
	if (act == _ctxt_menu_union) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator|,
		                                        false);
	}
	if (act == _ctxt_menu_difference) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator-,
		                                        false);
	}
	if (act == _ctxt_menu_intersection) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator&,
		                                        false);
	}
	if (act == _ctxt_menu_symmetric_differrence) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator^,
		                                        false);
	}
	if (act == _ctxt_menu_activate_union) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator|,
		                                        true);
	}
	if (act == _ctxt_menu_activate_difference) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator-,
		                                        true);
	}
	if (act == _ctxt_menu_activate_intersection) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator&,
		                                        true);
	}
	if (act == _ctxt_menu_activate_symmetric_differrence) {
		boolean_op_on_selection_with_this_layer(idx_click.row(), &Squey::PVSelection::operator^,
		                                        true);
	}
	if (act == _ctxt_menu_copy_to_clipboard_act) {
		copy_to_clipboard();
	}
}

void PVGuiQt::PVLayerStackView::boolean_op_on_selection_with_this_layer(int layer_idx,
                                                                        const operation_f& f,
                                                                        bool activate)
{
	Squey::PVLayer& layer = get_layer_from_idx(layer_idx);
	Squey::PVView& view = ls_model()->lib_view();

	if (activate)
		layer.set_visible(true);

	Squey::PVSelection selection = (view.get_real_output_selection().*f)(layer.get_selection());

	view.set_selection_view(selection, true);
}

void PVGuiQt::PVLayerStackView::set_current_selection_from_layer(int model_idx)
{
	Squey::PVLayer const& layer = get_layer_from_idx(model_idx);
	ls_model()->lib_view().set_selection_from_layer(layer);
}

void PVGuiQt::PVLayerStackView::export_layer_selection(int model_idx)
{
	Squey::PVLayer const& layer = get_layer_from_idx(model_idx);

	const Squey::PVSelection& sel = layer.get_selection();
	Squey::PVView& view = ls_model()->lib_view();

	PVGuiQt::PVExportSelectionDlg::export_selection(view, sel);
}

void PVGuiQt::PVLayerStackView::reset_layer_colors(int layer_idx)
{
	ls_model()->reset_layer_colors(layer_idx);
}

void PVGuiQt::PVLayerStackView::show_this_layer_only(int layer_idx)
{
	ls_model()->show_this_layer_only(layer_idx);
}
