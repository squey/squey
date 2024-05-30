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

#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVMapped.h>
#include <squey/PVScaled.h>
#include <squey/PVView.h>
#include <squey/widgets/PVMappingScalingEditDialog.h>

#include <pvguiqt/PVQMapped.h>
#include <pvguiqt/PVQScaled.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <QMenu>
#include <QMouseEvent>
#include <QAbstractItemModel>

PVGuiQt::PVRootTreeView::PVRootTreeView(QAbstractItemModel* model, QWidget* parent)
    : QTreeView(parent)
{
	// Sizing
	setMinimumSize(100, 0);
	setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

	setHeaderHidden(true);
	setModel(model);
	setAllColumnsShowFocus(true);

	// Context menu
	setContextMenuPolicy(Qt::DefaultContextMenu);

	// Actions
	_act_new_view = new QAction(tr("Create new view"), this);
	_act_new_scaled = new QAction(tr("Create new scaled..."), this);
	_act_new_mapped = new QAction(tr("Create new mapped..."), this);
	_act_edit_mapping = new QAction(tr("Edit mapping..."), this);
	_act_edit_scaling = new QAction(tr("Edit scaling..."), this);

	connect(_act_new_view, &QAction::triggered, this, &PVRootTreeView::create_new_view);
	connect(_act_edit_scaling, &QAction::triggered, this, &PVRootTreeView::edit_scaling);
	connect(_act_edit_mapping, &QAction::triggered, this, &PVRootTreeView::edit_mapping);
}

void PVGuiQt::PVRootTreeView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);

	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		return;
	}

	auto* obj = (PVCore::PVDataTreeObject*)idx_click.internalPointer();
	auto* view = dynamic_cast<Squey::PVView*>(obj);
	if (!view) {
		return;
	}

	// Double click on a view set this view as the current view of the parent
	// source

	view->get_parent<Squey::PVRoot>().select_view(*view);

	// TODO : Q_EMIT datachanged

	event->accept();
}

void PVGuiQt::PVRootTreeView::contextMenuEvent(QContextMenuEvent* event)
{
	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		return;
	}

	auto* obj = (PVCore::PVDataTreeObject*)idx_click.internalPointer();

	auto* scaled = dynamic_cast<Squey::PVScaled*>(obj);
	if (scaled) {
		auto* ctxt_menu = new QMenu(this);
		ctxt_menu->addAction(_act_new_view);
		ctxt_menu->addAction(_act_edit_scaling);
		ctxt_menu->popup(QCursor::pos());
		return;
	}

	auto* mapped = dynamic_cast<Squey::PVMapped*>(obj);
	if (mapped) {
		auto* ctxt_menu = new QMenu(this);
		ctxt_menu->addAction(_act_edit_mapping);
		ctxt_menu->popup(QCursor::pos());
		return;
	}
}

/*****************************************************************************
 * PVGuiQt::PVRootTreeView::enterEvent
 *****************************************************************************/

void PVGuiQt::PVRootTreeView::enterEvent(QEnterEvent*)
{
	setFocus(Qt::MouseFocusReason);
}

/*****************************************************************************
 * PVGuiQt::PVRootTreeView::leaveEvent
 *****************************************************************************/

void PVGuiQt::PVRootTreeView::leaveEvent(QEvent*)
{
	clearFocus();
}

void PVGuiQt::PVRootTreeView::create_new_view()
{
	auto* scaled = get_selected_obj_as<Squey::PVScaled>();
	if (scaled) {
		scaled->emplace_add_child();
	}
}

void PVGuiQt::PVRootTreeView::edit_mapping()
{
	auto* mapped = get_selected_obj_as<Squey::PVMapped>();
	if (mapped) {
		PVQMapped::edit_mapped(*mapped, this);
	}
}

void PVGuiQt::PVRootTreeView::edit_scaling()
{
	auto* scaled = get_selected_obj_as<Squey::PVScaled>();
	if (scaled) {
		PVQScaled::edit_scaled(*scaled, this);
	}
}

PVCore::PVDataTreeObject* PVGuiQt::PVRootTreeView::get_selected_obj()
{
	QModelIndexList sel = selectedIndexes();
	if (sel.size() == 0) {
		return nullptr;
	}
	return (PVCore::PVDataTreeObject*)sel.at(0).internalPointer();
}

PVGuiQt::PVRootTreeModel* PVGuiQt::PVRootTreeView::tree_model()
{
	return qobject_cast<PVRootTreeModel*>(model());
}

PVGuiQt::PVRootTreeModel const* PVGuiQt::PVRootTreeView::tree_model() const
{
	return qobject_cast<PVRootTreeModel const*>(model());
}
