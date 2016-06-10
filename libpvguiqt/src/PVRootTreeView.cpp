/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

#include <pvguiqt/PVQMapped.h>
#include <pvguiqt/PVQPlotted.h>
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
	_act_new_plotted = new QAction(tr("Create new plotted..."), this);
	_act_new_mapped = new QAction(tr("Create new mapped..."), this);
	_act_edit_mapping = new QAction(tr("Edit mapping..."), this);
	_act_edit_plotting = new QAction(tr("Edit plotting..."), this);

	connect(_act_new_view, SIGNAL(triggered()), this, SLOT(create_new_view()));
	connect(_act_edit_plotting, SIGNAL(triggered()), this, SLOT(edit_plotting()));
	connect(_act_edit_mapping, SIGNAL(triggered()), this, SLOT(edit_mapping()));
}

void PVGuiQt::PVRootTreeView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);

	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		return;
	}

	PVCore::PVDataTreeObject* obj = (PVCore::PVDataTreeObject*)idx_click.internalPointer();
	Inendi::PVView* view = dynamic_cast<Inendi::PVView*>(obj);
	if (!view) {
		return;
	}

	// Double click on a view set this view as the current view of the parent
	// source
	Inendi::PVRoot_sp root_sp(view->get_parent<Inendi::PVRoot>().shared_from_this());

	// Call select_view throught the Hive :)
	PVHive::call<FUNC(Inendi::PVRoot::select_view)>(root_sp, *view);

	// TODO : emit datachanged

	event->accept();
}

void PVGuiQt::PVRootTreeView::contextMenuEvent(QContextMenuEvent* event)
{
	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		return;
	}

	PVCore::PVDataTreeObject* obj = (PVCore::PVDataTreeObject*)idx_click.internalPointer();

	Inendi::PVPlotted* plotted = dynamic_cast<Inendi::PVPlotted*>(obj);
	if (plotted) {
		QMenu* ctxt_menu = new QMenu(this);
		ctxt_menu->addAction(_act_new_view);
		ctxt_menu->addAction(_act_edit_plotting);
		ctxt_menu->popup(QCursor::pos());
		return;
	}

	Inendi::PVMapped* mapped = dynamic_cast<Inendi::PVMapped*>(obj);
	if (mapped) {
		QMenu* ctxt_menu = new QMenu(this);
		ctxt_menu->addAction(_act_edit_mapping);
		ctxt_menu->popup(QCursor::pos());
		return;
	}
}

/*****************************************************************************
 * PVGuiQt::PVRootTreeView::enterEvent
 *****************************************************************************/

void PVGuiQt::PVRootTreeView::enterEvent(QEvent*)
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
	Inendi::PVPlotted* plotted = get_selected_obj_as<Inendi::PVPlotted>();
	if (plotted) {
		plotted->emplace_add_child().process_parent_plotted();
	}
}

void PVGuiQt::PVRootTreeView::edit_mapping()
{
	Inendi::PVMapped* mapped = get_selected_obj_as<Inendi::PVMapped>();
	if (mapped) {
		PVQMapped::edit_mapped(*mapped, this);
	}
}

void PVGuiQt::PVRootTreeView::edit_plotting()
{
	Inendi::PVPlotted* plotted = get_selected_obj_as<Inendi::PVPlotted>();
	if (plotted) {
		PVQPlotted::edit_plotted(*plotted, this);
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
