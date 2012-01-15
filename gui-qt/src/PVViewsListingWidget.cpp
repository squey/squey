//! \file PVViewsLIstingWidget.cpp
//! $Id: PVViewsLIstingWidget.cpp 2501 2011-04-25 14:56:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2012


#include <PVTabSplitter.h>
#include <PVViewsModel.h>
#include <PVViewsListingView.h>
#include <PVViewsListingWidget.h>

#include <QVBoxLayout>

/******************************************************************************
 *
 * PVInspector::PVViewsListingWidget::PVViewsListingWidget
 *
 *****************************************************************************/
PVInspector::PVViewsListingWidget::PVViewsListingWidget(PVTabSplitter* tab):
	QWidget(tab),
	_tab_parent(tab)
{
	_model = new PVViewsModel(*tab->get_lib_src(), this);
	_tree = new PVViewsListingView(_model, tab);

	connect(_tree, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	_tree->setContextMenuPolicy(Qt::CustomContextMenu);

	// SIZE STUFF
	// WARNING: nothing should be set here. It is up to the PVViewsListingView to fix a minimal size, etc.

	// LAYOUT STUFF
	// We need a Layout for that Widget
	QVBoxLayout* layout = new QVBoxLayout();
	// We fix the margins for that Layout
	layout->setContentsMargins(0,0,0,0);

	// We fill the Layout with the PVViewsListingView
	layout->addWidget(_tree);
	// We set the Layout in the Widget
	setLayout(layout);

	// FOCUS POLICY
	setFocusPolicy(Qt::StrongFocus);
}



/******************************************************************************
 *
 * PVInspector::PVViewsListingWidget::force_refresh
 *
 *****************************************************************************/
void PVInspector::PVViewsListingWidget::force_refresh()
{
	_model->force_refresh();
}



/******************************************************************************
 *
 * PVInspector::PVViewsListingWidget::show_ctxt_menu
 *
 *****************************************************************************/
void PVInspector::PVViewsListingWidget::show_ctxt_menu(const QPoint& pt)
{
	QModelIndex idx_click = _tree->indexAt(pt);
	PVViewsModel::PVIndexNode node_obj((Picviz::PVMapped*) NULL);
	bool enable_new_plotted = false;
	bool enable_edit = false;
	bool enable_sel = false;
	if (idx_click.isValid()) {
		node_obj = _model->get_object(idx_click);
		enable_new_plotted = node_obj.is_mapped();
		enable_sel = node_obj.is_plotted();
		enable_edit = true;
	}

	// Create menu
	QMenu* ctxt_menu = new QMenu();
	QAction* sel_view_action = new QAction(tr("Select as current view"), NULL);
	sel_view_action->setEnabled(enable_sel);
	if (enable_sel) {
		QFont font;
		font.setBold(true);
		sel_view_action->setFont(font);
	}
	QAction* new_mapped_action = new QAction(tr("Create new mapped..."), NULL);
	QAction* new_plotted_action = new QAction(tr("Create new plotted from this mapped..."), NULL);
	new_plotted_action->setEnabled(enable_new_plotted);
	QAction* edit_action = new QAction(tr("Edit..."), NULL);
	edit_action->setEnabled(enable_edit);
	ctxt_menu->addAction(sel_view_action);
	ctxt_menu->addAction(new_mapped_action);
	ctxt_menu->addAction(new_plotted_action);
	ctxt_menu->addAction(edit_action);

	// Show it
	QAction* sel_action = ctxt_menu->exec(QCursor::pos());
	if (sel_action == sel_view_action) {
		assert(node_obj.is_plotted());
		Picviz::PVPlotted* plotted = node_obj.as_plotted();
		_tab_parent->select_plotted(plotted);
		_model->emitDataChanged(idx_click);
	}
	else
	if (sel_action == new_mapped_action) {
		_tab_parent->create_new_mapped();
	}
	else
	if (sel_action == new_plotted_action)
	{
		assert(node_obj.is_mapped());
		Picviz::PVMapped* mapped_parent = node_obj.as_mapped();
		_tab_parent->create_new_plotted(mapped_parent);
		_tree->expand(idx_click);
	}
	else
	if (sel_action == edit_action) {
		Picviz::PVMapped* mapped = NULL;
		Picviz::PVPlotted* plotted = NULL;
		if (node_obj.is_mapped()) {
			mapped = node_obj.as_mapped();
			_tab_parent->edit_mapped(mapped);
		}
		else {
			plotted = node_obj.as_plotted();
			_tab_parent->edit_plotted(plotted);
		}
		_model->emitDataChanged(idx_click);
	}
}
