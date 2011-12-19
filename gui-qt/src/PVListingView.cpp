//! \file PVListingView.cpp
//! $Id: PVListingView.cpp 3253 2011-07-07 07:37:17Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>
#include <QCursor>

#include <pvkernel/core/general.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVListingView.h>
#include <PVListingModel.h>
#include <PVLayerFilterProcessWidget.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVLayerFilter.h>

/******************************************************************************
 *
 * PVInspector::PVListingView::PVListingView
 *
 *****************************************************************************/
PVInspector::PVListingView::PVListingView(PVMainWindow *mw, PVTabSplitter *parent):
	QTableView(parent),
	main_window(mw),
	_parent(parent)
{
	PVLOG_DEBUG("PVInspector::PVListingView::%s\n", __FUNCTION__);

	lib_view = parent->get_lib_view();
	_ctxt_process = NULL;

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::NoFocus);
	setSelectionMode(QAbstractItemView::ExtendedSelection);
	setSelectionBehavior(QAbstractItemView::SelectRows);
	
	// Custom context menu.
	// It is created based on what layer filter plugins tell us.
	_ctxt_menu = new QMenu(this);
	_show_ctxt_menu = false;
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes const& lf = LIB_CLASS(Picviz::PVLayerFilter)::get().get_list();
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes::const_iterator it,itlast;
	for (it = lf.begin(); it != lf.end(); it++) {
		Picviz::PVLayerFilter::hash_menu_function_t const& entries = it.value()->get_menu_entries();
		Picviz::PVLayerFilter::hash_menu_function_t::const_iterator it_ent;
		PVLOG_DEBUG("(listing context-menu) for filter '%s', there are %d entries\n", qPrintable(it.key()), entries.size());
		for (it_ent = entries.begin(); it_ent != entries.end(); it_ent++) {
			PVLOG_DEBUG("(listing context-menu) add action '%s' for filter '%s'\n", qPrintable(it_ent.key()), qPrintable(it.key()));
			_show_ctxt_menu = true;
			QAction* act = new QAction(it_ent.key(), _ctxt_menu);
			act->setData(QVariant(it.key())); // Save the name of the layer filter associated to this action
			_ctxt_menu->addAction(act);
		}
		_ctxt_menu->addSeparator();
	}
	_act_copy = new QAction(tr("Copy this value to the clipboard"), _ctxt_menu);
	_ctxt_menu->addAction(_act_copy);

	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	// Init the double click action
	connect(horizontalHeader(), SIGNAL(sectionDoubleClicked(int)), this, SLOT(slotDoubleClickOnHHead(int)));

	// A double click on the vertical header select the line in the lib view
	connect(verticalHeader(), SIGNAL(sectionDoubleClicked(int)), this, SLOT(slotDoubleClickOnVHead(int)));
}

/******************************************************************************
 *
 * PVInspector::PVListingView::update_view_selection_from_listing_selection
 *
 *****************************************************************************/
void PVInspector::PVListingView::update_view_selection_from_listing_selection()
{
	/* VARIABLES */
	Picviz::PVStateMachine *state_machine;
	int i;
	int number_of_items;
	QModelIndexList selected_items_list;
	PVListingModel *myModel = (PVListingModel *)model();
	unsigned int modifiers;
	// Get current lib view for this source

	/* CODE */
	state_machine = lib_view->state_machine;

	/* Commit the previous volatile selection */
	lib_view->commit_volatile_in_floating_selection();

	/* Modify the state of the state machine according to the modifiers */
	modifiers = (unsigned int) QApplication::keyboardModifiers();
	/* We don't care about a keypad button being pressed */
	modifiers &= ~Qt::KeypadModifier;
	/* Can't use a switch case here as Qt::ShiftModifier and Qt::ControlModifier aren't really
	 * constants */
	if (modifiers == (Qt::ShiftModifier | Qt::ControlModifier)) {
		state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE);
	}
	else
	if (modifiers == Qt::ControlModifier) {	
		state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE);
	}
	else
	if (modifiers == Qt::ShiftModifier) {
		state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE);
	}
	else {
		state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
		lib_view->floating_selection.select_none();
	}

	/* We define the volatile_selection using selection in the listing */
	lib_view->volatile_selection.select_none();
	selected_items_list = selectedIndexes();
	number_of_items = selected_items_list.size();
	for (i=0; i<number_of_items; i++) {
		lib_view->volatile_selection.set_line(myModel->getRealRowIndex(selected_items_list[i].row()), 1);
	}
	
	/* We reprocess the view from the selection */
	lib_view->process_from_selection();
	/* We refresh the PVGLView */
	main_window->update_pvglview(lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
	/* We refresh the listing */
	main_window->current_tab->refresh_listing_with_horizontal_header_Slot();
	main_window->current_tab->update_pv_listing_model_Slot();
	main_window->current_tab->refresh_layer_stack_view_Slot();

	main_window->statusBar()->clearMessage();
}

/******************************************************************************
 *
 * PVInspector::PVListingView::mouseDoubleClickEvent
 *
 *****************************************************************************/
void PVInspector::PVListingView::mouseDoubleClickEvent(QMouseEvent* event)
{
	// Here is the reference:
	// * if a double click is made on a line, then this line is selected in the table view *and* in the lib view
	if (selectedIndexes().size() > 0) {
		update_view_selection_from_listing_selection();
	}
	else {
		QTableView::mouseDoubleClickEvent(event);
	}
}

/******************************************************************************
 *
 * PVInspector::PVListingView::selectionChanged
 *
 *****************************************************************************/
void PVInspector::PVListingView::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
	bool has_sel = selected.indexes().size() > 0;
	QStatusBar* sb = main_window->statusBar();
	if (has_sel) {
		sb->showMessage(tr("Press Enter to select these lines, Shift+Enter to add these lines, Ctrl+Enter to remove these lines and Shift+Ctrl+Enter to have the intersection between these lines and the current selection."));
	}
	else {
		sb->clearMessage();
	}
	QTableView::selectionChanged(selected, deselected);
}

/******************************************************************************
 *
 * PVInspector::PVListingView::slotDoubleClickOnVHead
 *
 *****************************************************************************/
void PVInspector::PVListingView::slotDoubleClickOnVHead(int /*idHeader*/)
{
	// The double click automatically select the line, so just call our global
	// selection function.
	update_view_selection_from_listing_selection();
}

/******************************************************************************
 *
 * PVInspector::PVListingView::slotDoubleClickOnHHead
 *
 *****************************************************************************/
void PVInspector::PVListingView::slotDoubleClickOnHHead(int idHeader) 
{
	assert(model());
	static_cast<PVListingModel *>(model())->sortByColumn(idHeader);
}

/******************************************************************************
 *
 * PVInspector::PVListingView::keyEnterPressed
 *
 *****************************************************************************/
void PVInspector::PVListingView::keyEnterPressed()
{
	if (selectedIndexes().size() > 0) {
		update_view_selection_from_listing_selection();
	}
}

/******************************************************************************
 *
 * PVInspector::PVListingView::show_ctxt_menu
 *
 *****************************************************************************/
void PVInspector::PVListingView::show_ctxt_menu(const QPoint& pos)
{
	if (!_show_ctxt_menu) {
		return;
	}

	QModelIndex idx_click = indexAt(pos);
	if (!idx_click.isValid()) {
		return;
	}

	// Get the string associated (that is, taken from the NRAW)
	QString v = idx_click.data().toString();

	// Get the real axis index
	PVCol col = lib_view->get_real_axis_index(idx_click.column());

	// Get the real row index
	assert(model());
	PVRow row = static_cast<PVListingModel*>(model())->getRealRowIndex(idx_click.row());

	// Set these informations in our object, so that they will be retrieved by the slot connected
	// to the menu's actions.
	_ctxt_v = v;
	_ctxt_row = row;
	_ctxt_col = col;

	// Show the menu at the given pos
	QAction* act_sel = _ctxt_menu->exec(QCursor::pos());
	if (act_sel) {
		if (act_sel == _act_copy) {
			process_ctxt_menu_copy();
		}
		else {
			process_ctxt_menu_action(act_sel);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVListingView::process_ctxt_menu_copy
 *
 *****************************************************************************/
void PVInspector::PVListingView::process_ctxt_menu_copy()
{
	// The value to copy is in _ctxt_v
	QClipboard* cb = QApplication::clipboard();
	cb->setText(_ctxt_v);
}

/******************************************************************************
 *
 * PVInspector::PVListingView::process_ctxt_menu_action
 *
 *****************************************************************************/
void PVInspector::PVListingView::process_ctxt_menu_action(QAction* act)
{
	assert(act);
	// Get the filter associated with that menu entry
	QString filter_name = act->data().toString();
	Picviz::PVLayerFilter_p lib_filter = LIB_CLASS(Picviz::PVLayerFilter)::get().get_class_by_name(filter_name);
	if (!lib_filter) {
		PVLOG_ERROR("(listing context-menu) filter '%s' does not exist !\n", qPrintable(filter_name));
		return;
	}

	Picviz::PVLayerFilter::hash_menu_function_t entries = lib_filter->get_menu_entries();
	QString act_name = act->text();
	if (entries.find(act_name) == entries.end()) {
		PVLOG_ERROR("(listing context-menu) unable to find action '%s' in filter '%s'.\n", qPrintable(act_name), qPrintable(filter_name));
		return;
	}
	Picviz::PVLayerFilter::ctxt_menu_f args_f = entries[act_name];

	// Get the arguments
	_ctxt_args = args_f(_ctxt_row, _ctxt_col, _ctxt_v);

	// Show the layout filter widget
	Picviz::PVLayerFilter_p fclone = lib_filter->clone<Picviz::PVLayerFilter>();
	if (_ctxt_process) {
		_ctxt_process->deleteLater();
	}
	_ctxt_process = new PVLayerFilterProcessWidget(main_window->current_tab, _ctxt_args, fclone);
	_ctxt_process->show();
}

/******************************************************************************
 *
 * PVInspector::PVListingView::update_view
 *
 *****************************************************************************/
void PVInspector::PVListingView::update_view()
{
	lib_view = _parent->get_lib_view();
	resizeColumnToContents(2);
}
