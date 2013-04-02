/**
 * \file PVListingView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */


#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVLayerFilter.h>
#include <picviz/PVView.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>

#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVQNraw.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>
#include <pvguiqt/PVToolTipDelegate.h>

#include <pvdisplays/PVDisplaysContainer.h>
#include <pvdisplays/PVDisplaysImpl.h>

#include <QAbstractButton>
#include <QApplication>
#include <QClipboard>
#include <QCursor>
#include <QKeyEvent>
#include <QHeaderView>
#include <QScrollBar>
#include <QSizePolicy>
#include <QStatusBar>
#include <QMenu>
#include <QWheelEvent>

#define TBB_PREVIEW_DETERMINISTIC_REDUCE 1
#include <tbb/parallel_reduce.h>
#include <tbb/task_scheduler_init.h>

namespace PVGuiQt
{

namespace __impl
{

class PVListingViewSelectionExtractor
{
public:
	typedef tbb::blocked_range<int> blocked_range;

	PVListingViewSelectionExtractor(PVGuiQt::PVListingView *lv) : _lv(lv)
	{}

	PVListingViewSelectionExtractor(PVListingViewSelectionExtractor& e, tbb::split) :
		_lv(e._lv)
	{}

	Picviz::PVSelection &get_selection()
	{
		return _sel;
	}

	void operator()(const blocked_range &r)
	{
		const PVGuiQt::PVListingSortFilterProxyModel* list_model = _lv->get_listing_model();
		QItemSelectionModel *sel_model = _lv->selectionModel();

		_sel.select_none();

		_min = INT_MAX;
		_max = 0;
		for (int i = r.begin(); i < r.end(); ++i) {
			QModelIndex index = list_model->index(i, 0, QModelIndex());
			if (sel_model->isSelected(index)) {
				int row_index = list_model->mapToSource(index).row();
				_min = PVCore::min(_min, row_index);
				_max = PVCore::max(_max, row_index);
				_sel.set_bit_fast(row_index);
			}
		}
	}

	void join(PVListingViewSelectionExtractor &rhs)
	{
		_sel.or_range(rhs._sel, rhs._min, rhs._max);
		_min = PVCore::min(_min, rhs._min);
		_max = PVCore::max(_max, rhs._max);
	}

private:
	PVGuiQt::PVListingView *_lv;
	Picviz::PVSelection     _sel;
	int                     _min, _max;
};

}

}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::PVListingView
 *
 *****************************************************************************/
PVGuiQt::PVListingView::PVListingView(Picviz::PVView_sp& view, QWidget* parent):
	QTableView(parent),
	_ctxt_process(nullptr)
{
	PVHive::get().register_observer(view, _obs);
	PVHive::get().register_actor(view, _actor);

	_obs.connect_about_to_be_deleted(this, SLOT(deleteLater()));

	//_ctxt_process = NULL;
	
	// SIZE STUFF
	setMinimumSize(60,40);
	QSizePolicy temp_size_policy = QSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Expanding);
	temp_size_policy.setHorizontalStretch(1);
	setSizePolicy(temp_size_policy);

	// OBJECTNAME STUFF
	setObjectName("PVListingView");
	// We need to name the headers if we want to style them by CSS (without interfering with other headers...
	horizontalHeader()->setObjectName("horizontalHeader_of_PVListingView");
	verticalHeader()->setObjectName("verticalHeader_of_PVListingView");
	horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
	verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");

	setItemDelegate(new PVToolTipDelegate(this));

	// FOCUS POLICY
	setFocusPolicy(Qt::StrongFocus);
	setSelectionMode(QAbstractItemView::ExtendedSelection);
	setSelectionBehavior(QAbstractItemView::SelectRows);

	horizontalHeader()->setStretchLastSection(true);

	// Sorting
	setSortingEnabled(true);
	
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
	_act_set_color = new QAction(tr("Set color"), _ctxt_menu);
	_ctxt_menu->addAction(_act_copy);
	_ctxt_menu->addSeparator();
	_ctxt_menu->addAction(_act_set_color);

	// Horizontal header context menu
	//
	_hhead_ctxt_menu = new QMenu(this);
	_action_col_unique = new QAction(tr("List unique values of this axis..."), this);
	_hhead_ctxt_menu->addAction(_action_col_unique);


	// Context menu for the listing
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	// Context menu of the horizontal header
	connect(horizontalHeader(), SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_hhead_ctxt_menu(const QPoint&)));
	horizontalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);

	// A double click on the vertical header select the line in the lib view
	connect(verticalHeader(), SIGNAL(sectionDoubleClicked(int)), this, SLOT(slotDoubleClickOnVHead(int)));

	// Save horizontal headers width to be persistent across axes combination changes
	connect(horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(columnResized(int, int, int)));
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::update_view_selection_from_listing_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::update_view_selection_from_listing_selection()
{
	/* VARIABLES */
	QModelIndexList selected_items_list;
	int modifiers;
	// Get current lib view for this source

	/* Commit the previous volatile selection */
	_actor.call<FUNC(Picviz::PVView::commit_volatile_in_floating_selection)>();

	/* Modify the state of the state machine according to the modifiers */
	modifiers = (unsigned int) QApplication::keyboardModifiers();
	/* We don't care about a keypad button being pressed */
	modifiers &= ~Qt::KeypadModifier;
	/* Can't use a switch case here as Qt::ShiftModifier and Qt::ControlModifier aren't really
	 * constants */
	if (modifiers == (Qt::ShiftModifier | Qt::ControlModifier)) {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE);
	}
	else
	if (modifiers == Qt::ControlModifier) {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE);
	}
	else
	if (modifiers == Qt::ShiftModifier) {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE);
	}
	else {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	}

	/* We define the volatile_selection using selection in the listing */
	Picviz::PVSelection& vsel = lib_view().get_volatile_selection();
	extract_selection(vsel);

	/* We reprocess the view from the selection */
	_actor.call<FUNC(Picviz::PVView::process_real_output_selection)>();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::mouseDoubleClickEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::mouseDoubleClickEvent(QMouseEvent* event)
{
	// Here is the reference:
	// * if a double click is made on a line, then this line is selected in the table view *and* in the lib view
	if (selectionModel()->hasSelection()) {
		update_view_selection_from_listing_selection();
	}
	else {
		QTableView::mouseDoubleClickEvent(event);
	}
}

void PVGuiQt::PVListingView::resizeEvent(QResizeEvent * event)
{
	QTableView::resizeEvent(event);
	emit resized();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::getSelectedRows
 *
 *****************************************************************************/
QVector<PVRow> PVGuiQt::PVListingView::get_selected_rows()
{
	QModelIndexList selected_rows_list = selectionModel()->selectedRows(0);
	int selected_rows_count = selected_rows_list.count();
	QVector<PVRow> selected_rows_vector;
	selected_rows_vector.reserve(selected_rows_count);
	PVListingSortFilterProxyModel* myModel = get_listing_model();

	for (int i=0; i<selected_rows_count; ++i)
	{
		int row_index = myModel->mapToSource(selected_rows_list.at(i)).row();
		selected_rows_vector.append(row_index);
	}

	return selected_rows_vector;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::extract_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::extract_selection(Picviz::PVSelection &sel)
{
	typedef __impl::PVListingViewSelectionExtractor tvse_t;

	int thread_num = PVCore::PVHardwareConcurrency::get_physical_core_number();
	tbb::task_scheduler_init init(thread_num);
	int count = model()->rowCount(QModelIndex());
	tvse_t tvse(this);
	BENCH_START(sel_create);
	tbb::parallel_deterministic_reduce(tvse_t::blocked_range(0, count,
	                                                         count / thread_num),
	                                   tvse);
	BENCH_END(sel_create, "extract_selection", 1, 1, 1, 1);
	sel = tvse.get_selection();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::selectionChanged
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
#if 0
	bool has_sel = selected.indexes().size() > 0;
	QStatusBar* sb = main_window->statusBar();
	if (has_sel) {
		sb->showMessage(tr("Press Enter to select these lines, Shift+Enter to add these lines, Ctrl+Enter to remove these lines and Shift+Ctrl+Enter to have the intersection between these lines and the current selection."));
	}
	else {
		sb->clearMessage();
	}
	QTableView::selectionChanged(selected, deselected);
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::slotDoubleClickOnVHead
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::slotDoubleClickOnVHead(int /*idHeader*/)
{
	// The double click automatically select the line, so just call our global
	// selection function.
	update_view_selection_from_listing_selection();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::keyEnterPressed
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::keyPressEvent(QKeyEvent* event)
{
	switch (event->key()) {
		case Qt::Key_Return:
		case Qt::Key_Enter:
			if (selectionModel()->hasSelection()) {
				update_view_selection_from_listing_selection();
			}
			break;
		case 'a':
		case 'A':
			_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
			lib_view().get_floating_selection().select_none();
			lib_view().get_volatile_selection().select_all();

		default:
			QTableView::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::wheelEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::wheelEvent(QWheelEvent* e)
{
	if (e->modifiers() == Qt::ControlModifier)
	{
		int colIndex = columnAt(e->pos().x());
		int d = e->delta() / 12;
		uint32_t width = std::max(columnWidth(colIndex) + d, horizontalHeader()->minimumSectionSize());
		setColumnWidth(colIndex, width);
		_headers_width[lib_view().get_real_axis_index(colIndex)] = width;
	}
	else {
		QTableView::wheelEvent(e);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::columnResized
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::columnResized(int column, int /*oldWidth*/, int newWidth)
{
	_headers_width[lib_view().get_real_axis_index(column)] = newWidth;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::reset
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::reset()
{
	uint32_t default_width = horizontalHeader()->defaultSectionSize();
	for (int i = 0; i <  horizontalHeader()->count(); i++) {
		uint32_t axis_index = lib_view().get_real_axis_index(i);
		setColumnWidth(i, _headers_width[axis_index] ? _headers_width[axis_index] : default_width);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::show_ctxt_menu
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::show_ctxt_menu(const QPoint& pos)
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

	// Get the real row index
	PVRow row = get_listing_model()->mapToSource(idx_click).row();

	// Set these informations in our object, so that they will be retrieved by the slot connected
	// to the menu's actions.
	_ctxt_v = v;
	_ctxt_row = row;
	_ctxt_col = idx_click.column(); // This is the *combined* axis index

	// Show the menu at the given pos
	QAction* act_sel = _ctxt_menu->exec(QCursor::pos());
	if (act_sel) {
		if (act_sel == _act_copy) {
			process_ctxt_menu_copy();
		}
		else
		if (act_sel == _act_set_color) {
			process_ctxt_menu_set_color();
		}
		else {
			process_ctxt_menu_action(act_sel);
		}
	}
}

void PVGuiQt::PVListingView::show_hhead_ctxt_menu(const QPoint& pos)
{
	PVCol comb_col = horizontalHeader()->logicalIndexAt(pos);
	PVCol col = lib_view().get_original_axis_index(comb_col);

	_hhead_ctxt_menu->clear();
	PVDisplays::PVDisplaysContainer* container = PVDisplays::get().get_parent_container(this);
	if (container) {
		// Add entries to the horizontal header context menu for new widgets creation.
		PVDisplays::get().add_displays_view_axis_menu(*_hhead_ctxt_menu, container, SLOT(create_view_axis_widget()), (Picviz::PVView*) &lib_view(), comb_col);
		_hhead_ctxt_menu->addSeparator();
	}
	_hhead_ctxt_menu->addAction(_action_col_unique);

	QAction* sel = _hhead_ctxt_menu->exec(QCursor::pos());
	if (sel == _action_col_unique) {
		PVQNraw::show_unique_values(lib_view().get_rushnraw_parent(), col, *lib_view().get_selection_visible_listing(), this);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::process_ctxt_menu_copy
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::process_ctxt_menu_copy()
{
	// The value to copy is in _ctxt_v
	QClipboard* cb = QApplication::clipboard();
	cb->setText(_ctxt_v);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::process_ctxt_menu_set_color
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::process_ctxt_menu_set_color()
{
#if 0
	/* We let the user select a color */
	PVColorDialog* pv_ColorDialog = new PVColorDialog(*_parent->get_lib_view(), this);
	connect(pv_ColorDialog, SIGNAL(colorSelected(const QColor&)), this, SLOT(set_color_selected(const QColor&)));

	pv_ColorDialog->show();
	pv_ColorDialog->setFocus(Qt::PopupFocusReason);
	pv_ColorDialog->raise();
	pv_ColorDialog->activateWindow();
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::set_color_selected
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::set_color_selected(const QColor& c)
{
#if 0
	if (!c.isValid()) {
		return;
	}

	QVector<PVRow> selected_rows_vector = get_selected_rows();
	Picviz::PVLayer& layer = view->get_current_layer();
	Picviz::PVLinesProperties& lines_properties = layer.get_lines_properties();

	foreach (PVRow line, selected_rows_vector) {
		lines_properties.line_set_rgba(line, c.red(), c.green(), c.blue(), c.alpha());
	}

	// Reprocess pipeline + refresh view
	view->process_from_layer_stack();
#endif
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::process_ctxt_menu_action
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::process_ctxt_menu_action(QAction* act)
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
	_ctxt_args = lib_view().get_last_args_filter(filter_name);
	PVCore::PVArgumentList custom_args = args_f(_ctxt_row, _ctxt_col, lib_view().get_original_axis_index(_ctxt_col), _ctxt_v);
	PVCore::PVArgumentList_set_common_args_from(_ctxt_args, custom_args);

	// Show the layout filter widget
	Picviz::PVLayerFilter_p fclone = lib_filter->clone<Picviz::PVLayerFilter>();
	assert(fclone);
	if (_ctxt_process) {
		_ctxt_process->deleteLater();
	}

	// Creating the PVLayerFilterProcessWidget will save the current args for this filter.
	// Then we can change them !
	_ctxt_process = new PVGuiQt::PVLayerFilterProcessWidget(&lib_view(), _ctxt_args, fclone);
	_ctxt_process->show();
}

PVGuiQt::PVListingSortFilterProxyModel* PVGuiQt::PVListingView::get_listing_model()
{
	PVListingSortFilterProxyModel* proxy_model = dynamic_cast<PVListingSortFilterProxyModel*>(model());
	assert(proxy_model);
	return proxy_model;
}

void PVGuiQt::PVListingView::refresh_listing_filter()
{
	get_listing_model()->refresh_filter();
}

void PVGuiQt::PVListingView::selectAll()
{
	// AG: this function is called by QTableView when the corner button is pushed.
	// That behaviour can't be changed (it is hardcoded in Qt, see qtableview.cpp:632).
	// One hack (this one) is to check that the sender of this action is a QAbstractButton whose parent is this
	// QTableView. Another hack is to take the first child that is a QAbstractButton of this QTableView
	// and to reconnect it to another slot. Both have drawbacks, we chose this one because it is the quicker to
	// implement.
	
	QObject* s = sender();
	if (dynamic_cast<QAbstractButton*>(s) && s->parent() == this) {
		corner_button_clicked();
	}
	else {
		QTableView::selectAll();
	}
}

void PVGuiQt::PVListingView::corner_button_clicked()
{
	// Reset to default ordering
	get_listing_model()->reset_to_default_ordering_or_reverse();
	sortByColumn(-1, Qt::AscendingOrder);
}
