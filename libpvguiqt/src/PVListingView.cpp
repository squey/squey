/**
 * \file PVListingView.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */


#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVColorDialog.h>

#include <picviz/PVLayerFilter.h>
#include <picviz/PVView.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>

#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVQNraw.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>

#include <pvdisplays/PVDisplaysContainer.h>
#include <pvdisplays/PVDisplaysImpl.h>

#include <QApplication>
#include <QClipboard>
#include <QCursor>
#include <QHeaderView>
#include <QInputDialog>
#include <QKeyEvent>
#include <QMenu>
#include <QPainter>
#include <QSizePolicy>
#include <QWheelEvent>
#include <QToolTip>
#include <QScrollBar>

#define TBB_PREVIEW_DETERMINISTIC_REDUCE 1
#include <tbb/task_scheduler_init.h>

/******************************************************************************
 *
 * PVGuiQt::PVListingView
 *
 *****************************************************************************/

PVGuiQt::PVListingView::PVListingView(Picviz::PVView_sp& view, QWidget* parent):
	PVTableView(parent),
	_ctxt_menu(this),
	_hhead_ctxt_menu(this),
	_vhead_ctxt_menu(this),
	_help_widget(this),
	_ctxt_process(nullptr),
	_headers_width(view->get_original_axes_count(), horizontalHeader()->defaultSectionSize())
{
	PVHive::get().register_actor(view, _actor);

	// When removing the observer, also remove the GUI
	// FIXME : It loops...
	_obs.connect_about_to_be_deleted(this, SLOT(deleteLater()));
	PVHive::get().register_observer(view, _obs);

	/// Source events
	Picviz::PVSource_sp src_sp = view->get_parent<Picviz::PVSource>()->shared_from_this();
	// Register source for axes hovering events
	PVHive::get().register_observer(src_sp, [=](Picviz::PVSource& source) { return &source.axis_hovered(); }, _axis_hover_obs);
	_axis_hover_obs.connect_refresh(this, SLOT(highlight_column(PVHive::PVObserverBase*)));

	 connect(verticalScrollBar(), &QScrollBar::valueChanged, this, &PVGuiQt::PVListingView::slider_move_to);
	 connect(verticalScrollBar(), &QScrollBar::actionTriggered, this, &PVGuiQt::PVListingView::scrollclick);
	 connect(verticalScrollBar(), &QScrollBar::rangeChanged, this,
			 (void (PVGuiQt::PVListingView::*)(int, int)) &PVGuiQt::PVListingView::new_range);
	 connect(verticalScrollBar(), &QScrollBar::sliderReleased, this, &PVGuiQt::PVListingView::clip_slider);

	// SIZE STUFF
	setMinimumSize(60,40);
	QSizePolicy temp_size_policy = QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Expanding);
	temp_size_policy.setHorizontalStretch(1);
	setSizePolicy(temp_size_policy);

	// OBJECTNAME STUFF useful for css
	setObjectName("PVListingView");
	horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
	verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");

	// FOCUS POLICY
	// Used for WheelEvent handling as it is called only on focused widget
	setFocusPolicy(Qt::StrongFocus);

	// Sorting disable as we do it ourself
	setSortingEnabled(false);

	// Custom context menu.
	// It is created based on what layer filter plugins tell us.
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes const& lf = LIB_CLASS(Picviz::PVLayerFilter)::get().get_list();
	using const_layer_iterator = LIB_CLASS(Picviz::PVLayerFilter)::list_classes::const_iterator;
	// Iterator over all layer filter plugins
	// We can't use autoFor here because iterate over a QMap return only value
	// FIXME : Here we search for all layer filter plugins names and save only
	// the name in the action for a later look up for real action. It is bad!!
	// We should add a connect for every plugins action and save a context which
	// will be updated before sending the signal so that we can process plugins
	// widgets
	for (const_layer_iterator it = lf.begin(); it != lf.end(); it++) {
		Picviz::PVLayerFilter::hash_menu_function_t const& entries = it.value()->get_menu_entries();
		using const_layer_menu_iterator = Picviz::PVLayerFilter::hash_menu_function_t::const_iterator;
		PVLOG_DEBUG("(listing context-menu) for filter '%s', there are %d entries\n", qPrintable(it.key()), entries.size());
		for (const_layer_menu_iterator it_ent = entries.begin(); it_ent != entries.end(); it_ent++) {
			PVLOG_DEBUG("(listing context-menu) add action '%s' for filter '%s'\n", qPrintable(it_ent.key()), qPrintable(it.key()));
			QAction* act = new QAction(it_ent.key(), &_ctxt_menu);
			act->setData(QVariant(it.key())); // Save the name of the layer filter associated to this action
			_ctxt_menu.addAction(act);
		}
		_ctxt_menu.addSeparator();
	}
	_act_copy = new QAction(tr("Copy this value to the clipboard"), &_ctxt_menu);
	_act_set_color = new QAction(tr("Set color"), &_ctxt_menu);
	_ctxt_menu.addAction(_act_copy);
	_ctxt_menu.addSeparator();
	_ctxt_menu.addAction(_act_set_color);

	// Horizontal header context menu
	// Actions are added later as there depend on clicked column but we have to
	// add them here to to avoid memory leak if the widgets is remove before any
	// header context creation
	_action_col_unique = new QAction(tr("Distinct values"), this);
	_action_col_unique->setIcon(QIcon(":/fileslist_black"));
	_hhead_ctxt_menu.addAction(_action_col_unique);

	_menu_col_count_by = new QMenu(tr("Count by"), this);
	_menu_col_count_by->setIcon(QIcon(":/count_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_count_by);

	_menu_col_sum_by = new QMenu(tr("Sum by"), this);
	_menu_col_sum_by->setIcon(QIcon(":/sum_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_sum_by);

	_menu_col_min_by = new QMenu(tr("Min by"), this);
	_menu_col_min_by->setIcon(QIcon(":/min_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_min_by);

	_menu_col_max_by = new QMenu(tr("Max by"), this);
	_menu_col_max_by->setIcon(QIcon(":/max_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_max_by);

	_menu_col_avg_by = new QMenu(tr("Average by"), this);
	_menu_col_avg_by->setIcon(QIcon(":/avg_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_avg_by);

	_action_col_sort = new QAction(tr("Sort this axis"), this);
	_action_col_sort->setIcon(QIcon(":/sort_desc"));
	_hhead_ctxt_menu.addAction(_action_col_sort);

	// Context menu for the listing
	connect(this, &PVGuiQt::PVListingView::customContextMenuRequested, this, &PVGuiQt::PVListingView::show_ctxt_menu);
	setContextMenuPolicy(Qt::CustomContextMenu); // Enable context menu signal

	// A double click on the vertical header select the line in the lib view
	connect(verticalHeader(), &QHeaderView::sectionDoubleClicked, this,
			(void (PVGuiQt::PVListingView::*)(int)) &PVGuiQt::PVListingView::slotDoubleClickOnVHead);
	connect(this, &PVGuiQt::PVListingView::doubleClicked, this,
			(void (PVGuiQt::PVListingView::*)(QModelIndex const&)) &PVGuiQt::PVListingView::slotDoubleClickOnVHead);

	// Context menu on vertical header
	_action_copy_row_value = new QAction(tr("Copy line index to clipbard"), this);
	_vhead_ctxt_menu.addAction(_action_copy_row_value);

	verticalHeader()->setSectionsClickable(true);
	verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	verticalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);
	verticalHeader()->setObjectName("verticalHeader_of_PVListingView");
	connect(verticalHeader(), &QHeaderView::customContextMenuRequested,
	        this, &PVGuiQt::PVListingView::show_vhead_ctxt_menu);

	// Text elipsis
	setWordWrap(false);

	// Define help
	_help_widget.hide();

	_help_widget.initTextFromFile("listing view's help",
	                               ":help-style");
	_help_widget.addTextFromFile(":help-selection");
	_help_widget.addTextFromFile(":help-layers");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-lines");
	_help_widget.addTextFromFile(":help-application");

	_help_widget.newTable();
	_help_widget.addTextFromFile(":help-mouse-listing-view");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-shortcuts-listing-view");
	_help_widget.finalizeText();

	// We fix the vertical header size on bold max number of line to avoid
	// resizing on scrolling
	QFont font = verticalHeader()->font();
	font.setBold(true);
	_vhead_max_width = QFontMetrics(font).width(QString().leftJustified(QString::number(PICVIZ_LINES_MAX).size(), '9'));
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::~PVListingView
 *
 *****************************************************************************/
PVGuiQt::PVListingView::~PVListingView()
{
	if(_ctxt_process) {
		delete _ctxt_process;
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::update_view_selection_from_listing_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::update_view_selection_from_listing_selection()
{
	/* Commit the previous volatile selection */
	_actor.call<FUNC(Picviz::PVView::commit_volatile_in_floating_selection)>();

	/* Modify the state of the state machine according to the modifiers */
	Qt::KeyboardModifiers modifiers = QApplication::keyboardModifiers();

	// Interserct selection on Shift + Ctrl
	// Substract new selection on Ctrl
	// Expand the selection on Shift
	// Replace the old selection without modifiers
	if (modifiers & (Qt::ShiftModifier | Qt::ControlModifier)) {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE);
	}
	else
	if (modifiers & Qt::ControlModifier) {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE);
	}
	else
	if (modifiers & Qt::ShiftModifier) {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE);
	}
	else {
		_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	}

	/* We define the volatile_selection using selection in the listing */
	extract_selection();

	/* We reprocess the view from the selection */
	_actor.call<FUNC(Picviz::PVView::process_real_output_selection)>();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::resizeEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::resizeEvent(QResizeEvent * event)
{
	PVTableView::resizeEvent(event);
	emit resized();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::enterEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::enterEvent(QEvent*)
{
	setFocus(Qt::MouseFocusReason);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::leaveEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::leaveEvent(QEvent*)
{
	clearFocus();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::extract_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::extract_selection()
{
	// Validate the current selection and reset the local one
	Picviz::PVSelection& sel = lib_view().get_volatile_selection();
	std::swap(sel, listing_model()->current_selection());
	listing_model()->reset_selection();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::slotDoubleClickOnVHead
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::slotDoubleClickOnVHead(QModelIndex const&)
{
	// The double click automatically select the line, so just call our global
	// selection function.
	update_view_selection_from_listing_selection();
}

void PVGuiQt::PVListingView::slotDoubleClickOnVHead(int /*idHeader*/)
{
	// The double click automatically select the line, so just call our global
	// selection function.
	update_view_selection_from_listing_selection();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::keyPressEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::keyPressEvent(QKeyEvent* event)
{
	if(PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (help_widget()->isHidden()) {
			help_widget()->popup(viewport(),
			                     PVWidgets::PVTextPopupWidget::AlignCenter,
			                     PVWidgets::PVTextPopupWidget::ExpandAll);
		}
		return;
	}

	switch (event->key()) {
		case Qt::Key_Return:
		case Qt::Key_Enter:
			if (not listing_model()->current_selection().is_empty()) {
				update_view_selection_from_listing_selection();
			}
			break;
		case Qt::Key_G:
			goto_line();
			break;

		// Bind document displacement key
		case Qt::Key_PageUp:
			scrollclick(QAbstractSlider::SliderPageStepSub);
			break;
		case Qt::Key_PageDown:
			scrollclick(QAbstractSlider::SliderPageStepAdd);
			break;
		case Qt::Key_Up:
			scrollclick(QAbstractSlider::SliderSingleStepSub);
			break;
		case Qt::Key_Down:
			scrollclick(QAbstractSlider::SliderSingleStepAdd);
			break;
		case Qt::Key_Home:
			scrollclick(QAbstractSlider::SliderToMinimum);
			break;
		case Qt::Key_End:
			scrollclick(QAbstractSlider::SliderToMaximum);
			break;
		case Qt::Key_Right:
			horizontalScrollBar()->triggerAction(QAbstractSlider::SliderSingleStepAdd);
			break;
		case Qt::Key_Left:
			horizontalScrollBar()->triggerAction(QAbstractSlider::SliderSingleStepSub);
			break;
		default:
			PVTableView::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::mousePressEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::mousePressEvent(QMouseEvent * event)
{
	if(event->button() == Qt::LeftButton) {
		Qt::KeyboardModifiers mod = event->modifiers();
		// Shift and Ctrl continue the selection and don't reset it
		if(not (mod & (Qt::ControlModifier | Qt::ShiftModifier))) {
			listing_model()->current_selection().select_none();
		}

		int clc_row = rowAt(event->y());
		if(clc_row < 0) {
			// No row under the mouse.
			return; 
		}

		if(mod & Qt::ShiftModifier) {
			// Shift modifier complete the selection between clicks
			listing_model()->end_selection(clc_row);
		} else {
			// Start the selection
			listing_model()->start_selection(clc_row);
		}

		// Move below if we click on the half shown row
		int row_pos = rowViewportPosition(clc_row);
		if((row_pos + rowHeight(clc_row) + horizontalHeader()->height()) > (height() + 1)) {
			move_by(1);
		}

		viewport()->update(); // To show the selection
		PVTableView::mousePressEvent(event);
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
		// delta is wheel movement in degree. QtWheelEvent doc give this formule
		// to convert it to "wheel step"
		// http://doc.qt.io/qt-5/qwheelevent.html
		// Scroll 3 line by wheel step on listing
		move_by(- e->delta() / 8 / 15 * 3);
	}
	e->accept(); // I am the one who handle event
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::mouseReleaseEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::mouseReleaseEvent(QMouseEvent * event)
{
	// Mouse release commit the current selection
	listing_model()->commit_selection();
	viewport()->update();
	event->accept();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::mouseMoveEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::mouseMoveEvent(QMouseEvent * event)
{
	int pos = event->y();
	// Scroll up while the clicked mouse is above the listing
	while(pos < 0) {
		move_by(-1);
		listing_model()->end_selection(rowAt(0));
		pos += rowHeight(rowAt(0));
		if(listing_model()->current_page() == 0 and listing_model()->pos_in_page() == 0) {
			// We reach the top of the listing, stop scrolling upper
			return;
		}
	}

	int clc_row = rowAt(pos);
	if(clc_row < 0) {
		// We are max up and we keep moving upper
		return;
	}

	// Update selection
	listing_model()->end_selection(clc_row);

	// We are in the last partially shown cell, move below
	int row_pos = rowViewportPosition(clc_row);
	if((row_pos + rowHeight(clc_row) + horizontalHeader()->height()) > (height() + 1)) {
		move_by(1);
	}

	viewport()->update(); // Show selection modification
	event->accept();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::setModel
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::setModel(QAbstractItemModel * model)
{
	PVTableView::setModel(model);
	connect(model, &QAbstractItemModel::layoutChanged, this,
			(void (PVGuiQt::PVListingView::*)()) &PVGuiQt::PVListingView::new_range);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::columnResized
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::columnResized(int column, int oldWidth, int newWidth)
{
	PVTableView::columnResized(column, oldWidth, newWidth);
	_headers_width[lib_view().get_real_axis_index(column)] = newWidth;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::reset
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::reset()
{
	// Resize header_width with default value if it is greater
	_headers_width.resize(horizontalHeader()->count(), horizontalHeader()->defaultSectionSize());

	for (int i = 0; i <  horizontalHeader()->count(); i++) {
		uint32_t axis_index = lib_view().get_real_axis_index(i);
		setColumnWidth(i, _headers_width[axis_index]);
	}

	verticalHeader()->setFixedWidth(_vhead_max_width);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::show_ctxt_menu
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::show_ctxt_menu(const QPoint& pos)
{
	QModelIndex idx_click = indexAt(pos);
	if (!idx_click.isValid()) {
		return;
	}

	// Set these informations in our object, so that they will be retrieved by the slot connected
	// to the menu's actions.
	_ctxt_row = listing_model()->rowIndex(idx_click);
	_ctxt_col = idx_click.column(); // This is the *combined* axis index
	_ctxt_v = lib_view().get_parent<Picviz::PVSource>()->get_value(_ctxt_row, lib_view().get_original_axis_index(_ctxt_col));

	// Show the menu at the given pos
	QAction* act_sel = _ctxt_menu.exec(QCursor::pos());

	if (act_sel == _act_copy) {
		process_ctxt_menu_copy();
	} else if (act_sel == _act_set_color) {
		process_ctxt_menu_set_color();
	} else if(act_sel) {
		// process plugins extracted action
		process_ctxt_menu_action(act_sel);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::show_hhead_ctxt_menu
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::show_hhead_ctxt_menu(const QPoint& pos)
{
	PVCol comb_col = horizontalHeader()->logicalIndexAt(pos);
	PVCol col = lib_view().get_original_axis_index(comb_col);

	// Disable hover picture
	section_hovered_enter(col, false);

	// Create a new horizontal header context as it depend on the clicked column
	_hhead_ctxt_menu.clear();

	// Add view creation based on an axis.
	PVDisplays::PVDisplaysContainer* container = PVDisplays::get().get_parent_container(this);
	if (container) {
		// Add entries to the horizontal header context menu for new widgets creation.
		PVDisplays::get().add_displays_view_axis_menu(_hhead_ctxt_menu, container, SLOT(create_view_axis_widget()), (Picviz::PVView*) &lib_view(), comb_col);

		// Do not show view which need the next axis for the last axis.
		if (!lib_view().is_last_axis(comb_col)) {
			PVDisplays::get().add_displays_view_zone_menu(_hhead_ctxt_menu, container, SLOT(create_view_zone_widget()), (Picviz::PVView*) &lib_view(), comb_col);
		}
		_hhead_ctxt_menu.addSeparator();
	}
	_hhead_ctxt_menu.addAction(_action_col_unique);
	_menu_col_count_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_count_by);
	_menu_col_sum_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_sum_by);
	_menu_col_min_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_min_by);
	_menu_col_max_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_max_by);
	_menu_col_avg_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_avg_by);

	const QStringList axes = lib_view().get_axes_names_list();
	for (int i = 0; i < axes.size(); i++) {
		if (i != comb_col) {
			QAction* action_col_count_by = new QAction(axes[i], _menu_col_count_by);
			action_col_count_by->setData(QVariant(i));
			_menu_col_count_by->addAction(action_col_count_by);

			QAction* action_col_sum_by = new QAction(axes[i], _menu_col_sum_by);
			action_col_sum_by->setData(QVariant(i));
			_menu_col_sum_by->addAction(action_col_sum_by);

			QAction* action_col_min_by = new QAction(axes[i], _menu_col_min_by);
			action_col_min_by->setData(QVariant(i));
			_menu_col_min_by->addAction(action_col_min_by);

			QAction* action_col_max_by = new QAction(axes[i], _menu_col_max_by);
			action_col_max_by->setData(QVariant(i));
			_menu_col_max_by->addAction(action_col_max_by);

			QAction* action_col_avg_by = new QAction(axes[i], _menu_col_avg_by);
			action_col_avg_by->setData(QVariant(i));
			_menu_col_avg_by->addAction(action_col_avg_by);
		}
	}
	_hhead_ctxt_menu.addAction(_action_col_sort);

	QAction* sel = _hhead_ctxt_menu.exec(QCursor::pos());

	// Process actions
	if (sel == _action_col_unique) {
		Picviz::PVView_sp view = lib_view().shared_from_this();
		PVQNraw::show_unique_values(view, lib_view().get_rushnraw_parent(), col, *lib_view().get_selection_visible_listing(), this);
	} else if (sel == _action_col_sort) {
		Qt::SortOrder order =  (Qt::SortOrder)!((bool)horizontalHeader()->sortIndicatorOrder());
		sort(col, order);
	} else if(sel) {
		Picviz::PVView_sp view = lib_view().shared_from_this();
		PVCol col2 = lib_view().get_original_axis_index(sel->data().toUInt());
		if (sel->parent() == _menu_col_count_by) {
			PVQNraw::show_count_by(view, lib_view().get_rushnraw_parent(), col, col2, *lib_view().get_selection_visible_listing(), this); // FIXME: AxesCombination
		} else if (sel->parent() == _menu_col_sum_by) {
			PVQNraw::show_sum_by(view, lib_view().get_rushnraw_parent(), col, col2, *lib_view().get_selection_visible_listing(), this); // FIXME: AxesCombination
		} else if (sel->parent() == _menu_col_min_by) {
			PVQNraw::show_min_by(view, lib_view().get_rushnraw_parent(), col, col2, *lib_view().get_selection_visible_listing(), this); // FIXME: AxesCombination
		} else if (sel->parent() == _menu_col_max_by) {
			PVQNraw::show_max_by(view, lib_view().get_rushnraw_parent(), col, col2, *lib_view().get_selection_visible_listing(), this); // FIXME: AxesCombination
		} else if (sel->parent() == _menu_col_avg_by) {
			PVQNraw::show_avg_by(view, lib_view().get_rushnraw_parent(), col, col2, *lib_view().get_selection_visible_listing(), this); // FIXME: AxesCombination
		}
	} else {
		// No selected action
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::show_vhead_ctxt_menu
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::show_vhead_ctxt_menu(const QPoint& pos)
{
	// Use QCursor and not pos as pos is relative to the widgets while QCursor
	// is a screen position.
	QAction* sel = _vhead_ctxt_menu.exec(QCursor::pos());

	if(sel == _action_copy_row_value) {
		int idx = verticalHeader()->logicalIndexAt(pos);
		// FIXME : We should return the full line content
		QApplication::clipboard()->setText(QString::number(listing_model()->rowIndex(idx)));
	} else {
		assert(sel == nullptr && "No other possible vertical menu action");
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
	/* We let the user select a color */
	PVWidgets::PVColorDialog* pv_ColorDialog = new PVWidgets::PVColorDialog(this);
	if (pv_ColorDialog->exec() != QDialog::Accepted) {
		return;
	}
	PVCore::PVHSVColor color = pv_ColorDialog->color();

	set_color_selected(color);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::set_color_selected
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::set_color_selected(const PVCore::PVHSVColor& color)
{
	Picviz::PVLayer& layer = lib_view().get_current_layer();
	Picviz::PVLinesProperties& lines_properties = layer.get_lines_properties();

	// Color every lines in the current selection
	for(PVRow line : listing_model()->shown_lines()) {
		if(listing_model()->current_selection().get_line_fast(line)) {
			lines_properties.line_set_color(line, color);
		}
	}

	_actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::process_ctxt_menu_action
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::process_ctxt_menu_action(QAction* act)
{
	assert(act); // FIXME : Should use a reference
	// FIXME : This should be done another way (see menu creation)
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

	if (custom_args.get_edition_flag()) {
		_ctxt_process->show();
	} else {
		_ctxt_process->save_Slot();
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::listing_model
 *
 *****************************************************************************/
PVGuiQt::PVListingModel* PVGuiQt::PVListingView::listing_model()
{
	return static_cast<PVListingModel*>(model());
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::section_hovered_enter
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::section_hovered_enter(int col, bool entered)
{
	Picviz::PVSource_sp src = lib_view().get_parent<Picviz::PVSource>()->shared_from_this();
	PVHive::call<FUNC(Picviz::PVSource::set_section_hovered)>(src, col, entered);
	highlight_column(entered ? col : -1);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::section_clicked
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::section_clicked(int col)
{
	Picviz::PVSource_sp src = lib_view().get_parent<Picviz::PVSource>()->shared_from_this();
	int x = horizontalHeader()->sectionViewportPosition(col);
	int width = horizontalHeader()->sectionSize(col);
	PVHive::call<FUNC(Picviz::PVSource::set_section_clicked)>(src, col, verticalHeader()->width() + x + width/2);
}


/******************************************************************************
 *
 * PVGuiQt::PVListingView::highlight_column
 *
 *****************************************************************************/

void PVGuiQt::PVListingView::highlight_column(PVHive::PVObserverBase* o)
{
	// Extract column to highlight
	PVHive::PVObserverSignal<int>* real_o = dynamic_cast<PVHive::PVObserverSignal<int>*>(o);
	assert(real_o);
	int* obj = real_o->get_object();
	int col = *obj;

	highlight_column(col);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::highlight_column
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::highlight_column(int col)
{
	// Mark the column for future painting and force update
	_hovered_axis = col;
	viewport()->update();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingView::slider_move_to
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::slider_move_to(int value)
{
	if(value == verticalScrollBar()->maximum()) {
		// Move to the end of the listing
		move_to_end();
	} else {
		// Move to the top of the page
		move_to_page(value);
	}
	viewport()->update();
	verticalHeader()->viewport()->update();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::new_range
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::new_range(int min, int max)
{
	if(model()) {
		listing_model()->update_pages(max - min, verticalScrollBar()->pageStep());
		move_to_page(0);
	}
}

void PVGuiQt::PVListingView::new_range()
{
	if(model()) {
		listing_model()->update_pages(verticalScrollBar()->maximum() - verticalScrollBar()->minimum(),
				verticalScrollBar()->pageStep());
		move_to_page(0);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::clip_slider
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::clip_slider()
{
	if(verticalScrollBar()->value() == 0) {
		move_to_page(0);
	} else if(verticalScrollBar()->value() == verticalScrollBar()->maximum()) {
		move_to_end();
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::scrollclick
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::scrollclick(int action) {
	switch(action) {
		case QAbstractSlider::SliderSingleStepAdd:
			move_by(1);
			break;
		case QAbstractSlider::SliderSingleStepSub:
			move_by(-1);
			break;
		case QAbstractSlider::SliderPageStepAdd:
			move_by(verticalScrollBar()->pageStep());
			break;
		case QAbstractSlider::SliderPageStepSub:
			move_by(-verticalScrollBar()->pageStep());
			break;
		case QAbstractSlider::SliderToMinimum:
			move_to_page(0);
			break;
		case QAbstractSlider::SliderToMaximum:
			move_to_end();
			break;
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::move_by
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::move_by(int row)
{
	listing_model()->move_by(row, verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::move_to_nraw
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::move_to_nraw(PVRow row)
{
	listing_model()->move_to_nraw(row, verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::move_to_row
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::move_to_row(PVRow row)
{
	listing_model()->move_to_row(row, verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::move_to_page
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::move_to_page(int page)
{
	listing_model()->move_to_page(page);
	update_on_move();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::move_to_end
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::move_to_end()
{
	listing_model()->move_to_end(verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::update_on_move
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::update_on_move()
{
	// Save and restore pos_in_range as moving cursor call slider_move_to which
	// set pos_in_page_to 0.
	size_t pos_in_page = listing_model()->pos_in_page();
	verticalScrollBar()->setValue(listing_model()->current_page());
	listing_model()->pos_in_page() = pos_in_page;
	verticalHeader()->viewport()->update();
	viewport()->update();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::paintEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::paintEvent(QPaintEvent* event)
{
	PVTableView::paintEvent(event);

	if (_hovered_axis != -1) {

		//         Grey curve:
		//	 g4	|   |___|   |   |
		//		|  /|   |   |   |
		//		| / |   |   |   |
		//	 g3	|/  |   |   |   |
		//		|   |   |   |   |
		//		|   |   |   |   |
		//	 g2	|   |   |   |  /|
		//		|   |   |   | / |
		//	 g1	|   |   |___|/  |
		//		|   |   |   |   |
		//		0  t1  t2  t3   1

		qreal threshold1 = 0.3;
		qreal threshold2 = 0.5;
		qreal threshold3 = 0.7;
		qreal grey1 = 0.3;
		qreal grey2 = 0.5;
		qreal grey3 = 0.6;
		qreal grey4 = 0.8;

		int border_width = 6;


		int x = horizontalHeader()->sectionViewportPosition(_hovered_axis);
		int w = horizontalHeader()->sectionSize(_hovered_axis);

		QPainter painter(viewport());
		QRectF r(x, 0, w, height());

		QRectF rect1(r.x() - border_width/2 -1, 0, border_width, height());
		QRectF rect2(r.x() + r.width() - border_width/2 -1, 0, border_width, height());

		border_width *= 2;
		QColor color = lib_view().get_axis(_hovered_axis).get_titlecolor().toQColor();

		qreal weighted_value = (((color.redF() * 0.299) + (color.greenF() * 0.587) + (color.blueF() * 0.114)));

		int grey_level;
		if (weighted_value < threshold1) {
			grey_level = 255 * (grey3 + (weighted_value * (grey4 - grey3) / threshold1));
		}
		else {
			if (weighted_value <  threshold2) {
				grey_level = 255 * grey4;
			}
			else {
				if (weighted_value < threshold3) {
					grey_level = 255 * grey1;
				}
				else {
					grey_level = 255 * (grey1 + ((weighted_value - threshold3) * (grey2 - grey1) / (1 - threshold3)));
				}
			}
		}

		QColor bg_color(grey_level, grey_level, grey_level);
		QPixmap texture(border_width, border_width);
		QPainter texture_painter(&texture);
		QBrush b(bg_color);
		texture_painter.setBrush(b);
		texture_painter.setPen(bg_color);

		texture_painter.fillRect(0, 0, border_width, border_width, color);
		QPolygon poly1;
		poly1 << QPoint(0, 0);
		poly1 << QPoint(border_width/2-1, 0);
		poly1 << QPoint(0, border_width/2-1);
		QPolygon poly2;
		poly2 << QPoint(border_width, 0);
		poly2 << QPoint(0, border_width);
		poly2 << QPoint(border_width/2-1, border_width);
		poly2 << QPoint(border_width, border_width/2-1);
		texture_painter.drawPolygon(poly1);
		texture_painter.drawPolygon(poly2);
		texture_painter.end();

		QBrush brush(bg_color, texture);
		painter.setPen(Qt::NoPen);
		painter.fillRect(rect1, brush);
		painter.fillRect(rect2, brush);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::goto_line
 *
 *****************************************************************************/

void PVGuiQt::PVListingView::goto_line()
{
	PVRow nrows = lib_view().get_rushnraw_parent().get_number_rows();
	const Picviz::PVSelection& sel = lib_view().get_real_output_selection();

	bool ok;
	PVRow row = QInputDialog::getInt(this,
	                                 "Go to line", "Select line index",
	                                 0, 0, nrows - 1, 1, &ok);

	if (ok == false) {
		return;
	}

	/* If the event corresponding to the wanted row is not selected, we
	 * first search for the first previous set one, if there is no such
	 * row, we search for the first next set one. If there is no such row,
	 * there is nothing to do.
	 */
	// FIXME : It doesn't really work with sorted values
	if (sel.get_line_fast(row) == false) {
		row = sel.find_previous_set_bit(row - 1, nrows);

		if(row == PVROW_INVALID_VALUE) {
			row = sel.find_next_set_bit(row + 1, nrows);
		}
	}

	if (row != PVROW_INVALID_VALUE) {
		move_to_nraw(row);
	} // nothing to do otherwise
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::sort
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::sort(int col, Qt::SortOrder order)
{
	assert(col >= 0 && col < listing_model()->columnCount());
	PVCore::PVProgressBox* box = new PVCore::PVProgressBox(tr("Sorting..."), this);
	box->set_enable_cancel(true);
	tbb::task_group_context ctxt;
	bool changed = PVCore::PVProgressBox::progress([&]() { listing_model()->sort(col, order, ctxt); }, ctxt, box);
	if(changed) {
		horizontalHeader()->setSortIndicator(col, order);
	}
	horizontalHeader()->setSortIndicatorShown(true);
}

/******************************************************************************
 *
 * PVGuiQt::PVHorizontalHeaderView
 *
 *****************************************************************************/

PVGuiQt::PVHorizontalHeaderView::PVHorizontalHeaderView(Qt::Orientation orientation, PVListingView* parent) : QHeaderView(orientation, parent)
{
	// These two calls are required since they are done on the headers in QTableView::QTableView
	// instead of in QHeaderView::QHeaderView !
	setSectionsClickable(true);
	setHighlightSections(true);

	// Context menu of the horizontal header
	setStretchLastSection(true);
	connect(this, &PVGuiQt::PVHorizontalHeaderView::customContextMenuRequested, parent, &PVListingView::show_hhead_ctxt_menu);
	setContextMenuPolicy(Qt::CustomContextMenu);

	// Save horizontal headers width to be persistent across axes combination changes
	connect(this, &PVGuiQt::PVHorizontalHeaderView::sectionResized, parent, &PVListingView::columnResized);

	// section <-> axis synchronisation
	connect(this, &PVGuiQt::PVHorizontalHeaderView::mouse_hovered_section, parent, &PVListingView::section_hovered_enter);
	connect(this, &PVGuiQt::PVHorizontalHeaderView::sectionClicked, parent, &PVListingView::section_clicked);

	// Force hover events on every theme so that "column -> axis" visual synchronisation always works !
	setAttribute(Qt::WA_Hover);
}

bool PVGuiQt::PVHorizontalHeaderView::event(QEvent* ev)
{
	if (ev->type() == QEvent::HoverLeave || ev->type() == QEvent::Leave) {
		emit mouse_hovered_section(_index, false);
		_index = -1;
	}
	else if (ev->type() == QEvent::HoverMove) { // in eventFilter, this event would have been "QEvent::MouseMove"...
		QHoverEvent* mouse_event = dynamic_cast<QHoverEvent*>(ev);
		int index = logicalIndexAt(mouse_event->pos());
		if(index != _index) {
			if (_index != -1) {
				emit mouse_hovered_section(_index, false);
			}
			emit mouse_hovered_section(index, true);
		}
		_index = index;
	}
	return QHeaderView::event(ev);
}

void PVGuiQt::PVHorizontalHeaderView::paintSection(
	QPainter* painter,
	const QRect& rect,
	int logicalIndex) const
{
	painter->save();
	QHeaderView::paintSection(painter, rect, logicalIndex);
	painter->restore();
}
