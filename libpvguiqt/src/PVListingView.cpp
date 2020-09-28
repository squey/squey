/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVColorDialog.h>
#include <pvkernel/widgets/PVFilterableMenu.h>

#include <inendi/PVLayerFilter.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>
#include <inendi/PVSource.h>

#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVQNraw.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>

#include <pvdisplays/PVDisplaysContainer.h>
#include <pvdisplays/PVDisplayIf.h>

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

#include <boost/thread.hpp>

/******************************************************************************
 *
 * PVGuiQt::PVListingView
 *
 *****************************************************************************/

PVGuiQt::PVListingView::PVListingView(Inendi::PVView& view, QWidget* parent)
    : PVAbstractTableView(parent)
    , _view(view)
    , _ctxt_menu(this)
    , _hhead_ctxt_menu(this)
    , _vhead_ctxt_menu(this)
    , _help_widget(this)
    , _ctxt_process(nullptr)
    , _headers_width(view.get_column_count(), horizontalHeader()->defaultSectionSize())
{

	view._axis_hovered.connect(sigc::mem_fun(this, &PVGuiQt::PVListingView::highlight_column));
	view._axis_clicked.connect(sigc::mem_fun(this, &PVGuiQt::PVListingView::set_section_visible));

	// SIZE STUFF
	setMinimumSize(60, 40);
	QSizePolicy temp_size_policy =
	    QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Expanding);
	temp_size_policy.setHorizontalStretch(1);
	setSizePolicy(temp_size_policy);

	// OBJECTNAME STUFF useful for css
	setObjectName("PVListingView");
	horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
	verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");

	// FOCUS POLICY
	// Used for WheelEvent handling as it is called only on focused widget
	setFocusPolicy(Qt::StrongFocus);

	// Custom context menu.
	// It is created based on what layer filter plugins tell us.
	LIB_CLASS(Inendi::PVLayerFilter)
	::list_classes const& lf = LIB_CLASS(Inendi::PVLayerFilter)::get().get_list();
	using const_layer_iterator = LIB_CLASS(Inendi::PVLayerFilter)::list_classes::const_iterator;
	// Iterator over all layer filter plugins
	// We can't use autoFor here because iterate over a QMap return only value
	// FIXME : Here we search for all layer filter plugins names and save only
	// the name in the action for a later look up for real action. It is bad!!
	// We should add a connect for every plugins action and save a context which
	// will be updated before sending the signal so that we can process plugins
	// widgets
	for (const_layer_iterator it = lf.begin(); it != lf.end(); it++) {
		Inendi::PVLayerFilter::hash_menu_function_t const& entries =
		    it->value()->get_menu_entries();
		using const_layer_menu_iterator =
		    Inendi::PVLayerFilter::hash_menu_function_t::const_iterator;
		PVLOG_DEBUG("(listing context-menu) for filter '%s', there are %d entries\n",
		            qPrintable(it->key()), entries.size());
		for (const_layer_menu_iterator it_ent = entries.begin(); it_ent != entries.end();
		     it_ent++) {
			PVLOG_DEBUG("(listing context-menu) add action '%s' for filter '%s'\n",
			            qPrintable(it_ent->key()), qPrintable(it->key()));
			QAction* act = new QAction(it_ent->key(), &_ctxt_menu);
			act->setData(QVariant(it->key())); // Save the name of the layer filter
			                                   // associated to this action
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

	_menu_col_count_by = new PVWidgets::PVFilterableMenu(tr("Count by"), this);
	_menu_col_count_by->setIcon(QIcon(":/count_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_count_by);

	_menu_col_sum_by = new PVWidgets::PVFilterableMenu(tr("Sum by"), this);
	_menu_col_sum_by->setIcon(QIcon(":/sum_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_sum_by);

	_menu_col_min_by = new PVWidgets::PVFilterableMenu(tr("Min by"), this);
	_menu_col_min_by->setIcon(QIcon(":/min_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_min_by);

	_menu_col_max_by = new PVWidgets::PVFilterableMenu(tr("Max by"), this);
	_menu_col_max_by->setIcon(QIcon(":/max_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_max_by);

	_menu_col_avg_by = new PVWidgets::PVFilterableMenu(tr("Average by"), this);
	_menu_col_avg_by->setIcon(QIcon(":/avg_by"));
	_hhead_ctxt_menu.addMenu(_menu_col_avg_by);

	_action_col_copy = new QAction(tr("Copy column name to clipboad"), this);
	_action_col_copy->setIcon(QIcon(":/edit-paste.png"));
	_hhead_ctxt_menu.addAction(_action_col_copy);

	_action_col_sort = new QAction(tr("Sort this axis"), this);
	_action_col_sort->setIcon(QIcon(":/sort_desc"));
	_hhead_ctxt_menu.addAction(_action_col_sort);

	_hhead_ctxt_menu.addSeparator();
	_menu_add_correlation = new QMenu(tr("Bind this axis with..."), this);
	_menu_add_correlation->setIcon(QIcon(":/bind"));
	_hhead_ctxt_menu.addMenu(_menu_add_correlation);

	// A double click on the vertical header select the line in the lib view
	connect(verticalHeader(), &QHeaderView::sectionDoubleClicked, this,
	        (void (PVGuiQt::PVListingView::*)(int)) &
	            PVGuiQt::PVListingView::slotDoubleClickOnVHead);
	connect(this, &PVGuiQt::PVListingView::doubleClicked, this,
	        (void (PVGuiQt::PVListingView::*)(QModelIndex const&)) &
	            PVGuiQt::PVListingView::slotDoubleClickOnVHead);

	// Context menu on vertical header
	_action_copy_row_value = new QAction(tr("Copy line index to clipbard"), this);
	_vhead_ctxt_menu.addAction(_action_copy_row_value);

	verticalHeader()->setSectionsClickable(true);
	verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	verticalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);
	verticalHeader()->setObjectName("verticalHeader_of_PVListingView");
	connect(verticalHeader(), &QHeaderView::customContextMenuRequested, this,
	        &PVGuiQt::PVListingView::show_vhead_ctxt_menu);

	// enabling QSS for the horizontal headers
	horizontalHeader()->setObjectName("horizontalHeader_of_PVAbstractTableView");

	// Define help
	_help_widget.hide();

	_help_widget.initTextFromFile("listing view's help", ":help-style");
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
	_vhead_max_width = QFontMetrics(font).horizontalAdvance(QString().leftJustified(
	    QString::number(view.get_rushnraw_parent().row_count() + 1).size(), '9'));

	// Handle selection modification signal.
	connect(this, &PVAbstractTableView::validate_selection, this,
	        &PVListingView::update_view_selection_from_listing_selection);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::~PVListingView
 *
 *****************************************************************************/
PVGuiQt::PVListingView::~PVListingView()
{
	if (_ctxt_process) {
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
	/* Modify the state of the state machine according to the modifiers */
	Qt::KeyboardModifiers modifiers = QApplication::keyboardModifiers();

	// Interserct selection on Shift + Ctrl
	// Substract new selection on Ctrl
	// Expand the selection on Shift
	// Replace the old selection without modifiers
	if ((modifiers & Qt::ShiftModifier) and (modifiers & Qt::ControlModifier)) {
		lib_view().set_selection_view(lib_view().get_real_output_selection() &
		                              table_model()->current_selection());
	} else if (modifiers & Qt::ControlModifier) {
		lib_view().set_selection_view(lib_view().get_real_output_selection() -
		                              table_model()->current_selection());
	} else if (modifiers & Qt::ShiftModifier) {
		lib_view().set_selection_view(lib_view().get_real_output_selection() |
		                              table_model()->current_selection());
	} else {
		lib_view().set_selection_view(table_model()->current_selection());
	}

	table_model()->reset_selection();
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
	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (help_widget()->isHidden()) {
			help_widget()->popup(viewport(), PVWidgets::PVTextPopupWidget::AlignCenter,
			                     PVWidgets::PVTextPopupWidget::ExpandAll);
		}
		return;
	}

	switch (event->key()) {
	case Qt::Key_G:
		goto_line();
		break;
	default:
		PVAbstractTableView::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::wheelEvent
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::wheelEvent(QWheelEvent* e)
{
	if (e->modifiers() == Qt::ControlModifier) {
		PVCombCol colIndex(columnAt(e->pos().x()));
		int d = e->angleDelta().y() / 12;
		uint32_t width =
		    std::max(columnWidth(colIndex) + d, horizontalHeader()->minimumSectionSize());
		horizontalHeader()->resizeSection(colIndex, width);
		_headers_width[lib_view().get_nraw_axis_index(colIndex)] = width;
		e->accept(); // I am the one who handle event
	} else {
		PVAbstractTableView::wheelEvent(e);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::columnResized
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::columnResized(int column, int oldWidth, int newWidth)
{
	PVTableView::columnResized(column, oldWidth, newWidth);
	_headers_width[lib_view().get_nraw_axis_index((PVCombCol)column)] = newWidth;
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

	for (PVCombCol i(0); i < horizontalHeader()->count(); i++) {
		PVCol axis_index = lib_view().get_nraw_axis_index(i);
		horizontalHeader()->resizeSection(i, _headers_width[axis_index]);
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

	// Set these informations in our object, so that they will be retrieved by the
	// slot connected
	// to the menu's actions.
	_ctxt_row = listing_model()->rowIndex(idx_click);
	_ctxt_col = (PVCombCol)idx_click.column();
	PVCol col = _view.get_axes_combination().get_nraw_axis(_ctxt_col);

	const Inendi::PVSource& src = _view.get_parent<Inendi::PVSource>();

	QStringList l;
	for (PVRow line : listing_model()->shown_lines()) {
		if (listing_model()->current_selection().get_line_fast(line)) {
			l << QString::fromStdString(src.get_value(line, col));
		}
	}
	_ctxt_v = l.join("\n");

	// Show the menu at the given pos
	QAction* act_sel = _ctxt_menu.exec(QCursor::pos());

	if (act_sel == _act_copy) {
		process_ctxt_menu_copy();
	} else if (act_sel == _act_set_color) {
		process_ctxt_menu_set_color();
	} else if (act_sel) {
		// process plugins extracted action
		process_ctxt_menu_action(*act_sel);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::show_hhead_ctxt_menu
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::show_hhead_ctxt_menu(const QPoint& pos)
{
	PVCombCol comb_col = (PVCombCol)horizontalHeader()->logicalIndexAt(pos);
	PVCol col = _view.get_axes_combination().get_nraw_axis(comb_col);

	// Disable hover picture
	section_hovered_enter(comb_col, false);

	// Create a new horizontal header context as it depend on the clicked column
	_hhead_ctxt_menu.clear();

	bool empty_sel = lib_view().get_output_layer().get_selection().is_empty();

	// Add view creation based on an axis.
	if (auto container =
	        PVCore::get_qobject_parent_of_type<PVDisplays::PVDisplaysContainer*>(this)) {
		// Add entries to the horizontal header context menu for new widgets
		// creation.
		PVDisplays::add_displays_view_axis_menu(_hhead_ctxt_menu, container,
		                                        (Inendi::PVView*)&lib_view(), col, comb_col);
		_hhead_ctxt_menu.addSeparator();
	}
	_action_col_unique->setEnabled(not empty_sel);
	_hhead_ctxt_menu.addAction(_action_col_unique);
	_menu_col_count_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_count_by);
	_menu_col_count_by->setEnabled(not empty_sel);
	_menu_col_sum_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_sum_by);
	_menu_col_sum_by->setEnabled(not empty_sel);
	_menu_col_min_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_min_by);
	_menu_col_min_by->setEnabled(not empty_sel);
	_menu_col_max_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_max_by);
	_menu_col_max_by->setEnabled(not empty_sel);
	_menu_col_avg_by->clear();
	_hhead_ctxt_menu.addMenu(_menu_col_avg_by);
	_menu_col_avg_by->setEnabled(not empty_sel);

	const QStringList axes = lib_view().get_axes_names_list();

	QList<QAction*> count_by_actions;
	QList<QAction*> min_by_actions;
	QList<QAction*> max_by_actions;
	QList<QAction*> sum_by_actions;
	QList<QAction*> avg_by_actions;

	QStringList summable_types = {"number_int64",  "number_uint64", "number_int32", "number_uint32",
	                              "number_uint16", "number_int16",  "number_uint8", "number_int8",
	                              "number_float",  "number_double", "duration"};

	for (PVCombCol i(0); i < axes.size(); i++) {
		if (i != comb_col) {
			const QString& axis_type = lib_view().get_axes_combination().get_axis(i).get_type();

			QAction* action_col_count_by = new QAction(axes[i], _menu_col_count_by);
			count_by_actions << action_col_count_by;
			connect(action_col_count_by, &QAction::triggered, action_col_count_by, [=]() {
				PVCol col2 = _view.get_axes_combination().get_nraw_axis(i);
				PVQNraw::show_count_by(lib_view(), col, col2,
				                       lib_view().get_selection_visible_listing(), this);
			});

			if (summable_types.contains(axis_type)) {
				QAction* action_col_min_by = new QAction(axes[i], _menu_col_min_by);
				min_by_actions << action_col_min_by;
				connect(action_col_min_by, &QAction::triggered, action_col_min_by, [=]() {
					PVCol col2 = _view.get_axes_combination().get_nraw_axis(i);
					PVQNraw::show_min_by(lib_view(), col, col2,
					                     lib_view().get_selection_visible_listing(), this);
				});
			}

			if (summable_types.contains(axis_type)) {
				QAction* action_col_max_by = new QAction(axes[i], _menu_col_max_by);
				max_by_actions << action_col_max_by;
				connect(action_col_max_by, &QAction::triggered, action_col_max_by, [=]() {
					PVCol col2 = _view.get_axes_combination().get_nraw_axis(i);
					PVQNraw::show_max_by(lib_view(), col, col2,
					                     lib_view().get_selection_visible_listing(), this);
				});
			}

			if (summable_types.contains(axis_type)) {
				QAction* action_col_sum_by = new QAction(axes[i], _menu_col_sum_by);
				sum_by_actions << action_col_sum_by;
				connect(action_col_sum_by, &QAction::triggered, action_col_sum_by, [=]() {
					PVCol col2 = _view.get_axes_combination().get_nraw_axis(i);
					PVQNraw::show_sum_by(lib_view(), col, col2,
					                     lib_view().get_selection_visible_listing(), this);
				});

				QAction* action_col_avg_by = new QAction(axes[i], _menu_col_avg_by);
				avg_by_actions << action_col_avg_by;
				connect(action_col_avg_by, &QAction::triggered, action_col_avg_by, [=]() {
					PVCol col2 = _view.get_axes_combination().get_nraw_axis(i);
					PVQNraw::show_avg_by(lib_view(), col, col2,
					                     lib_view().get_selection_visible_listing(), this);
				});
			}
		}
	}
	_menu_col_count_by->addActions(count_by_actions);
	_menu_col_min_by->addActions(min_by_actions);
	_menu_col_max_by->addActions(max_by_actions);
	_menu_col_sum_by->addActions(sum_by_actions);
	_menu_col_avg_by->addActions(avg_by_actions);

	_hhead_ctxt_menu.addSeparator();
	_hhead_ctxt_menu.addAction(_action_col_copy);
	_hhead_ctxt_menu.addAction(_action_col_sort);
	_action_col_sort->setEnabled(not empty_sel);

	/**
	 * Correlation menu
	 */
	show_hhead_ctxt_menu_correlation(comb_col);

	QAction* sel = _hhead_ctxt_menu.exec(QCursor::pos());

	// Process actions
	if (sel == _action_col_unique) {
		PVQNraw::show_unique_values(lib_view(), col, this);
	} else if (sel == _action_col_sort) {
		Qt::SortOrder order = (Qt::SortOrder) !((bool)horizontalHeader()->sortIndicatorOrder());
		sort(comb_col, order);
	} else if (sel == _action_col_copy) {
		QApplication::clipboard()->setText(_view.get_axis_name(comb_col));
	} else {
		// No selected action
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::show_hhead_ctxt_menu_correlation
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::show_hhead_ctxt_menu_correlation(PVCombCol col)
{
	const QString& this_axis_type = lib_view().get_axes_combination().get_axis(col).get_type();
	const QStringList& correlation_types_for_values = { "number_int8", "number_uint8", "number_int16", "number_uint16",
												        "number_int32", "number_uint32", "number_int64", "number_uint64",
												        "ipv4", "ipv6", "mac_address", "string" };
	const QStringList& correlation_types_for_range =  { "number_int8", "number_uint8", "number_int16", "number_uint16",
												        "number_int32", "number_uint32", "number_int64", "number_uint64",
												        "ipv4", "ipv6", "mac_address", /*"strings"*/
	                                                    "time", "duration", "number_float", "number_double" };

	// Don't show correlation menu for unsupported axes types
	if (not correlation_types_for_range.contains(this_axis_type)) {
		return;
	}

	_menu_add_correlation->clear();

	Inendi::PVRoot& root = lib_view().get_parent<Inendi::PVRoot>();

	size_t total_compatible_views_count = 0;

	for (auto* source : root.get_children<Inendi::PVSource>()) {

		size_t compatible_views_count = 0;

		// Don't allow correlation on same source
		if (source == root.current_source()) {
			continue;
		}

		QMenu* source_menu = new QMenu(QString::fromStdString(source->get_name()), this);

		size_t compatible_axes_count = 0;

		auto const views = source->get_children<Inendi::PVView>();
		bool need_view_menu = views.size() > 1;
		for (Inendi::PVView* view : views) {

			QMenu* view_menu = source_menu;

			// Don't create an intermediary view menu if there is only one view for this source
			if (need_view_menu) {
				view_menu = new QMenu(QString::fromStdString(view->get_name()), this);
				source_menu->addMenu(view_menu);
			}

			const Inendi::PVAxesCombination& ac = view->get_axes_combination();
			std::set<PVCol> unique_comb_cols(ac.get_combination().begin(),
			                                 ac.get_combination().end());
			auto const& axes = view->get_parent<Inendi::PVSource>().get_format().get_axes();
			for (PVCol original_col2 : unique_comb_cols) {
				const QString& axis_name = axes[original_col2].get_name();
				const QString& axis_type = axes[original_col2].get_type();

				// Don't show incompatible axes
				if (axis_type != this_axis_type) {
					continue;
				}

				QMenu* type_menu = new QMenu(axis_name, this);
				view_menu->addMenu(type_menu);

				// TODO : use QActionGroup for radio buttons

				auto add_correlation_f = [&](const QString& correlation_type_name, Inendi::PVCorrelationType type){
					QAction* action = new QAction(correlation_type_name, this);
					action->setCheckable(true);

					PVCol original_col1 = _view.get_axes_combination().get_nraw_axis(col);

					Inendi::PVCorrelation correlation{&lib_view(), original_col1, view, original_col2, type};
					bool existing_correlation = root.correlations().exists(correlation);

					action->setChecked(existing_correlation);

					connect(action, &QAction::triggered, [=, &root]() {
						if (not existing_correlation) {
							root.correlations().add(correlation);
						} else {
							root.correlations().remove(correlation.view1);
						}
						// refresh headers to show correlation icon right now
						horizontalHeader()->viewport()->update();
					});

					type_menu->addAction(action);
				};

				if (correlation_types_for_values.contains(axis_type)) {
					add_correlation_f("distinct values", Inendi::PVCorrelationType::VALUES);
				}
				if (correlation_types_for_range.contains(axis_type)) {
					add_correlation_f("minmax range", Inendi::PVCorrelationType::RANGE);
				}

				compatible_axes_count++;
			}

			// Don't show view menu if there is no compatible axes
			if (compatible_axes_count > 0) {
				_menu_add_correlation->addMenu(view_menu);
				compatible_views_count++;
			} else {
				delete view_menu;
			}
		}

		if (compatible_views_count == 0 && need_view_menu) {
			delete source_menu;
		}

		total_compatible_views_count += compatible_views_count;
	}

	// Don't show correlation menu if there is no compatible views
	if (total_compatible_views_count > 0) {
		_hhead_ctxt_menu.addSeparator();
		_hhead_ctxt_menu.addMenu(_menu_add_correlation);
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

	if (sel == _action_copy_row_value) {
		int idx = verticalHeader()->logicalIndexAt(pos);
		// FIXME : We should return the full line content
		QApplication::clipboard()->setText(QString::number(listing_model()->rowIndex(idx) + 1));
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
	Inendi::PVLayer& layer = lib_view().get_current_layer();
	Inendi::PVLinesProperties& lines_properties = layer.get_lines_properties();

	// Color every lines in the current selection
	for (PVRow line : listing_model()->shown_lines()) {
		if (listing_model()->current_selection().get_line_fast(line)) {
			lines_properties.set_line_properties(line, color);
		}
	}

	lib_view().process_layer_stack();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::process_ctxt_menu_action
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::process_ctxt_menu_action(QAction const& act)
{
	// FIXME : This should be done another way (see menu creation)
	// Get the filter associated with that menu entry
	QString filter_name = act.data().toString();
	Inendi::PVLayerFilter_p lib_filter =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(filter_name);
	if (!lib_filter) {
		PVLOG_ERROR("(listing context-menu) filter '%s' does not exist !\n",
		            qPrintable(filter_name));
		return;
	}

	Inendi::PVLayerFilter::hash_menu_function_t entries = lib_filter->get_menu_entries();
	QString act_name = act.text();
	if (entries.find(act_name) == entries.end()) {
		PVLOG_ERROR("(listing context-menu) unable to find action '%s' in filter '%s'.\n",
		            qPrintable(act_name), qPrintable(filter_name));
		return;
	}
	Inendi::PVLayerFilter::ctxt_menu_f args_f = entries[act_name];

	// Get the arguments
	_ctxt_args = lib_view().get_last_args_filter(filter_name);
	PVCore::PVArgumentList custom_args = args_f(
	    _ctxt_row, _ctxt_col, _view.get_axes_combination().get_nraw_axis(_ctxt_col), _ctxt_v);
	PVCore::PVArgumentList_set_common_args_from(_ctxt_args, custom_args);

	// Show the layout filter widget
	Inendi::PVLayerFilter_p fclone = lib_filter->clone<Inendi::PVLayerFilter>();
	assert(fclone);
	if (_ctxt_process) {
		_ctxt_process->deleteLater();
	}

	// Creating the PVLayerFilterProcessWidget will save the current args for this
	// filter.
	// Then we can change them !
	_ctxt_process = new PVGuiQt::PVLayerFilterProcessWidget(&lib_view(), _ctxt_args, fclone, this);

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
void PVGuiQt::PVListingView::section_hovered_enter(PVCombCol col, bool entered)
{
	lib_view().set_axis_hovered(col, entered);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::section_clicked
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::section_clicked(int col)
{
	int x = horizontalHeader()->sectionViewportPosition(col);
	int width = horizontalHeader()->sectionSize(col);
	lib_view().set_section_clicked((PVCombCol)col, verticalHeader()->width() + x + width / 2);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::highlight_column
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::highlight_column(int col, bool entered)
{
	// Mark the column for future painting and force update
	_hovered_axis = PVCombCol(entered ? col : -1);
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

		QRectF rect1(r.x() - border_width / 2 - 1, 0, border_width, height());
		QRectF rect2(r.x() + r.width() - border_width / 2 - 1, 0, border_width, height());

		border_width *= 2;
		QColor color = lib_view().get_axis(_hovered_axis).get_titlecolor().toQColor();

		qreal weighted_value =
		    (((color.redF() * 0.299) + (color.greenF() * 0.587) + (color.blueF() * 0.114)));

		int grey_level;
		if (weighted_value < threshold1) {
			grey_level = 255 * (grey3 + (weighted_value * (grey4 - grey3) / threshold1));
		} else {
			if (weighted_value < threshold2) {
				grey_level = 255 * grey4;
			} else {
				if (weighted_value < threshold3) {
					grey_level = 255 * grey1;
				} else {
					grey_level = 255 * (grey1 + ((weighted_value - threshold3) * (grey2 - grey1) /
					                             (1 - threshold3)));
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
		poly1 << QPoint(border_width / 2 - 1, 0);
		poly1 << QPoint(0, border_width / 2 - 1);
		QPolygon poly2;
		poly2 << QPoint(border_width, 0);
		poly2 << QPoint(0, border_width);
		poly2 << QPoint(border_width / 2 - 1, border_width);
		poly2 << QPoint(border_width, border_width / 2 - 1);
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
	PVRow nrows = lib_view().get_rushnraw_parent().row_count();
	const Inendi::PVSelection& sel = lib_view().get_real_output_selection();

	bool ok;
	PVRow row = QInputDialog::getInt(this, "Go to line", "Select line index", 1, 1, nrows, 1, &ok);
	row--; // Displayed rows start at index 1 but not internally

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

		if (row == PVROW_INVALID_VALUE) {
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
void PVGuiQt::PVListingView::sort(int column, Qt::SortOrder order)
{
	PVCombCol col(column);

	assert(col >= 0 && col < listing_model()->columnCount());
	auto changed = PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(false);
		    tbb::task_group_context ctxt;
		    try {
			    listing_model()->sort_on_col(col, order, ctxt);
		    } catch (boost::thread_interrupted) {
			    ctxt.cancel_group_execution();
			    throw;
		    }
		},
	    tr("Sorting..."), this);
	if (changed == PVCore::PVProgressBox::CancelState::CONTINUE) {
		horizontalHeader()->setSortIndicator(col, order);
		verticalHeader()->viewport()->update();
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingView::set_section_visible
 *
 *****************************************************************************/
void PVGuiQt::PVListingView::set_section_visible(PVCombCol col)
{
	/* Temporarily setting selection behavior to 'SelectColumns' is needed
	 * to make a column visible
	 */
	QAbstractItemView::SelectionBehavior old_sel_behavior = selectionBehavior();
	setSelectionBehavior(QAbstractItemView::SelectColumns);

	selectColumn(col);
	clearSelection();

	setSelectionBehavior(old_sel_behavior);
}

/******************************************************************************
 *
 * PVGuiQt::PVHorizontalHeaderView
 *
 *****************************************************************************/

PVGuiQt::PVHorizontalHeaderView::PVHorizontalHeaderView(Qt::Orientation orientation,
                                                        PVListingView* parent)
    : QHeaderView(orientation, parent)
{
	// These two calls are required since they are done on the headers in
	// QTableView::QTableView
	// instead of in QHeaderView::QHeaderView !
	setSectionsClickable(true);
	setHighlightSections(true);

	// Context menu of the horizontal header
	setStretchLastSection(true);
	connect(this, &PVGuiQt::PVHorizontalHeaderView::customContextMenuRequested, parent,
	        &PVListingView::show_hhead_ctxt_menu);
	setContextMenuPolicy(Qt::CustomContextMenu);

	// Save horizontal headers width to be persistent across axes combination
	// changes
	connect(this, &PVGuiQt::PVHorizontalHeaderView::sectionResized, parent,
	        &PVListingView::columnResized);

	// section <-> axis synchronisation
	connect(this, &PVGuiQt::PVHorizontalHeaderView::mouse_hovered_section, parent,
	        &PVListingView::section_hovered_enter);
	connect(this, &PVGuiQt::PVHorizontalHeaderView::sectionClicked, parent,
	        &PVListingView::section_clicked);

	// Force hover events on every theme so that "column -> axis" visual
	// synchronisation always works !
	setAttribute(Qt::WA_Hover);
}

bool PVGuiQt::PVHorizontalHeaderView::event(QEvent* ev)
{
	if (ev->type() == QEvent::HoverLeave || ev->type() == QEvent::Leave) {
		if (_index != -1) {
			Q_EMIT mouse_hovered_section(_index, false);
			_index = PVCombCol(-1);
		}
	} else if (ev->type() == QEvent::HoverMove) { // in eventFilter, this event
		                                          // would have been
		                                          // "QEvent::MouseMove"...
		QHoverEvent* mouse_event = dynamic_cast<QHoverEvent*>(ev);
		PVCombCol index = (PVCombCol)logicalIndexAt(mouse_event->pos());
		if (index != _index) {
			if (_index != -1) {
				Q_EMIT mouse_hovered_section(_index, false);
			}
			Q_EMIT mouse_hovered_section(index, true);
		}
		_index = index;
	}
	return QHeaderView::event(ev);
}

void PVGuiQt::PVHorizontalHeaderView::paintSection(QPainter* painter,
                                                   const QRect& rect,
                                                   int logicalIndex) const
{
	painter->save();
	QHeaderView::paintSection(painter, rect, logicalIndex);
	painter->restore();

	PVListingView* listing = (PVListingView*)parent();
	Inendi::PVRoot& root = listing->lib_view().get_parent<Inendi::PVRoot>();

	PVCol original_col1 =
	    listing->lib_view().get_axes_combination().get_nraw_axis((PVCombCol)logicalIndex);

	bool existing_correlation = root.correlations().exists(&listing->lib_view(), original_col1);

	if (existing_correlation) {
		QPixmap p(":/bind");
		p = p.scaledToWidth(rect.height(), Qt::SmoothTransformation);
		QRect r(QPoint(rect.right() - p.width(), rect.top()), rect.bottomRight());
		painter->drawPixmap(r, p);
	}
}
