/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QApplication>
#include <QVBoxLayout>
#include <QPushButton>
#include <QScrollBar>
#include <QMenu>
#include <QCursor>
#include <QPushButton>
#include <QDialog>

#include <pvkernel/core/qobject_helpers.h>

#include <inendi/PVSource.h>

#include <pvguiqt/PVStatsListingWidget.h>
#include <pvguiqt/PVQNraw.h>

#include <pvcop/db/algo.h>

constexpr int QTABLEWIDGET_OFFSET = 4;

// Originally from
// http://stackoverflow.com/questions/8766633/how-to-determine-the-correct-size-of-a-qtablewidget
static uint32_t compute_qtablewidget_height(QTableWidget* stats)
{
	int h = stats->horizontalHeader()->height() + QTABLEWIDGET_OFFSET;
	for (int i = 0; i < stats->verticalHeader()->count(); i++) {
		if (!stats->isRowHidden(i)) {
			h += stats->rowHeight(i);
		}
	}

	return h;
}

/******************************************************************************
 *
 * PVGuiQt::PVStatsListingWidget
 *
 *****************************************************************************/
const QColor PVGuiQt::PVStatsListingWidget::INVALID_COLOR = QColor(0xf9, 0xd7, 0xd7);

PVGuiQt::PVStatsListingWidget::PVStatsListingWidget(PVGuiQt::PVListingView* listing_view)
    : _listing_view(listing_view)
{
	_params.clear();

	QVBoxLayout* main_layout = new QVBoxLayout();

	_stats_panel = new QTableWidget(this);
	_stats_panel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->viewport()->setFocusPolicy(Qt::NoFocus);
	//_stats_panel->hide();
	QStringList horizontal_header_labels;
	for (PVCol col = 0; col < _listing_view->horizontalHeader()->count(); col++) {
		horizontal_header_labels
		    << _listing_view->model()->headerData(col, Qt::Horizontal).toString();
	}
	_stats_panel->setHorizontalHeaderLabels(horizontal_header_labels);
	_stats_panel->horizontalHeader()->setStretchLastSection(true);
	_stats_panel->horizontalHeader()->hide();
	_stats_panel->setSelectionMode(QTableWidget::NoSelection);

	main_layout->setSpacing(0);
	main_layout->setContentsMargins(0, 0, 0, 0);

	QPushButton* hide_button = new QPushButton("...");
	hide_button->setToolTip(tr("Toggle stats panel visibility"));
	hide_button->setMaximumHeight(10);
	hide_button->setFlat(true);
	connect(hide_button, SIGNAL(clicked(bool)), this, SLOT(toggle_stats_panel_visibility()));

	main_layout->addWidget(_listing_view);
	main_layout->addWidget(hide_button);
	main_layout->addWidget(_stats_panel);

	setLayout(main_layout);

	connect(_listing_view->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this,
	        SLOT(update_header_width(int, int, int)));
	_stats_panel->setVerticalHeader(new __impl::PVVerticalHeaderView(this));
	_stats_panel->verticalHeader()->viewport()->installEventFilter(this);
	connect(_listing_view, SIGNAL(resized()), this, SLOT(resize_panel()));
	connect(_listing_view->horizontalScrollBar(), SIGNAL(actionTriggered(int)), this,
	        SLOT(update_scrollbar_position()));

	// Observe selection to handle automatic refresh mode
	Inendi::PVView& view_sp = _listing_view->lib_view();
	view_sp._update_output_selection.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVStatsListingWidget::selection_changed));

	// Observe layerstack to handle automatic refresh mode
	view_sp._update_output_layer.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVStatsListingWidget::selection_changed));

	// Observer axes combination changes
	view_sp._axis_combination_updated.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVStatsListingWidget::axes_comb_changed));

	init_plugins();
	create_vhead_ctxt_menu();

	resize_panel();
	refresh();
}

void PVGuiQt::PVStatsListingWidget::create_vhead_ctxt_menu()
{
	_vhead_ctxt_menu = new QMenu(this);

	for (int row = 0; row < _stats_panel->rowCount(); row++) {
		const QString& section_text = _stats_panel->verticalHeaderItem(row)->text();
		QAction* act = new QAction(section_text, this);
		act->setCheckable(true);
		act->setEnabled(_stats_panel->isRowHidden(row));
		act->setChecked(!_stats_panel->isRowHidden(row));
		act->setData(row);
		connect(act, SIGNAL(triggered(bool)), this, SLOT(plugin_visibility_toggled(bool)));
		_vhead_ctxt_menu->addAction(act);
	}
}

void PVGuiQt::PVStatsListingWidget::plugin_visibility_toggled(bool checked)
{
	QAction* act = (QAction*)sender();
	assert(act);
	int row = act->data().toInt();
	assert(row < _stats_panel->rowCount());
	if (checked) {
		_stats_panel->showRow(row);
	} else {
		_stats_panel->hideRow(row);
	}
	_stats_panel->setMaximumHeight(compute_qtablewidget_height(_stats_panel));
};

void PVGuiQt::PVStatsListingWidget::resize_listing_column_if_needed(int col)
{
	int cell_max_size = 0;
	for (int row = 0; row < _stats_panel->rowCount(); row++) {
		__impl::PVCellWidgetBase* cell_widget =
		    ((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col));
		assert(cell_widget);
		cell_max_size = std::max(cell_max_size, cell_widget->minimum_size());
	}

	int listing_col_width = _listing_view->columnWidth(col);
	if (listing_col_width < cell_max_size) {
		_listing_view->setColumnWidth(col, cell_max_size);
	}
}

void PVGuiQt::PVStatsListingWidget::init_plugins()
{
	for (PVCol col = 0; col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->insertColumn(col);
	}

	_row_distinct =
	    init_plugin<__impl::PVUniqueValuesCellWidget>("distinct\nvalues", /* visible = */ true);
	_row_sum = init_plugin<__impl::PVSumCellWidget>("sum", /* visible = */ false);
	_row_min = init_plugin<__impl::PVMinCellWidget>("min", /* visible = */ false);
	_row_max = init_plugin<__impl::PVMaxCellWidget>("max", /* visible = */ false);
	_row_avg = init_plugin<__impl::PVAverageCellWidget>("avg", /* visible = */ false);

	for (PVCol col = 0; col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->horizontalHeader()->sectionSize(col));
	}
}

void PVGuiQt::PVStatsListingWidget::refresh()
{
	sync_vertical_headers();

	for (PVCol col = 0; col < _stats_panel->columnCount(); col++) {
		for (int row = 0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))->auto_refresh();
		}
	}
}

void PVGuiQt::PVStatsListingWidget::sync_vertical_headers()
{
	// Sync tables vertical header width
	int stats_header_width = _stats_panel->verticalHeader()->width();
	int listing_header_width = _listing_view->verticalHeader()->width();

	if (stats_header_width > listing_header_width) {
		_listing_view->verticalHeader()->setFixedWidth(stats_header_width);
		QMetaObject::invokeMethod(_listing_view, "updateGeometries");
	} else {
		_stats_panel->verticalHeader()->setFixedWidth(listing_header_width);
		QMetaObject::invokeMethod(_stats_panel, "updateGeometries");
	}
}

void PVGuiQt::PVStatsListingWidget::toggle_stats_panel_visibility()
{
	_stats_panel->setVisible(!_stats_panel->isVisible());
}

void PVGuiQt::PVStatsListingWidget::update_header_width(int column,
                                                        int /*old_width*/,
                                                        int new_width)
{
	_stats_panel->setColumnWidth(column, new_width);
}

void PVGuiQt::PVStatsListingWidget::resize_panel()
{
	_stats_panel->setMaximumHeight(compute_qtablewidget_height(_stats_panel));
	for (PVCol col = 0; col < _stats_panel->columnCount(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->columnWidth(col));
	}
}

void PVGuiQt::PVStatsListingWidget::update_scrollbar_position()
{
	// Difference between QScrollBar::value() and QScrollBar::sliderPosition():
	// From Qt documentation: If tracking is enabled (the default), the slider
	// emits the valueChanged() signal while the slider is being dragged.
	//                        If tracking is disabled, the slider emits the
	//                        valueChanged() signal only when the user releases
	//                        the slider.

	// In order to avoid an offset between the stats and listing tables when the
	// vertical scrollbar is on the rightmost position
	// the maximum width of the stats panel is in this case reduced of by the
	// vertical scrollbar width. (fix bug #238)
	if (_listing_view->horizontalScrollBar()->sliderPosition() ==
	    _listing_view->horizontalScrollBar()->maximum()) {
		_old_maximum_width = _stats_panel->maximumSize().width();
		_stats_panel->setMaximumWidth(_stats_panel->minimumSize().width() +
		                              _listing_view->horizontalScrollBar()->width() +
		                              QTABLEWIDGET_OFFSET / 2);
		_maxed = true;
	} else {
		if (_maxed) {
			_stats_panel->setMaximumWidth(_old_maximum_width);
			_maxed = false;
		}
	}
	_stats_panel->horizontalScrollBar()->setSliderPosition(
	    _listing_view->horizontalScrollBar()->sliderPosition());
}

void PVGuiQt::PVStatsListingWidget::selection_changed()
{
	// Abort the thread if running
	__impl::PVCellWidgetBase::cancel_thread();
	refresh();
}

void PVGuiQt::PVStatsListingWidget::set_refresh_buttons_enabled(bool loading)
{
	for (PVCol col = 0; col < _stats_panel->columnCount(); col++) {
		for (int row = 0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))
			    ->set_refresh_button_enabled(loading);
		}
	}
}

void PVGuiQt::PVStatsListingWidget::axes_comb_changed()
{
	int old_count = _stats_panel->columnCount();
	int new_count = _listing_view->lib_view().get_axes_count();
	int delta = new_count - old_count;
	if (delta > 0) {
		for (PVCol col = old_count - 1; col < new_count - 1; col++) {
			_stats_panel->insertColumn(col);
			create_item<__impl::PVUniqueValuesCellWidget>(_row_distinct, col);
			create_item<__impl::PVSumCellWidget>(_row_sum, col);
			create_item<__impl::PVMinCellWidget>(_row_min, col);
			create_item<__impl::PVMaxCellWidget>(_row_max, col);
			create_item<__impl::PVAverageCellWidget>(_row_avg, col);
		}
	} else {
		_stats_panel->setColumnCount(
		    _listing_view->lib_view().get_axes_count()); // Widgets gets deleted
	}
	resize_panel();
	for (PVCol col = 0; col < _stats_panel->columnCount(); col++) {
		for (int row = 0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))->refresh(true);
		}
		_stats_panel->setColumnWidth(col, _listing_view->columnWidth(col));
	}
}

void PVGuiQt::PVStatsListingWidget::vertical_header_section_clicked(const QPoint&)
{
	if (!_vhead_ctxt_menu) {
		return;
	}

	_vhead_ctxt_menu->exec(QCursor::pos());
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVVerticalHeaderView
 *
 *****************************************************************************/
PVGuiQt::__impl::PVVerticalHeaderView::PVVerticalHeaderView(PVStatsListingWidget* parent)
    : QHeaderView(Qt::Vertical, parent)
{
	// These two calls are required since they are done on the headers in
	// QTableView::QTableView
	// instead of in QHeaderView::QHeaderView !
	setSectionsClickable(true);
	setHighlightSections(true);

	// Context menu of the horizontal header
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), parent,
	        SLOT(vertical_header_section_clicked(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVCellWidgetBase
 *
 *****************************************************************************/
QMovie* PVGuiQt::__impl::PVCellWidgetBase::_loading_movie = nullptr;
std::thread PVGuiQt::__impl::PVCellWidgetBase::_thread = std::thread();
tbb::task_group_context* PVGuiQt::__impl::PVCellWidgetBase::_ctxt = new tbb::task_group_context();
bool PVGuiQt::__impl::PVCellWidgetBase::_thread_running = false;

PVGuiQt::__impl::PVCellWidgetBase::PVCellWidgetBase(QTableWidget* table,
                                                    Inendi::PVView& view,
                                                    QTableWidgetItem* item)
    : _table(table)
    , _view(view)
    , _item(item)
    , _refresh_pixmap(QPixmap::fromImage(QImage(":/refresh_small_grey")))
    , _autorefresh_on_pixmap(QPixmap::fromImage(QImage(":/icon_linked")))
    , _autorefresh_off_pixmap(QPixmap::fromImage(QImage(":/icon_unlinked")))
{
	// connect(table->verticalHeader(), SIGNAL(sectionClicked(int)), this,
	// SLOT(vertical_header_clicked(int)));

	_text = new QLabel();

	_refresh_icon = new QPushButton();
	_refresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_refresh_icon->setFlat(true);
	_refresh_icon->setStyleSheet("QPushButton { border: none; }");
	_refresh_icon->setIcon(_refresh_pixmap);
	_refresh_icon->setFocusPolicy(Qt::NoFocus);
	_refresh_icon->setToolTip("Refresh");
	connect(_refresh_icon, SIGNAL(clicked(bool)), this, SLOT(refresh()));

	_loading_label = new PVLoadingLabel(this);
	_loading_label->setMovie(get_movie());
	_loading_label->setStyleSheet("QLabel { padding-right: 4px }");
	_loading_label->setVisible(false);
	_loading_label->setCursor(QCursor(Qt::PointingHandCursor));
	_loading_label->setToolTip("Click to abort");
	connect(_loading_label, SIGNAL(clicked()), this, SLOT(cancel_thread()));

	_autorefresh_icon = new QPushButton();
	_autorefresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_autorefresh_icon->setFlat(true);
	_autorefresh_icon->setStyleSheet("QPushButton { border: none; }");
	_autorefresh_icon->setIcon(_autorefresh_off_pixmap);
	_autorefresh_icon->setFocusPolicy(Qt::NoFocus);
	_autorefresh_icon->setToolTip("Toggle auto refresh");
	_autorefresh_icon->setVisible(false); // Disabled before having a better job handling pipeline
	connect(_autorefresh_icon, SIGNAL(clicked(bool)), this, SLOT(toggle_auto_refresh()));

	connect(this, SIGNAL(refresh_impl_finished(QString)), this, SLOT(refreshed(QString)));

	_main_layout = new QHBoxLayout();
	_main_layout->setSizeConstraint(QLayout::SetMinimumSize);
	QSizePolicy size_policy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	setSizePolicy(size_policy);
	_main_layout->setSpacing(2);
	_main_layout->setContentsMargins(2, 0, 2, 0);
	_main_layout->addWidget(_text);
	_main_layout->addStretch(1);
	_customizable_layout = new QHBoxLayout();
	_customizable_layout->setSizeConstraint(QLayout::SetMinimumSize);
	_main_layout->addLayout(_customizable_layout);
	_main_layout->addWidget(_refresh_icon);
	_main_layout->addWidget(_loading_label);
	//_main_layout->addWidget(_autorefresh_icon);

	// Context menu
	_ctxt_menu = new QMenu(this);
	QAction* copy = new QAction(tr("Copy"), _ctxt_menu);
	connect(copy, SIGNAL(triggered()), this, SLOT(copy_to_clipboard()));
	_ctxt_menu->addAction(copy);
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this,
	        SLOT(context_menu_requested(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	setLayout(_main_layout);

	QString column_type = _view.get_parent<Inendi::PVSource>()
	                          .get_format()
	                          .get_axes()
	                          .at(get_real_axis_col())
	                          .get_type();
	_is_summable =
	    (column_type == "number_float" || column_type == "number_uint32" ||
	     column_type == "number_int32"); // FIXME : this should be capabilities, not types names !
}

void PVGuiQt::__impl::PVCellWidgetBase::context_menu_requested(const QPoint&)
{
	if (_valid) {
		_ctxt_menu->exec(QCursor::pos());
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::copy_to_clipboard()
{
	QByteArray ba;
	ba = _text->text().toLocal8Bit();

	QClipboard* clipboard = QApplication::clipboard();
	QMimeData* mdata = new QMimeData();
	mdata->setData("text/plain", ba);
	clipboard->setMimeData(mdata);
}

PVGuiQt::PVStatsListingWidget* PVGuiQt::__impl::PVCellWidgetBase::get_panel()
{
	return (PVGuiQt::PVStatsListingWidget*)
	    PVCore::get_qobject_parent_of_type<PVGuiQt::PVStatsListingWidget*>(this);
}

typename PVGuiQt::PVStatsListingWidget::PVParams& PVGuiQt::__impl::PVCellWidgetBase::get_params()
{
	PVGuiQt::PVStatsListingWidget* stats_panel = get_panel();
	assert(stats_panel);

	return stats_panel->get_params()[get_real_axis_row()][get_real_axis_col()];
}

QMovie* PVGuiQt::__impl::PVCellWidgetBase::get_movie()
{
	if (_loading_movie == nullptr) {
		_loading_movie = new QMovie(":/inendi-loading-animation");
		_loading_movie->setScaledSize(QSize(16, 16));
	}
	return _loading_movie;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_loading(bool loading)
{
	_refresh_icon->setVisible(!loading);
	_loading_label->setVisible(loading);
	if (loading) {
		get_movie()->start();
	} else {
		get_movie()->stop();
	}
	get_panel()->set_refresh_buttons_enabled(loading);
}

void PVGuiQt::__impl::PVCellWidgetBase::cancel_thread()
{
	if (_thread_running) {
		_ctxt->cancel_group_execution();
		if (_thread.joinable()) {
			_thread.join();
		}
		_ctxt->reset();
		_thread_running = false;
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::refresh(bool from_cache /* = false */)
{
	QString cached_value = get_params().cached_value;
	bool auto_refresh = get_params().auto_refresh;

	if (!from_cache) {
		if (cached_value.isEmpty()) {
			if (!_thread_running) {
				set_loading(true);
				if (_thread.joinable()) {
					_thread.join();
				}
				_thread_running = true;
				std::thread th(&PVGuiQt::__impl::PVCellWidgetBase::refresh_impl, this);
				_thread.swap(th);
			}
		} else {
			set_valid(cached_value, auto_refresh);
		}
	} else if (from_cache && !cached_value.isEmpty()) {
		set_valid(cached_value, auto_refresh);
	} else {
		set_invalid();
		get_params().cached_value = QString();
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::auto_refresh()
{
	get_params().cached_value = QString();
	bool auto_refresh = get_params().auto_refresh;
	if (auto_refresh) {
		refresh();
	} else {
		set_invalid();
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::refreshed(QString value)
{
	if (value != "") {
		QString cached_value = value;
		get_params().cached_value = cached_value;
		bool auto_refresh = get_params().auto_refresh;
		set_valid(cached_value, auto_refresh);
	}

	QString cached_value = value;
	get_params().cached_value = cached_value;
	bool auto_refresh = get_params().auto_refresh;
	set_valid(cached_value, auto_refresh);

	set_loading(false);
	_thread_running = false;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_refresh_button_enabled(bool loading)
{
	//_refresh_icon->setCursor(QCursor(loading ? Qt::ArrowCursor :
	// Qt::PointingHandCursor));
	if (_valid) {
		return;
	}
	_refresh_icon->setEnabled(!loading);
}

void PVGuiQt::__impl::PVCellWidgetBase::set_invalid()
{
	_refresh_icon->setEnabled(true);
	_item->setBackgroundColor(PVGuiQt::PVStatsListingWidget::INVALID_COLOR);
	_text->setText("N/A");
	_valid = false;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_valid(const QString& value, bool auto_refresh)
{
	_text->setText(value);
	_refresh_icon->setEnabled(false);
	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_on_pixmap : _autorefresh_off_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
	_item->setBackground(QBrush(Qt::NoBrush));
	_valid = true;
	Q_EMIT cell_refreshed(get_widget_cell_col());
}

void PVGuiQt::__impl::PVCellWidgetBase::vertical_header_clicked(int)
{
	refresh();
}

void PVGuiQt::__impl::PVCellWidgetBase::toggle_auto_refresh()
{
	bool& auto_refresh = get_params().auto_refresh;
	auto_refresh = !auto_refresh;

	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_on_pixmap : _autorefresh_off_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVUniqueValuesCellWidget
 *
 *****************************************************************************/
PVGuiQt::__impl::PVUniqueValuesCellWidget::PVUniqueValuesCellWidget(QTableWidget* table,
                                                                    Inendi::PVView& view,
                                                                    QTableWidgetItem* item)
    : PVCellWidgetBase(table, view, item)
{
	QPushButton* unique_values_dlg_icon = new QPushButton();
	unique_values_dlg_icon->setCursor(QCursor(Qt::PointingHandCursor));
	unique_values_dlg_icon->setFlat(true);
	unique_values_dlg_icon->setStyleSheet("QPushButton { border: none; } "
	                                      "QPushButton:pressed { padding-left : "
	                                      "0px; }");
	unique_values_dlg_icon->setIcon(QPixmap::fromImage(QImage(":/fileslist_black")));
	unique_values_dlg_icon->setFocusPolicy(Qt::NoFocus);
	unique_values_dlg_icon->setToolTip("Show distinct values");
	connect(unique_values_dlg_icon, SIGNAL(clicked(bool)), this, SLOT(show_unique_values_dlg()));
	_customizable_layout->addWidget(unique_values_dlg_icon);
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::refresh_impl()
{
	const pvcop::db::array col_in =
	    _view.get_rushnraw_parent().collection().column(get_real_axis_col());
	pvcop::db::array col1_out;
	pvcop::db::array col2_out;

	pvcop::db::algo::distinct(col_in, col1_out, col2_out, _view.get_selection_visible_listing());

	Q_EMIT refresh_impl_finished(
	    QString("%L1").arg(col1_out.size())); // We must go back on the Qt thread to update the GUI
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::show_unique_values_dlg()
{
	if (!_dialog) {
		PVQNraw::show_unique_values(_view, _view.get_rushnraw_parent(), get_real_axis_col(),
		                            _view.get_selection_visible_listing(), this, &_dialog);
		connect(_dialog, SIGNAL(finished(int)), this, SLOT(unique_values_dlg_closed()));
	} else {
		_dialog->close();
	}
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::unique_values_dlg_closed()
{
	disconnect(_dialog, SIGNAL(finished(int)));
	_dialog = nullptr;
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVSumCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVSumCellWidget::refresh_impl()
{
	const pvcop::db::array column =
	    _view.get_rushnraw_parent().collection().column(get_real_axis_col());

	double sum = pvcop::db::algo::sum(column, _view.get_selection_visible_listing());

	double intpart;
	bool integer = std::modf(sum, &intpart) == 0.0;
	QString sum_str = integer ? QString("%L1").arg((int64_t)sum) : QString("%L1").arg(sum, 0, 'f');

	Q_EMIT refresh_impl_finished(sum_str); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVMinCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVMinCellWidget::refresh_impl()
{
	const pvcop::db::array column =
	    _view.get_rushnraw_parent().collection().column(get_real_axis_col());

	const pvcop::db::array& min_array =
	    pvcop::db::algo::min(column, _view.get_selection_visible_listing());

	std::string min = min_array.size() == 1 ? min_array.at(0) : "";

	Q_EMIT refresh_impl_finished(
	    QString(min.c_str())); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVMaxCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVMaxCellWidget::refresh_impl()
{
	const pvcop::db::array column =
	    _view.get_rushnraw_parent().collection().column(get_real_axis_col());

	const pvcop::db::array& max_array =
	    pvcop::db::algo::max(column, _view.get_selection_visible_listing());

	std::string max = max_array.size() == 1 ? max_array.at(0) : "";

	Q_EMIT refresh_impl_finished(
	    QString(max.c_str())); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVAverageCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVAverageCellWidget::refresh_impl()
{
	const pvcop::db::array column =
	    _view.get_rushnraw_parent().collection().column(get_real_axis_col());

	double avg = pvcop::db::algo::average(column, _view.get_selection_visible_listing());

	double intpart;
	bool integer = std::modf(avg, &intpart) == 0.0;
	QString avg_str = integer ? QString("%L1").arg((int64_t)avg) : QString("%L1").arg(avg, 0, 'f');

	Q_EMIT refresh_impl_finished(avg_str); // We must go back on the Qt thread to update the GUI
}
