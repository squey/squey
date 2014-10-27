/**
 * \file PVStatsListingWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */
#include <unistd.h> // for usleep

#include <QApplication>
#include <QVBoxLayout>
#include <QPushButton>
#include <QScrollBar>
#include <QMenu>
#include <QCursor>
#include <QPushButton>
#include <QDialog>

#include <pvkernel/core/qobject_helpers.h>

#include <pvguiqt/PVStatsListingWidget.h>
#include <pvguiqt/PVQNraw.h>

#ifdef PICVIZ_DEVELOPER_MODE
	#define SIMULATE_LONG_COMPUTATION 0
#endif

constexpr int QTABLEWIDGET_OFFSET = 4;

// Originally from http://stackoverflow.com/questions/8766633/how-to-determine-the-correct-size-of-a-qtablewidget
static QSize compute_qtablewidget_size(QTableWidget* stats, QTableView* listing)
{
   int w = listing->verticalHeader()->width() + /*listing->verticalScrollBar()->width()*/ + QTABLEWIDGET_OFFSET;
   for (int i = 0; i < listing->horizontalHeader()->count(); i++) {
	   w += listing->columnWidth(i);
   }

   int h = stats->horizontalHeader()->height() + QTABLEWIDGET_OFFSET;
   for (int i = 0; i < stats->verticalHeader()->count(); i++) {
	   if(!stats->isRowHidden(i)) {
		   h += stats->rowHeight(i);
	   }
   }

   return QSize(w, h);
}

/******************************************************************************
 *
 * PVGuiQt::PVStatsListingWidget
 *
 *****************************************************************************/
const QColor PVGuiQt::PVStatsListingWidget::INVALID_COLOR = QColor(0xf9, 0xd7, 0xd7);

PVGuiQt::PVStatsListingWidget::PVStatsListingWidget(PVGuiQt::PVListingView* listing_view) : _listing_view(listing_view)
{
	_params.clear();

	QVBoxLayout* main_layout = new QVBoxLayout();

	_stats_panel = new QTableWidget(this);
	_stats_panel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->viewport()->setFocusPolicy(Qt::NoFocus);
	//_stats_panel->hide();
	QStringList horizontal_header_labels;
	for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
		horizontal_header_labels << _listing_view->model()->headerData(col, Qt::Horizontal).toString();
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

	connect(_listing_view->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(update_header_width(int, int, int)));
	_stats_panel->setVerticalHeader(new __impl::PVVerticalHeaderView(this));
	_stats_panel->verticalHeader()->viewport()->installEventFilter(this);
	connect(_listing_view, SIGNAL(resized()), this, SLOT(resize_panel()));
	connect(_listing_view->horizontalScrollBar(), SIGNAL(actionTriggered(int)), this, SLOT(update_scrollbar_position()));

	// Observe selection to handle automatic refresh mode
	PVHive::PVObserverSignal<Picviz::PVSelection>* obs_sel = new PVHive::PVObserverSignal<Picviz::PVSelection>(this);
	Picviz::PVView_sp view_sp = _listing_view->lib_view().shared_from_this();
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *obs_sel);
	obs_sel->connect_refresh(this, SLOT(selection_changed()));

	// Observe layerstack to handle automatic refresh mode
	PVHive::PVObserverSignal<Picviz::PVLayer>* obs_layer = new PVHive::PVObserverSignal<Picviz::PVLayer>();
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& v) { return &v.get_output_layer(); }, *obs_layer);
	obs_layer->connect_refresh(this, SLOT(selection_changed()));

	// Observer axes combination changes
	PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t>* obs_axes_comb = new PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t>;
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& v) { return &v.get_axes_combination().get_axes_index_list(); }, *obs_axes_comb);
	obs_axes_comb->connect_refresh(this, SLOT(axes_comb_changed()));

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
	QAction* act = (QAction*) sender();
	assert(act);
	int row = act->data().toInt();
	assert(row < _stats_panel->rowCount());
	if (checked) {
		_stats_panel->showRow(row);
	}
	else {
		_stats_panel->hideRow(row);
	}
	_stats_panel->setMaximumSize(compute_qtablewidget_size(_stats_panel, _listing_view));
};

void PVGuiQt::PVStatsListingWidget::resize_listing_column_if_needed(int col)
{
	int cell_max_size = 0;
	for (int row=0; row < _stats_panel->rowCount(); row++) {
		__impl::PVCellWidgetBase* cell_widget = ((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col));
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
	for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->insertColumn(col);
	}

	init_plugin<__impl::PVUniqueValuesCellWidget>("distinct\nvalues", /* visible = */ true );
	init_plugin<__impl::PVSumCellWidget>("sum", /* visible = */ false);
	init_plugin<__impl::PVMinCellWidget>("min", /* visible = */ false);
	init_plugin<__impl::PVMaxCellWidget>("max", /* visible = */ false);
	init_plugin<__impl::PVAverageCellWidget>("avg", /* visible = */ false);

	for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->horizontalHeader()->sectionSize(col));
	}
}

bool PVGuiQt::PVStatsListingWidget::eventFilter(QObject* obj, QEvent* event)
{
	// This is needed as _stats_panel->verticalHeader()->setCursor(QCursor(Qt::PointingHandCursor)) isn't working obviously...
	/*if (event->type() == QEvent::Enter) {
		setCursor(QCursor(Qt::PointingHandCursor));
		return true;
	}
	else if (event->type() == QEvent::Leave) {
		setCursor(QCursor(Qt::ArrowCursor));
		return true;
	}*/
	return QWidget::eventFilter(obj, event);
}

void PVGuiQt::PVStatsListingWidget::refresh()
{
	sync_vertical_headers();

	for (PVCol col=0; col < _stats_panel->columnCount(); col++) {
		for (int row=0; row < _stats_panel->rowCount(); row++) {
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
	}
	else {
		_stats_panel->verticalHeader()->setFixedWidth(listing_header_width);
		QMetaObject::invokeMethod(_stats_panel, "updateGeometries");
	}
}

void PVGuiQt::PVStatsListingWidget::toggle_stats_panel_visibility()
{
	_stats_panel->setVisible(!_stats_panel->isVisible());
}

void PVGuiQt::PVStatsListingWidget::update_header_width(int column, int /*old_width*/, int new_width)
{
	_stats_panel->setColumnWidth(column, new_width);
}

void PVGuiQt::PVStatsListingWidget::resize_panel()
{
	_stats_panel->setMaximumSize(compute_qtablewidget_size(_stats_panel, _listing_view));
	for (PVCol col=0; col < _stats_panel->columnCount(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->columnWidth(col));
	}
}

void PVGuiQt::PVStatsListingWidget::update_scrollbar_position()
{
	// Difference between QScrollBar::value() and QScrollBar::sliderPosition():
	// From Qt documentation: If tracking is enabled (the default), the slider emits the valueChanged() signal while the slider is being dragged.
	//                        If tracking is disabled, the slider emits the valueChanged() signal only when the user releases the slider.

	// In order to avoid an offset between the stats and listing tables when the vertical scrollbar is on the rightmost position
	// the maximum width of the stats panel is in this case reduced of by the vertical scrollbar width. (fix bug #238)
	if (_listing_view->horizontalScrollBar()->sliderPosition() == _listing_view->horizontalScrollBar()->maximum()) {
		_old_maximum_width = _stats_panel->maximumSize().width();
		_stats_panel->setMaximumWidth(_stats_panel->minimumSize().width() + _listing_view->horizontalScrollBar()->width() +QTABLEWIDGET_OFFSET/2);
		_maxed = true;
	}
	else {
		if (_maxed) {
			_stats_panel->setMaximumWidth(_old_maximum_width);
			_maxed = false;
		}
	}
	_stats_panel->horizontalScrollBar()->setSliderPosition(_listing_view->horizontalScrollBar()->sliderPosition());
}

void PVGuiQt::PVStatsListingWidget::selection_changed()
{
	// Abort the thread if running
	__impl::PVCellWidgetBase::cancel_thread();
	refresh();
}

void PVGuiQt::PVStatsListingWidget::set_refresh_buttons_enabled(bool loading)
{
	for (PVCol col=0; col < _stats_panel->columnCount(); col++) {
		for (int row=0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))->set_refresh_button_enabled(loading);
		}
	}
}

void PVGuiQt::PVStatsListingWidget::axes_comb_changed()
{
	int old_count = _stats_panel->columnCount();
	int new_count = _listing_view->lib_view().get_axes_count();
	int delta = new_count - old_count;
	if (delta > 0) {
		for (PVCol col=old_count-1; col < new_count-1; col++) {
			_stats_panel->insertColumn(col);
			for (int row=0; row < _stats_panel->rowCount(); row++) {
				create_item<__impl::PVUniqueValuesCellWidget>(row, col);
			}
		}
	}
	else {
		_stats_panel->setColumnCount(_listing_view->lib_view().get_axes_count()); // Widgets gets deleted
	}
	resize_panel();
	for (PVCol col=0; col < _stats_panel->columnCount(); col++) {
		for (int row=0; row < _stats_panel->rowCount(); row++) {
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
PVGuiQt::__impl::PVVerticalHeaderView::PVVerticalHeaderView(PVStatsListingWidget* parent) : QHeaderView(Qt::Vertical, parent)
{
	// These two calls are required since they are done on the headers in QTableView::QTableView
	// instead of in QHeaderView::QHeaderView !
	setClickable(true);
	setHighlightSections(true);

	// Context menu of the horizontal header
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), parent, SLOT(vertical_header_section_clicked(const QPoint&)));
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

PVGuiQt::__impl::PVCellWidgetBase::PVCellWidgetBase(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) :
	_table(table),
	_view(view),
	_item(item),
	_refresh_pixmap(QPixmap::fromImage(QImage(":/refresh_small_grey"))),
	_autorefresh_on_pixmap(QPixmap::fromImage(QImage(":/icon_linked"))),
	_autorefresh_off_pixmap(QPixmap::fromImage(QImage(":/icon_unlinked")))
{
	//connect(table->verticalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(vertical_header_clicked(int)));

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

	connect(this, SIGNAL(refresh_impl_finished(QString, bool)), this, SLOT(refreshed(QString, bool)));

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
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(context_menu_requested(const QPoint&)));
	setContextMenuPolicy(Qt::CustomContextMenu);

	setLayout(_main_layout);
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

	QClipboard *clipboard = QApplication::clipboard();
	QMimeData* mdata = new QMimeData();
	mdata->setData("text/plain", ba);
	clipboard->setMimeData(mdata);
}

PVGuiQt::PVStatsListingWidget* PVGuiQt::__impl::PVCellWidgetBase::get_panel()
{
	return (PVGuiQt::PVStatsListingWidget*) PVCore::get_qobject_parent_of_type<PVGuiQt::PVStatsListingWidget*>(this);
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
		_loading_movie = new QMovie(":/picviz-loading-animation");
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
	}
	else {
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
		}
		else {
			set_valid(cached_value, auto_refresh);
		}
	}
	else if (from_cache && !cached_value.isEmpty()) {
		set_valid(cached_value, auto_refresh);
	}
	else {
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
	}
	else {
		set_invalid();
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::refreshed(QString value, bool valid)
{
	if (valid) {
		QString cached_value = value;
		get_params().cached_value = cached_value;
		bool auto_refresh = get_params().auto_refresh;
		set_valid(cached_value, auto_refresh);
	}
	else {
		set_invalid();
	}
	set_loading(false);
	_thread_running = false;
}

void PVGuiQt::__impl::PVCellWidgetBase::set_refresh_button_enabled(bool loading)
{
	//_refresh_icon->setCursor(QCursor(loading ? Qt::ArrowCursor : Qt::PointingHandCursor));
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
	emit cell_refreshed(get_widget_cell_col());
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
PVGuiQt::__impl::PVUniqueValuesCellWidget::PVUniqueValuesCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) :
	PVCellWidgetBase(table, view, item),
	_unique_values_pixmap(QPixmap::fromImage(QImage(":/fileslist_black")))
{
	_unique_values_dlg_icon = new QPushButton();
	_unique_values_dlg_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_unique_values_dlg_icon->setFlat(true);
	_unique_values_dlg_icon->setStyleSheet("QPushButton { border: none; } QPushButton:pressed { padding-left : 0px; }");
	_unique_values_dlg_icon->setIcon(_unique_values_pixmap);
	_unique_values_dlg_icon->setFocusPolicy(Qt::NoFocus);
	_unique_values_dlg_icon->setToolTip("Show distinct values");
	connect(_unique_values_dlg_icon, SIGNAL(clicked(bool)), this, SLOT(show_unique_values_dlg()));
	_customizable_layout->addWidget(_unique_values_dlg_icon);
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::refresh_impl()
{
	PVRush::PVNraw::unique_values_t values;
	uint64_t min;
	uint64_t max;
	bool valid = _view.get_rushnraw_parent().get_unique_values(get_real_axis_col(), values, min, max, *_view.get_selection_visible_listing(), _ctxt);
#if SIMULATE_LONG_COMPUTATION
	for (uint32_t i = 0; i < 10 && !_ctxt->is_group_execution_cancelled(); i++) {
		usleep(500000);
	}
	valid = !_ctxt->is_group_execution_cancelled();
#endif
	emit refresh_impl_finished(QString("%L1").arg(values.size()), valid); // We must go back on the Qt thread to update the GUI
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::show_unique_values_dlg()
{
	if (!_dialog) {
		Picviz::PVView_sp view = (const_cast<Picviz::PVView&>(_view)).shared_from_this();
		PVQNraw::show_unique_values(view, _view.get_rushnraw_parent(), get_real_axis_col(), *_view.get_selection_visible_listing(), this, &_dialog);
		connect(_dialog, SIGNAL(finished(int)), this, SLOT(unique_values_dlg_closed()));
	}
	else {
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
	uint64_t sum = 0;
	bool valid = _view.get_rushnraw_parent().get_sum(get_real_axis_col(), sum, *_view.get_selection_visible_listing(), _ctxt);

	emit refresh_impl_finished(QString("%L1").arg(sum), valid); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVMinCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVMinCellWidget::refresh_impl()
{
	uint64_t min = 0;
	bool valid = _view.get_rushnraw_parent().get_min(get_real_axis_col(), min, *_view.get_selection_visible_listing(), _ctxt);

	emit refresh_impl_finished(QString("%L1").arg(min), valid); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVMaxCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVMaxCellWidget::refresh_impl()
{
	uint64_t max = 0;
	bool valid = _view.get_rushnraw_parent().get_max(get_real_axis_col(), max, *_view.get_selection_visible_listing(), _ctxt);

	emit refresh_impl_finished(QString("%L1").arg(max), valid); // We must go back on the Qt thread to update the GUI
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVAverageCellWidget
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVAverageCellWidget::refresh_impl()
{
	uint64_t avg = 0;
	bool valid = _view.get_rushnraw_parent().get_avg(get_real_axis_col(), avg, *_view.get_selection_visible_listing(), _ctxt);

	emit refresh_impl_finished(QString("%L1").arg(avg), valid); // We must go back on the Qt thread to update the GUI
}
