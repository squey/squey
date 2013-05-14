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
#include <QLabel>
#include <QCursor>
#include <QPushButton>

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
   for (int i = 0; i < listing->horizontalHeader()->count(); i++)
      w += listing->columnWidth(i);

   int h = stats->horizontalHeader()->height() + QTABLEWIDGET_OFFSET;
   for (int i = 0; i < stats->verticalHeader()->count(); i++)
      h += stats->rowHeight(i);

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
	_stats_panel->viewport()->setFocusPolicy(Qt::NoFocus);
	_stats_panel->hide();
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
	connect(_listing_view, SIGNAL(resized()), this, SLOT(resize_panel()));
	connect(_listing_view->horizontalScrollBar(), SIGNAL(actionTriggered(int)), this, SLOT(update_scrollbar_position()));

	// Observe selection to handle automatic refresh mode
	PVHive::PVObserverSignal<Picviz::PVSelection>* obs_sel = new PVHive::PVObserverSignal<Picviz::PVSelection>(this);
	Picviz::PVView_sp view_sp = _listing_view->lib_view().shared_from_this();
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *obs_sel);
	obs_sel->connect_refresh(this, SLOT(selection_changed()));

	// Observer axes combination changes
	PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t>* obs_axes_comb = new PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t>;
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& v) { return &v.get_axes_combination().get_axes_index_list(); }, *obs_axes_comb);
	obs_axes_comb->connect_refresh(this, SLOT(axes_comb_changed()));

	init_plugins();

	_stats_panel->verticalHeader()->viewport()->installEventFilter(this);
	_stats_panel->verticalHeader()->setMinimumSectionSize(10);

	resize_panel();
	refresh();
}
void PVGuiQt::PVStatsListingWidget::init_plugins()
{
	init_plugin<__impl::PVUniqueValuesCellWidget>("unique\nvalues");

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
	// Sync tables vertical header width
	int stats_header_width = _stats_panel->verticalHeader()->sizeHint().width();
	int listing_header_width = _listing_view->verticalHeader()->width();

	if (stats_header_width > listing_header_width) {
		_listing_view->verticalHeader()->setFixedWidth(stats_header_width);
		QMetaObject::invokeMethod(_listing_view, "updateGeometries");
	}
	else {
		_stats_panel->verticalHeader()->setFixedWidth(listing_header_width);
		QMetaObject::invokeMethod(_stats_panel, "updateGeometries");
	}

	for (PVCol col=0; col < _stats_panel->columnCount(); col++) {
		for (int row=0; row < _stats_panel->rowCount(); row++) {
			((__impl::PVCellWidgetBase*)_stats_panel->cellWidget(row, col))->auto_refresh();
		}
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
		for (int row=0; row < _stats_panel->rowCount(); row++) {
			for (PVCol col=old_count-1; col < new_count-1; col++) {
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

	_loading_label = new QLabel(this);
	_loading_label->setMovie(get_movie());
	_loading_label->setStyleSheet("QLabel { padding-right: 4px}");
	_loading_label->setVisible(false);

	_autorefresh_icon = new QPushButton();
	_autorefresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_autorefresh_icon->setFlat(true);
	_autorefresh_icon->setStyleSheet("QPushButton { border: none; }");
	_autorefresh_icon->setIcon(_autorefresh_off_pixmap);
	_autorefresh_icon->setFocusPolicy(Qt::NoFocus);
	_autorefresh_icon->setToolTip("Toggle auto refresh");
	_autorefresh_icon->setVisible(false); // Disabled before having a better job handling pipeline
	connect(_autorefresh_icon, SIGNAL(clicked(bool)), this, SLOT(toggle_auto_refresh()));

	connect(this, SIGNAL(refresh_impl_finished(int, bool)), this, SLOT(refreshed(int, bool)));

	_main_layout = new QHBoxLayout();
	_main_layout->setSpacing(2);
	_main_layout->setContentsMargins(2, 0, 2, 0);
	_main_layout->addWidget(_text);
	_main_layout->addStretch(1);
	_main_layout->addWidget(_refresh_icon);
	_main_layout->addWidget(_loading_label);
	_main_layout->addWidget(_autorefresh_icon);

	setLayout(_main_layout);
}

PVGuiQt::PVStatsListingWidget* PVGuiQt::__impl::PVCellWidgetBase::get_panel()
{
	return (PVGuiQt::PVStatsListingWidget*) PVCore::get_qobject_parent_of_type<PVGuiQt::PVStatsListingWidget*>(this);
}

typename PVGuiQt::PVStatsListingWidget::param_t& PVGuiQt::__impl::PVCellWidgetBase::get_params()
{
	PVGuiQt::PVStatsListingWidget* stats_panel = get_panel();
	assert(stats_panel);
	return stats_panel->get_params();
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
	_ctxt->cancel_group_execution();
	if (_thread.joinable()) {
		_thread.join();
	}
	_ctxt->reset();
	_thread_running = false;
}

void PVGuiQt::__impl::PVCellWidgetBase::refresh(bool from_cache /* = false */)
{
	int col = _view.get_real_axis_index(_table->column(_item));

	uint32_t cached_value = get_params()[col].cached_value;
	bool auto_refresh = get_params()[col].auto_refresh;

	if (!from_cache) {
		if (!cached_value) {
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
	else if (from_cache && cached_value) {
		set_valid(cached_value, auto_refresh);
	}
	else {
		set_invalid();
		get_params()[col].cached_value = 0;
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::auto_refresh()
{
	int col = _view.get_real_axis_index(_table->column(_item));

	get_params()[col].cached_value = 0;
	bool auto_refresh = get_params()[col].auto_refresh;
	if (auto_refresh) {
		refresh();
	}
	else {
		set_invalid();
	}
}

void PVGuiQt::__impl::PVCellWidgetBase::refreshed(int value, bool valid)
{
	if (valid) {
		int col = _view.get_real_axis_index(_table->column(_item));
		uint32_t cached_value = value;
		get_params()[col].cached_value = cached_value;
		bool auto_refresh = get_params()[col].auto_refresh;
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

void PVGuiQt::__impl::PVCellWidgetBase::set_valid(uint32_t value, bool auto_refresh)
{
	_text->setText(QString("%L1").arg(value));
	_refresh_icon->setEnabled(false);
	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_on_pixmap : _autorefresh_off_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
	_item->setBackground(QBrush(Qt::NoBrush));
	_valid = true;
}

void PVGuiQt::__impl::PVCellWidgetBase::vertical_header_clicked(int)
{
	refresh();
}

void PVGuiQt::__impl::PVCellWidgetBase::toggle_auto_refresh()
{
	int col = _view.get_real_axis_index(_table->column(_item));
	bool& auto_refresh = get_params()[col].auto_refresh;
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
	_unique_values_dlg_icon->setToolTip("List unique values");
	connect(_unique_values_dlg_icon, SIGNAL(clicked(bool)), this, SLOT(show_unique_values_dlg()));

	_main_layout->addWidget(_unique_values_dlg_icon);
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::refresh_impl()
{
	PVRush::PVNraw::unique_values_t values;
	int col = _view.get_real_axis_index(_table->column(_item));
	bool valid = _view.get_rushnraw_parent().get_unique_values_for_col_with_sel(col, values, *_view.get_selection_visible_listing(), _ctxt);
#if SIMULATE_LONG_COMPUTATION
	for (uint32_t i = 0; i < 10 && !_ctxt->is_group_execution_cancelled(); i++) {
		usleep(300000);
	}
	valid = !_ctxt->is_group_execution_cancelled();
#endif
	emit refresh_impl_finished(values.size(), valid); // We must go back on the Qt thread to update the GUI
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::show_unique_values_dlg()
{
	int col = _view.get_real_axis_index(_table->column(_item));
	PVQNraw::show_unique_values(_view.get_rushnraw_parent(), col, *_view.get_selection_visible_listing(), this);
}
