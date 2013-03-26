/**
 * \file PVStatsListingWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QScrollBar>
#include <QLabel>
#include <QCursor>
#include <QPushButton>

#include <pvguiqt/PVStatsListingWidget.h>
#include <pvguiqt/PVQNraw.h>

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
PVGuiQt::PVStatsListingWidget::PVStatsListingWidget(PVGuiQt::PVListingView* listing_view) : _listing_view(listing_view)
{
	QVBoxLayout* main_layout = new QVBoxLayout();

	_stats_panel = new QTableWidget();
	_stats_panel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->hide();
	_stats_panel->viewport()->setFocusPolicy(Qt::NoFocus);

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
	obs_sel->connect_refresh(this, SLOT(refresh()));

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
	if (event->type() == QEvent::Enter) {
		setCursor(QCursor(Qt::PointingHandCursor));
		return true;
	}
	else if (event->type() == QEvent::Leave) {
		setCursor(QCursor(Qt::ArrowCursor));
		return true;
	}
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
 * PVGuiQt::__impl::PVUniqueValuesCellWidget
 *
 *****************************************************************************/
std::unordered_map<uint32_t, std::tuple<bool, uint32_t> > PVGuiQt::__impl::PVUniqueValuesCellWidget::_params;

PVGuiQt::__impl::PVUniqueValuesCellWidget::PVUniqueValuesCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) :
	PVCellWidgetBase(table, view, item),
	_refresh_pixmap(QPixmap::fromImage(QImage(":/refresh_small_grey"))),
	_autorefresh_pixmap(QPixmap::fromImage(QImage(":/icon_linked"))),
	_no_autorefresh_pixmap(QPixmap::fromImage(QImage(":/icon_unlinked"))),
	_unique_values_pixmap(QPixmap::fromImage(QImage(":/fileslist_black")))
{
	_params.clear();
	_text = new QLabel();

	_refresh_icon = new QPushButton();
	_refresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_refresh_icon->setFlat(true);
	_refresh_icon->setStyleSheet("QPushButton { border: none; }");
	_refresh_icon->setIcon(_refresh_pixmap);
	_refresh_icon->setFocusPolicy(Qt::NoFocus);
	_refresh_icon->setToolTip("Refresh");
	connect(_refresh_icon, SIGNAL(clicked(bool)), this, SLOT(refresh()));

	_autorefresh_icon = new QPushButton();
	_autorefresh_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_autorefresh_icon->setFlat(true);
	_autorefresh_icon->setStyleSheet("QPushButton { border: none; }");
	_autorefresh_icon->setIcon(_no_autorefresh_pixmap);
	_autorefresh_icon->setFocusPolicy(Qt::NoFocus);
	_autorefresh_icon->setToolTip("Toggle auto refresh");
	connect(_autorefresh_icon, SIGNAL(clicked(bool)), this, SLOT(toggle_auto_refresh()));

	_unique_values_dlg_icon = new QPushButton();
	_unique_values_dlg_icon->setCursor(QCursor(Qt::PointingHandCursor));
	_unique_values_dlg_icon->setFlat(true);
	_unique_values_dlg_icon->setStyleSheet("QPushButton { border: none; } QPushButton:pressed { padding-left : 0px; }");
	_unique_values_dlg_icon->setIcon(_unique_values_pixmap);
	_unique_values_dlg_icon->setFocusPolicy(Qt::NoFocus);
	_unique_values_dlg_icon->setToolTip("List unique values");
	connect(_unique_values_dlg_icon, SIGNAL(clicked(bool)), this, SLOT(show_unique_values_dlg()));

	QHBoxLayout* main_layout = new QHBoxLayout();
	main_layout->setSpacing(2);
	main_layout->setContentsMargins(2, 0, 2, 0);
	main_layout->addWidget(_text);
	main_layout->addStretch(1);
	main_layout->addWidget(_refresh_icon);
	main_layout->addWidget(_autorefresh_icon);
	main_layout->addWidget(_unique_values_dlg_icon);


	setLayout(main_layout);
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::auto_refresh() //override
{
	int col = _view.get_real_axis_index(_table->column(_item));

	std::get<EParams::CACHED_VALUE>(_params[col]) = 0;
	bool auto_refresh = std::get<EParams::AUTO_REFRESH>(_params[col]);
	if (auto_refresh) {
		refresh();
	}
	else {
		set_invalid();
	}
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::refresh(bool from_cache /* = false */) //override
{
	int col = _view.get_real_axis_index(_table->column(_item));

	uint32_t cached_value = std::get<EParams::CACHED_VALUE>(_params[col]);
	bool auto_refresh = std::get<EParams::AUTO_REFRESH>(_params[col]);

	if (!from_cache) {
		if (!cached_value) {
			PVRush::PVNraw::unique_values_t values;
			_view.get_rushnraw_parent().get_unique_values_for_col_with_sel(col, values, *_view.get_selection_visible_listing());
			cached_value = values.size();
			std::get<EParams::CACHED_VALUE>(_params[col]) = cached_value;
		}
		set_valid(cached_value, auto_refresh);
	}
	else if (from_cache && cached_value) {
		set_valid(cached_value, auto_refresh);
	}
	else {
		set_invalid();
		std::get<EParams::CACHED_VALUE>(_params[col]) = 0;
	}
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::set_invalid()
{
	_item->setBackgroundColor(_invalid_color);
	_text->setText("N/A");
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::set_valid(uint32_t value, bool auto_refresh)
{
	_text->setText(QString("%L1").arg(value));
	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_pixmap : _no_autorefresh_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
	_item->setBackground(QBrush(Qt::NoBrush));
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::vertical_header_clicked(int index)  //override
{
	refresh();
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::toggle_auto_refresh()
{
	int col = _view.get_real_axis_index(_table->column(_item));
	bool& auto_refresh = std::get<EParams::AUTO_REFRESH>(_params[col]);
	auto_refresh = !auto_refresh;

	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_pixmap : _no_autorefresh_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::show_unique_values_dlg()
{
	int col = _view.get_real_axis_index(_table->column(_item));
	PVQNraw::show_unique_values(_view.get_rushnraw_parent(), col, *_view.get_selection_visible_listing(), this);
}
