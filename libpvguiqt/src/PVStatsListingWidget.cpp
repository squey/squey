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

// Originally from http://stackoverflow.com/questions/8766633/how-to-determine-the-correct-size-of-a-qtablewidget
static QSize compute_qtablewidget_size(QTableWidget* t)
{
   int w = t->verticalHeader()->width() + t->verticalScrollBar()->width() + 4; // +4 seems to be needed
   for (int i = 0; i < t->columnCount(); i++)
      w += t->columnWidth(i); // seems to include gridline (on my machine)

   int h = t->horizontalHeader()->height() + 4;
   for (int i = 0; i < t->rowCount(); i++)
      h += t->rowHeight(i);

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

	// Observe selection to handle automatic refresh mode
	PVHive::PVObserverSignal<Picviz::PVSelection>* obs_sel = new PVHive::PVObserverSignal<Picviz::PVSelection>(this);
	Picviz::PVView_sp view_sp = _listing_view->lib_view().shared_from_this();
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *obs_sel);
	obs_sel->connect_refresh(this, SLOT(refresh()));

	// Observer axes combination changes
	PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t>* obs_axes_comb = new PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t>;
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& v) { return &v.get_axes_combination().get_axes_index_list(); }, *obs_axes_comb);
	obs_axes_comb->connect_refresh(this, SLOT(axes_comb_changed()));

	init_plugin<__impl::PVUniqueValuesCellWidget>("unique\nvalues");

	for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->setColumnWidth(col, _listing_view->horizontalHeader()->sectionSize(col));
	}

	_stats_panel->verticalHeader()->viewport()->installEventFilter(this);

	refresh();
	resize_panel();
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
	// TODO: refresh when axes combination change... (it would be nice to avoid useless recomputations)
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
	_stats_panel->setMaximumSize(compute_qtablewidget_size(_stats_panel));
}

void PVGuiQt::PVStatsListingWidget::axes_comb_changed()
{
	PVLOG_INFO("axes_comb_changed\n");
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVUniqueValuesCellWidget
 *
 *****************************************************************************/
std::unordered_map<uint32_t, bool> PVGuiQt::__impl::PVUniqueValuesCellWidget::_auto_refresh;

PVGuiQt::__impl::PVUniqueValuesCellWidget::PVUniqueValuesCellWidget(QTableWidget* table, Picviz::PVView const& view, QTableWidgetItem* item) :
	PVCellWidgetBase(table, view, item),
	_refresh_pixmap(QPixmap::fromImage(QImage(":/refresh_small_grey"))),
	_autorefresh_pixmap(QPixmap::fromImage(QImage(":/icon_linked"))),
	_no_autorefresh_pixmap(QPixmap::fromImage(QImage(":/icon_unlinked"))),
	_unique_values_pixmap(QPixmap::fromImage(QImage(":/fileslist_black")))
{
	_auto_refresh.clear();
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

	_refreshed = false;
	if (_auto_refresh[col]) {
		refresh();
	}
	else {
		_item->setBackgroundColor(_invalid_color);
		_text->setText("N/A");
	}
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::refresh() //override
{
	if (!_refreshed) {
		int col = _view.get_real_axis_index(_table->column(_item));
		PVRush::PVNraw::unique_values_t values;
		_view.get_rushnraw_parent().get_unique_values_for_col_with_sel(col, values, *_view.get_selection_visible_listing());
		_text->setText(QString("%1").arg(values.size()));
		_item->setBackground(QBrush(Qt::NoBrush));
		_refreshed = true;
	}
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::vertical_header_clicked(int index)  //override
{
	refresh();
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::toggle_auto_refresh()
{
	int col = _view.get_real_axis_index(_table->column(_item));
	bool auto_refresh = !_auto_refresh[col];
	_auto_refresh[col] = auto_refresh;

	_autorefresh_icon->setIcon(auto_refresh ? _autorefresh_pixmap : _no_autorefresh_pixmap);
	_refresh_icon->setVisible(!auto_refresh);
	refresh(); // or maybe not...
}

void PVGuiQt::__impl::PVUniqueValuesCellWidget::show_unique_values_dlg()
{
	int col = _view.get_real_axis_index(_table->column(_item));
	PVQNraw::show_unique_values(_view.get_rushnraw_parent(), col, *_view.get_selection_visible_listing(), this);
}
