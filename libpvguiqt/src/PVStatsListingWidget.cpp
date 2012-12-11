/**
 * \file PVStatsListingWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QHeaderView>
#include <QVBoxLayout>
#include <QPushButton>
#include <QScrollBar>
#include <QTableWidget>

#include <pvguiqt/PVStatsListingWidget.h>

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

PVGuiQt::PVStatsListingWidget::PVStatsListingWidget(PVGuiQt::PVListingView* listing_view) : _listing_view(listing_view)
{
	QVBoxLayout* main_layout = new QVBoxLayout();

	_stats_panel = new QTableWidget();
	_stats_panel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_stats_panel->hide();

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
	auto view_sp = _listing_view->lib_view().shared_from_this();
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *obs_sel);
	obs_sel->connect_refresh(this, SLOT(refresh()));

	init();
}

void PVGuiQt::PVStatsListingWidget::init()
{
	_stats_panel->insertRow(0);
	for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
		_stats_panel->insertColumn(col);
	}

	QStringList vertical_headers;
	vertical_headers << "unique\nvalues";
	_stats_panel->setVerticalHeaderLabels(vertical_headers);
	_stats_panel->horizontalHeader()->setStretchLastSection(true);
	_stats_panel->horizontalHeader()->hide();

	refresh();
}

void PVGuiQt::PVStatsListingWidget::refresh()
{
	// TODO: refresh when axes combination change... (it would be nice to avoid useless recomputations)

	for (PVCol col=0; col < _listing_view->horizontalHeader()->count(); col++) {
		PVRush::PVNraw::unique_values_t values;
		Picviz::PVView const& view = _listing_view->lib_view();
		view.get_rushnraw_parent().get_unique_values_for_col_with_sel(col, values, *view.get_selection_visible_listing());
		_stats_panel->setItem(0, view.get_real_axis_index(col), new QTableWidgetItem(QString("%1").arg(values.size())));
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
