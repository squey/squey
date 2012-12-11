/**
 * \file PVStatsListingWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVSTATSLISTINGWIDGET_H__
#define __PVSTATSLISTINGWIDGET_H__

#include <pvguiqt/PVListingView.h>
#include <QWidget>

class QTableWidget;

namespace PVGuiQt
{

class PVStatsListingWidget : public QWidget
{
	Q_OBJECT

public:
	PVStatsListingWidget(PVListingView* listing_view);

private:
	void init();

private slots:
	void toggle_stats_panel_visibility();
	void update_header_width(int column, int old_width, int new_width);
	void refresh();
	void resize_panel();

private:
	PVListingView* _listing_view;
	QTableWidget* _stats_panel;
};

}

#endif // __PVSTATSLISTINGWIDGET_H__
