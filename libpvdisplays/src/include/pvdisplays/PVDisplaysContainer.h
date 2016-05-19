/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYSCONTAINER_H
#define PVDISPLAYS_PVDISPLAYSCONTAINER_H

#include <pvbase/general.h>
#include <QMainWindow>

namespace PVDisplays
{

class PVDisplaysContainer : public QMainWindow
{
	Q_OBJECT

  public:
	PVDisplaysContainer(QWidget* w) : QMainWindow(w) {}

  public slots:
	virtual void create_view_widget(QAction* act = nullptr) = 0;
	virtual void create_view_axis_widget(QAction* act = nullptr) = 0;
	virtual void create_view_zone_widget(QAction* act = nullptr) = 0;
	virtual void toggle_unique_source_widget(QAction* act = nullptr) = 0;
};
}

#endif
