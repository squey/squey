#ifndef PVDISPLAYS_PVDISPLAYSCONTAINER_H
#define PVDISPLAYS_PVDISPLAYSCONTAINER_H

#include <pvbase/general.h>
#include <QMainWindow>

namespace PVDisplays {

class PVDisplaysContainer: public QMainWindow
{
	Q_OBJECT

public:
	PVDisplaysContainer(QWidget* w):
		QMainWindow(w)
	{ }

public slots:
	virtual void create_view_widget(QAction* act = nullptr) { PV_UNUSED(act); }
	virtual void create_view_axis_widget(QAction* act = nullptr) { PV_UNUSED(act); }
	virtual void toggle_unique_source_widget(QAction* act = nullptr) { PV_UNUSED(act); }
};

}

#endif
