#ifndef PVDISPLAYS_PVDISPLAYSCONTAINER_H
#define PVDISPLAYS_PVDISPLAYSCONTAINER_H

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
	virtual void create_view_widget() { }
	virtual void create_view_axis_widget() { }
	virtual void toggle_unique_source_widget() { }
};

}

#endif
