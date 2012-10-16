/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <QTabWidget>
#include <QTimer>
#include <QWidget>

namespace Picviz
{
class PVSource;
}

namespace PVGuiQt
{

class PVWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_size READ get_tab_size WRITE set_tab_size);

public:
	PVWorkspacesTabWidget(QWidget* parent = 0);
	void remove_workspace(int index);
	int addTab(QWidget* page, const QString & label);
	//void removeTab(int index);
	int count() const;

protected:
	void tabInserted(int index) override;
	void mouseMoveEvent(QMouseEvent* event) override;

signals:
	void workspace_closed(Picviz::PVSource* source);
	void is_empty();

private slots:
	void tabCloseRequested_Slot(int index);
	void start_checking_for_automatic_tab_switch();
	void switch_tab();
	void tab_changed(int index);
	int get_tab_size() { return 50; }
	void set_tab_size(int tab_size_percent);

private:
	QTimer _automatic_tab_switch_timer;
	int _tab_index;
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
