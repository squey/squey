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

public:
	PVWorkspacesTabWidget(QWidget* parent = 0);
	void remove_workspace(int index);

protected:
	void tabInserted(int index) override;

signals:
	void workspace_closed(Picviz::PVSource* source);
	void is_empty();

private slots:
	void tabCloseRequested_Slot(int index);
	void start_checking_for_automatic_tab_switch();
	void switch_tab();

private:
	QTimer _automatic_tab_switch_timer;
	int _tab_index;
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
