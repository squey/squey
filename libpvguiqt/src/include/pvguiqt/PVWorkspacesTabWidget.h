/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <QTabWidget>
#include <QTimer>
#include <QTabBar>
#include <QWidget>
#include <QMouseEvent>

namespace Picviz
{
class PVSource;
}

namespace PVGuiQt
{

class PVWorkspaceBase;

class PVTabBar : public QTabBar
{
public:
	QSize tabSizeHint(int index) const
	{
		return QTabBar::tabSizeHint(index);
	}

protected:
	void mouseReleaseEvent(QMouseEvent* event)
	{
		// Tabs are closed on middle button click
		if (event->button() == Qt::MidButton) {
			emit tabCloseRequested(tabAt(event->pos()));
		}
		QTabBar::mouseReleaseEvent(event);
	}
};

class PVWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

public:
	PVWorkspacesTabWidget(QWidget* parent = 0);
	void remove_workspace(int index);
	int addTab(PVWorkspaceBase* page, const QString & label, bool animation = true);
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
	int get_tab_width() const { return 0; }
	void set_tab_width(int tab_width);

private:
	QTimer _automatic_tab_switch_timer;
	int _tab_index;
	PVTabBar* _tab_bar;
	int _tab_width;

	int _workspaces_count = 0;
	int _openworkspaces_count = 0;
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
