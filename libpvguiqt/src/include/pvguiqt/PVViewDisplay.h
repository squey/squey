/**
 * \file PVViewDisplay.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVVIEWDISPLAY_H__
#define __PVGUIQT_PVVIEWDISPLAY_H__

#include <QAction>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFocusEvent>

#include <iostream>

namespace Picviz
{
class PVView;
}

namespace PVGuiQt
{

class PVWorkspace;
class FocusInEventFilter;

class PVViewDisplay : public QDockWidget
{
	Q_OBJECT;

	friend PVWorkspace;
	friend FocusInEventFilter;

protected:
	void closeEvent(QCloseEvent * event)
	{
		emit display_closed();
		QDockWidget::closeEvent(event);
	}

private:
	void set_current_view();

signals:
	void display_closed();

private:
	PVViewDisplay(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool can_be_central_widget = true, QWidget* parent = 0);

private:
	Picviz::PVView* _view;
};

class FocusInEventFilter : public QObject
{
public:
	FocusInEventFilter(PVViewDisplay* parent) : _parent(parent) {}
protected:
	bool eventFilter(QObject* obj, QEvent *event)
	{
		if (event->type() == QEvent::FocusIn) {
			_parent->set_current_view();
			return true;
		}

		return QObject::eventFilter(obj, event);
	}
private:
	PVViewDisplay* _parent;
};

}

#endif // #ifndef __PVGUIQT_PVVIEWDISPLAY_H__
