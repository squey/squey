/**
 * \file PVCorrelationMenu.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVCORRELATIONMENU_H__
#define __PVGUIQT_PVCORRELATIONMENU_H__

#include <QMenu>
#include <QAction>
#include <QWidget>
#include <QEvent>
#include <QMouseEvent>
#include <QLineEdit>

#include <iostream>

namespace PVGuiQt
{

class PVCorrelationMenu;

namespace __impl
{

class CreateNewCorrelationEventFilter : public QObject
{
public:
	CreateNewCorrelationEventFilter(PVGuiQt::PVCorrelationMenu* menu, QLineEdit* line_edit) : _menu(menu), _line_edit(line_edit) {}

protected:
	bool eventFilter(QObject* watched, QEvent* event);

private:
	PVGuiQt::PVCorrelationMenu* _menu;
	QLineEdit* _line_edit;
};

}


class PVCorrelationMenu : public QMenu
{
	Q_OBJECT;
	friend class __impl::CreateNewCorrelationEventFilter;

public:
	PVCorrelationMenu(QWidget* parent = 0);

protected:
	bool event(QEvent* event)
	{
		if (event->type() == QEvent::MouseButtonRelease) {
			QMouseEvent* mouse_event = (QMouseEvent*) event;
			if (actionGeometry(_action_create_correlation).contains(mouse_event->pos())) {
				create_new_correlation();
				return true;
			}
		}
		return QMenu::event(event);
	}

signals:
	void correlation_added(const QString & name);
	void correlation_shown(int index);
	void correlation_deleted(int index);

private slots:
	void create_new_correlation();
	void show_correlation();
	void delete_correlation();

private:
	void add_new_correlation(const QString & title);
	int get_correlation_index_from_subaction(QAction* action);

private:
	QAction* _separator_first_correlation;
	QAction* _separator_create_correlation;
	QAction* _action_create_correlation;
};

}

#endif // __PVGUIQT_PVCORRELATIONMENU_H__
