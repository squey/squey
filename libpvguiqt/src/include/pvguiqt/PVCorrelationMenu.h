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

class PVCorrelationMenu : public QMenu
{
	Q_OBJECT;

public:
	PVCorrelationMenu(QWidget* parent = 0);

signals:
	void correlation_added(const QString & name);
	void correlation_shown(int index);
	void correlation_deleted(int index);
	void correlations_enabled(bool enabled);

private slots:
	void create_new_correlation();
	void show_correlation();
	void show_correlation(int index);
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
