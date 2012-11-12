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

#include <picviz/PVAD2GView_types.h>

#include <iostream>

namespace Picviz
{
class PVRoot;
}

namespace PVGuiQt
{

class PVCorrelationMenu : public QMenu
{
	Q_OBJECT;

public:
	PVCorrelationMenu(Picviz::PVRoot* root, QWidget* parent = 0);

private slots:
	void create_new_correlation();
	void show_correlation();
	void show_correlation(Picviz::PVAD2GView* correlation);
	void delete_correlation();
	void enable_correlations(bool enable);

private:
	Picviz::PVRoot* _root;

	QAction* _separator_first_correlation;
	QAction* _separator_create_correlation;
	QAction* _action_create_correlation;
};

}

#endif // __PVGUIQT_PVCORRELATIONMENU_H__
