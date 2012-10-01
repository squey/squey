/**
 * \file PVViewDisplay.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVVIEWDISPLAY_H__
#define __PVGUIQT_PVVIEWDISPLAY_H__

#include <QDockWidget>
#include <QAction>
#include <QCloseEvent>

namespace PVGuiQt
{

class PVWorkspace;

class PVViewDisplay : public QDockWidget
{
	Q_OBJECT;

	friend PVWorkspace;

protected:
	void closeEvent(QCloseEvent * event)
	{
		emit display_closed();
		QDockWidget::closeEvent(event);
	}

signals:
	void display_closed();

private:
	PVViewDisplay(bool can_be_central_widget = true, QWidget* parent = 0) : QDockWidget(parent)
	{
		if (can_be_central_widget) {

			setAttribute(Qt::WA_DeleteOnClose, true);

			QAction* switch_action = new QAction(tr("Set as central display"), this);

			addAction(switch_action);
			setContextMenuPolicy(Qt::ActionsContextMenu);

			connect(switch_action, SIGNAL(triggered(bool)), parent, SLOT(switch_with_central_widget()));
		}
	}
};

}

#endif // #ifndef __PVGUIQT_PVVIEWDISPLAY_H__
