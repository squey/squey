/**
 * \file PVViewDisplay.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QDockWidget>
#include <QAction>

namespace PVGuiQt
{

class PVWorkspace;

class PVViewDisplay : public QDockWidget
{
	Q_OBJECT;

	friend PVWorkspace;

private:
	PVViewDisplay(QWidget* parent = 0) : QDockWidget(parent)
	{
		QAction* switch_action = new QAction(tr("Set as central display"), this);

		addAction(switch_action);
		setContextMenuPolicy(Qt::ActionsContextMenu);

		connect(switch_action, SIGNAL(triggered(bool)), parent, SLOT(switch_with_central_widget()));
	}
};

}
