/**
 * \file PVViewDisplay.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <iostream>

#include <pvguiqt/PVViewDisplay.h>

#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

PVGuiQt::PVViewDisplay::PVViewDisplay(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool can_be_central_widget /*= true*/, QWidget* parent /*= 0*/) :
	QDockWidget(parent),
	_view(view)
{
	setFocusPolicy(Qt::StrongFocus);
	setWidget(view_widget);
	setWindowTitle(name);

	view_widget->installEventFilter(new FocusInEventFilter(this));
	view_widget->setFocusPolicy(Qt::StrongFocus);

	if (view) {
		QColor view_color = view->get_color();
		setStyleSheet(QString("QDockWidget::title {background: %1;} QDockWidget { background: %2;} ").arg(view_color.name()).arg(view_color.name()));
		setFocusPolicy(Qt::StrongFocus);
	}

	if (can_be_central_widget) {

		setAttribute(Qt::WA_DeleteOnClose, true);

		QAction* switch_action = new QAction(tr("Set as central display"), this);

		addAction(switch_action);
		setContextMenuPolicy(Qt::ActionsContextMenu);

		connect(switch_action, SIGNAL(triggered(bool)), parent, SLOT(switch_with_central_widget()));
	}
}

void PVGuiQt::PVViewDisplay::set_current_view()
{
	if (_view) {
		auto source = _view->get_parent<Picviz::PVSource>()->shared_from_this();
		PVHive::call<FUNC(Picviz::PVSource::select_view)>(source, *_view);
	}
}
