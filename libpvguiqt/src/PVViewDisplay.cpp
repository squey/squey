/**
 * \file PVViewDisplay.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QMenu>

#include <QAbstractScrollArea>
#include <QScrollBar>

#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVWorkspace.h>

#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>
#include <pvhive/waxes/waxes.h>

PVGuiQt::PVViewDisplay::PVViewDisplay(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool can_be_central_widget, PVWorkspace* workspace) :
	QDockWidget((QWidget*)workspace),
	_view(view),
	_workspace(workspace)
{
	setFocusPolicy(Qt::StrongFocus);
	setWidget(view_widget);
	setWindowTitle(name);

	view_widget->installEventFilter(new FocusInEventFilter(this));
	view_widget->setFocusPolicy(Qt::StrongFocus);

	QAbstractScrollArea* scroll_area = dynamic_cast<QAbstractScrollArea*>(view_widget);
	if (scroll_area) {
		scroll_area->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
		scroll_area->horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
	}

	if (view) {
		QColor view_color = view->get_color();
		setStyleSheet(QString("QDockWidget::title {background: %1;} QDockWidget { background: %2;} ").arg(view_color.name()).arg(view_color.name()));
		setFocusPolicy(Qt::StrongFocus);
	}

	if (can_be_central_widget) {
		setAttribute(Qt::WA_DeleteOnClose, true);
	}
}

void PVGuiQt::PVViewDisplay::contextMenuEvent(QContextMenuEvent* event)
{
	bool add_menu = true;
	add_menu &= _workspace->centralWidget() != this;
	add_menu &=  !widget()->isAncestorOf(childAt(event->pos()));

	if (add_menu) {
		QMenu* ctxt_menu = new QMenu(this);
		QAction* switch_action = new QAction(tr("Set as central display"), this);
		connect(switch_action, SIGNAL(triggered(bool)), (QWidget*)_workspace, SLOT(switch_with_central_widget()));
		ctxt_menu->addAction(switch_action);
		ctxt_menu->popup(QCursor::pos());
	}
}

void PVGuiQt::PVViewDisplay::set_current_view()
{
	if (_view) {
		auto source = _view->get_parent<Picviz::PVSource>()->shared_from_this();
		PVHive::call<FUNC(Picviz::PVSource::select_view)>(source, *_view);
	}
}
